#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""随机点过程 与「自相关 <-> 功率谱」关系 的最小验证工具。

对应《现阶段学习规划与资源.md》Day 9：
    从 spike statistics 到 neural coding（Gerstner 7.4-7.6：自相关、噪声谱、神经编码）

这个文件只做一件事：把你最近学的随机过程，用可运行的数值实验接到 spike train 的
统计量上。它验证四组结论：

1. Poisson process
   - ISI 服从指数分布 Exp(nu)：均值 1/nu，方差 1/nu^2，CV = 1
   - 固定时长 T 内的发放数服从 Poisson(nu*T)：Fano factor = Var/Mean = 1
   - 叠加（superposition）两个独立 Poisson 过程：速率相加，且仍是 Poisson
   - 稀疏化（thinning）以概率 p 保留：速率变成 p*nu，仍是 Poisson
2. Bernoulli process（离散时间）
   - 每次试验 iid，成功概率 p；成功次数服从 Binomial(N, p)
   - 两次成功之间的等待时间服从 Geometric(p)
   - 当 p = nu*dt 且 dt -> 0 时，Bernoulli 过程收敛到 Poisson process
     （于是 Fano factor = 1 - p -> 1，ISI 的 CV = sqrt(1 - p) -> 1）
3. Renewal process
   - ISI 独立同分布；Gamma(k, theta) 的 CV = 1/sqrt(k)（k = 1 就是 Poisson）
   - 基本更新定理：长期发放率 = 1/E[ISI]，与 ISI 分布的形状无关
   - 条件强度 / 全间隔直方图 h(tau) = sum_{n>=1} f_n(tau)（f_n 为 n 重卷积）
4. Autocorrelation <-> Power spectrum（Wiener-Khinchin）
   - 精确恒等式：P[m] = DFT(circular ACF)[m]（可验证到机器精度）
   - Poisson：自协方差只有一个 tau = 0 的尖峰 -> 功率谱是平的（白噪声）
   - 规则放电（CV < 1）：自相关在 tau ~ 1/nu 出现峰 -> 功率谱出现峰
   - 理论谱 S(f) = nu + 2*nu*Int_0^inf (h(tau) - nu) cos(2*pi*f*tau) dtau

另外附一个最小的 decoding toy：比较「只看 rate」和「保留时间结构」分别能解码出
什么 —— 这正是「平坦功率谱（Poisson）」与「有结构功率谱（规则放电）」的区别。

--------------------------------------------------------------------------
使用方式（先写预测，再运行！每次接受 Agent 代码前先写"我预测会看到什么"）
--------------------------------------------------------------------------
    python decoding_toy.py --demo all
    python decoding_toy.py --demo poisson
    python decoding_toy.py --demo acf_psd --show

参数：
    --demo {all,poisson,bernoulli,renewal,acf_psd,decoding,lif}
    --seed N       随机种子（默认 0）
    --dt SEC       时间 bin 宽度（默认 1e-3）
    --outdir DIR   图片输出目录（默认 <本文件目录>/figures）
    --show         运行后弹出图窗（默认只保存图片，适合无显示器环境）

--------------------------------------------------------------------------
约定（很重要：换一套约定，公式里的系数就会变，所以这里写死一套）
--------------------------------------------------------------------------
    * 时间单位秒，频率单位 Hz，bin 宽 dt。
    * x[n] = 第 n 个 bin 内的 spike 计数（整数），n = 0..N-1，总时长 T = N*dt。
    * 居中信号 xc[n] = x[n] - mean(x)。
    * 自协方差（biased 估计，除以 N）：
          R[k] = (1/N) * sum_n xc[n] * xc[n+k]        (k >= 0)
    * 功率谱用「方差贡献」约定（量纲 = counts^2）：
          P[m] = (1/N) * |X[m]|^2,   X[m] = sum_n xc[n] exp(-2*pi*i*m*n/N)
      满足 sum_m P[m] = sum_n xc[n]^2（Parseval），所以 P[m] 就是频率 m 对总方差
      的贡献。这一条就是 Wiener-Khinchin 的离散形式。
    * 若要和连续理论谱（单位 Hz）比较，用 PSD[f] = P[m] / dt。对 Poisson 过程
      理论值恰好是常数 nu，也就是一条水平线（白噪声）。
"""

import argparse
import math
import os
import sys

import numpy as np

# 无显示器环境下默认用 Agg 后端；--show 时才启用交互后端。
# 必须在 import pyplot 之前决定，所以这里读 sys.argv。
import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (必须在设定后端之后 import)

# ---------------------------------------------------------------------------
# 0. 通用工具：随机种子、检查报告、图片保存
# ---------------------------------------------------------------------------

def make_rng(seed=None):
    """统一使用现代 Generator API（与 projects/lif/lif.py 保持一致）。"""
    return np.random.default_rng(seed)


def _fmt(value):
    """把数值格式化成便于对齐阅读的短字符串。"""
    if value is None:
        return "-"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    if isinstance(value, np.ndarray):
        return np.array2string(np.asarray(value), precision=4, max_line_width=110)
    return str(value)


def relative_error(measured, theory):
    """相对误差；理论值为 0 时退化成绝对误差，避免除零。"""
    theory = float(theory)
    denom = abs(theory) if abs(theory) > 1e-12 else 1.0
    return abs(float(measured) - theory) / denom


class Report:
    """收集每个数值检查的 名称 / 是否通过 / 实测 / 理论 / 容差。

    故意不使用 assert：这是学习工具，希望一次跑完看到全部结果，
    失败之处标 FAIL 并进入小结，而不是在第一个失败处中断。
    """

    def __init__(self):
        self.rows = []

    def check(self, name, ok, measured=None, theory=None, tol=None):
        ok = bool(ok)
        self.rows.append({"name": name, "ok": ok})
        parts = [f"[{'PASS' if ok else 'FAIL'}] {name}"]
        if measured is not None:
            parts.append(f"measured={_fmt(measured)}")
        if theory is not None:
            parts.append(f"theory={_fmt(theory)}")
        if tol is not None:
            parts.append(f"tol={_fmt(tol)}")
        print("  " + "  ".join(parts))
        return ok

    def summary(self):
        total = len(self.rows)
        failed = [row["name"] for row in self.rows if not row["ok"]]
        print("-" * 74)
        if failed:
            print(f"小结：{total - len(failed)}/{total} 通过；未通过 {len(failed)} 条：")
            for name in failed:
                print(f"  - {name}")
        else:
            print(f"小结：{total}/{total} 全部通过")
        print("-" * 74)
        return not failed


def save_figure(fig, outdir, filename, show=False):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  [figure] {path}")
    if show:
        plt.show()
    plt.close(fig)
    return path


def make_axis_grid(n_panels, width=6.2, height=3.6):
    """按面板数量生成 (fig, axes 列表)，方便每个 demo 自己填。"""
    n_cols = 1 if n_panels <= 2 else 2
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width * n_cols, height * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for axis in axes[n_panels:]:
        axis.axis("off")
    return fig, axes


# ---------------------------------------------------------------------------
# 1. 谱工具：spike train -> 自协方差 -> 功率谱
# ---------------------------------------------------------------------------

def spike_times_to_counts(spike_times, dt, n_bins):
    """把发放时刻（秒）转成每个 bin 内的计数（整数序列）。"""
    spike_times = np.asarray(spike_times, dtype=float)
    counts = np.zeros(n_bins, dtype=float)
    index = np.floor(spike_times / dt).astype(int)
    index = index[(index >= 0) & (index < n_bins)]
    np.add.at(counts, index, 1.0)
    return counts


def autocovariance_direct(x, n_lags):
    """按定义直接算 biased 自协方差：R[k] = (1/N) sum_n xc[n]*xc[n+k]。"""
    xc = np.asarray(x, dtype=float)
    xc = xc - xc.mean()
    n = xc.size
    n_lags = min(int(n_lags), n - 1)
    out = np.empty(n_lags + 1)
    for k in range(n_lags + 1):
        out[k] = float(np.dot(xc[: n - k], xc[k:])) / n
    return out


def autocovariance_fft(x, n_lags):
    """用 FFT 算同一个 biased 自协方差。

    零填充到 >= 2N，使线性相关不被循环卷积污染。
    """
    xc = np.asarray(x, dtype=float)
    xc = xc - xc.mean()
    n = xc.size
    nfft = 1 << int(np.ceil(np.log2(max(2 * n, 2))))
    spectrum = np.fft.rfft(xc, n=nfft)
    acf = np.fft.irfft(np.abs(spectrum) ** 2, n=nfft)[: int(n_lags) + 1] / n
    return acf


def circular_autocovariance(x):
    """循环自协方差 c[k] = (1/N) sum_n xc[n]*xc[(n+k) mod N]。

    Wiener-Khinchin 的「精确」形式就建立在这个对象上（见 periodogram_variance_twosided）。
    """
    xc = np.asarray(x, dtype=float)
    xc = xc - xc.mean()
    n = xc.size
    return np.real(np.fft.ifft(np.abs(np.fft.fft(xc)) ** 2)) / n


def periodogram_variance_twosided(x):
    """双边功率谱 P[m] = (1/N)|X[m]|^2，频率索引 m = 0..N-1（含负频率）。

    恒等式（就是 Wiener-Khinchin）：np.fft.fft(circular_autocovariance(x)) == 本函数。
    """
    xc = np.asarray(x, dtype=float)
    xc = xc - xc.mean()
    return np.abs(np.fft.fft(xc)) ** 2 / xc.size


def periodogram_density_twosided(x, dt):
    """双边功率谱密度 S(f) = |X[m]|^2 / (N*dt)，单位 Hz（含正负频率）。

    这是与 Gerstner 7.4「噪声谱」直接对应的约定：对速率为 nu 的 Poisson 过程，
    E[S(f)] = nu，也就是一条高度为 nu 的水平线（白噪声）。

    想看 f >= 0 时用 positive_half() 切片，并注意不要乘 2，
    否则得到的变成单边密度，理论值就会成为 2*nu。
    """
    xc = np.asarray(x, dtype=float)
    xc = xc - xc.mean()
    n = xc.size
    return np.abs(np.fft.fft(xc)) ** 2 / (n * dt)


def positive_half(freqs_one_sided, spectrum_twosided):
    """从双边谱中取出 f >= 0 的那一半（长度与 rfftfreq 对齐）。"""
    return np.asarray(spectrum_twosided, dtype=float)[: np.asarray(freqs_one_sided).size]


def fold_to_onesided(spectrum_twosided):
    """把双边谱折叠成单边谱（内部频点乘 2），只用于对照两种约定（要求 N 为偶数）。"""
    spectrum_twosided = np.asarray(spectrum_twosided, dtype=float)
    n = spectrum_twosided.size
    one_sided = spectrum_twosided[: n // 2 + 1].copy()
    one_sided[1:-1] *= 2.0
    return one_sided


def rfft_frequencies(n_bins, dt):
    return np.fft.rfftfreq(n_bins, d=dt)


# ---------------------------------------------------------------------------
# 2. 过程生成器：Poisson / Bernoulli / renewal
# ---------------------------------------------------------------------------

def simulate_poisson_spike_times(rate, duration, rng, t_start=0.0):
    """齐次 Poisson 过程：ISI 独立同分布 ～ Exp(rate)，累加得到发放时刻。"""
    if rate <= 0:
        return np.empty(0)
    n_probe = int(np.ceil(rate * duration * 1.5)) + 100
    isi = rng.exponential(scale=1.0 / rate, size=n_probe)
    times = t_start + np.cumsum(isi)
    return times[times < t_start + duration]


def simulate_bernoulli_trials(p, n_trials, rng):
    """Bernoulli 过程：每一步独立地以概率 p 发放，返回 0/1 序列。

    这正是 projects/lif/lif.py::generate_background_spikes 里
    `rng.random(...) < rate_hz * dt` 那两行在做的事。
    """
    if not 0.0 <= float(p) <= 1.0:
        raise ValueError("p 必须落在 [0, 1] 区间内")
    return (rng.random(int(n_trials)) < float(p)).astype(float)


def gamma_isi_sampler(shape, scale):
    """Gamma(k=shape, theta=scale) 的 ISI 采样器。CV = 1/sqrt(shape)。

    shape = 1 时退化为 Exp(1/scale)，也就是 Poisson 过程。
    """
    def sampler(n, rng):
        return rng.gamma(shape=float(shape), scale=float(scale), size=int(n))

    return sampler


def simulate_renewal_spike_times(isi_sampler, duration, rng, t_start=0.0):
    """一般更新过程：ISI 独立同分布，累加得到发放时刻。

    先用一小批 pilot 样本估计平均 ISI，再决定需要抽多少个间隔。
    """
    pilot = isi_sampler(2000, rng)
    mean_isi = float(np.mean(pilot))
    n_needed = int(np.ceil(duration / mean_isi * 1.2)) + 100
    isi = isi_sampler(n_needed, rng)
    times = t_start + np.cumsum(isi)
    return times[times < t_start + duration]


def isi_statistics(spike_times):
    """从发放时刻算 ISI 的个数 / 均值 / 方差 / CV。"""
    isi = np.diff(np.asarray(spike_times, dtype=float))
    if isi.size < 2:
        return {"n": int(isi.size), "mean": np.nan, "var": np.nan,
                "cv": np.nan, "isi": isi}
    mean = float(isi.mean())
    var = float(isi.var(ddof=0))
    return {"n": int(isi.size), "mean": mean, "var": var,
            "cv": float(np.sqrt(var) / mean) if mean > 0 else np.nan,
            "isi": isi}


def fano_factor(counts):
    """Fano factor = Var(counts) / Mean(counts)。Poisson 过程恒等于 1。"""
    counts = np.asarray(counts, dtype=float)
    mean = float(counts.mean())
    if mean == 0:
        return np.nan
    return float(counts.var(ddof=0)) / mean


def counts_in_windows(spike_times, window_s, duration):
    """把长度为 duration 的记录切成互不重叠的窗口，返回每个窗口内的计数。

    对 Poisson 过程，互不重叠窗口里的计数彼此独立，所以这既是
    「固定窗口计数 ~ Poisson(nu*T)」的直接检验，也是估 Fano 的稳当做法。
    """
    edges = np.arange(0.0, duration + window_s, window_s)
    return np.histogram(np.asarray(spike_times, dtype=float), bins=edges)[0]


def all_interval_histogram(spike_times, tau_max, n_bins):
    """估计条件强度 h(tau)：给定一个 spike，tau 之后出现 spike 的密度。

    做法：对每个 spike 只看它「之后」的 spike（j > i），所以不含 tau = 0 的自配对，
    也就自动去掉了 nu*delta(tau) 那一项。

    只把满足 t_i <= T - tau_max 的 spike 当作原点，避免右端截断带来的偏差；
    最后除以 (有效原点个数 * d_tau) 得到密度（单位 1/s）。
    """
    t = np.sort(np.asarray(spike_times, dtype=float))
    d_tau = tau_max / n_bins
    edges = np.arange(n_bins + 1) * d_tau
    origin_limit = t[-1] - tau_max
    origins = t[t <= origin_limit]
    if origins.size == 0:
        return (edges[:-1] + d_tau / 2), np.full(n_bins, np.nan)

    # 向量化：先对每个原点求出 [lo, hi) 的未来 spike 下标区间，
    # 再用 repeat 把所有 (t_j - t_i) 展开成一个数组，避免 Python 循环。
    lo = np.searchsorted(t, origins, side="right")
    hi = np.searchsorted(t, origins + tau_max, side="right")
    counts = hi - lo
    total = int(counts.sum())
    if total == 0:
        return (edges[:-1] + d_tau / 2), np.zeros(n_bins)

    origin_index = np.repeat(np.arange(origins.size), counts)
    start_offset = np.repeat(np.cumsum(counts) - counts, counts)
    spike_index = lo[origin_index] + (np.arange(total) - start_offset)
    tau_values = t[spike_index] - origins[origin_index]

    hist, _ = np.histogram(tau_values, bins=edges)
    return (edges[:-1] + d_tau / 2), hist / (origins.size * d_tau)


# ---------------------------------------------------------------------------
# 3. 理论值（只用 stdlib 的 math.gamma，不引入 scipy，与既有代码一致）
# ---------------------------------------------------------------------------

def poisson_pmf(k, lam):
    """Poisson(lam) 的概率质量函数，k 为非负整数数组。"""
    k = np.asarray(k, dtype=float)
    lam = float(lam)
    with np.errstate(divide="ignore"):
        log_p = -lam + k * np.log(lam) - np.array(
            [math.lgamma(i + 1.0) for i in k]
        )
    return np.exp(log_p)


def gamma_pdf(t, shape, scale):
    """Gamma(k=shape, theta=scale) 的概率密度。"""
    t = np.asarray(t, dtype=float)
    k = float(shape)
    theta = float(scale)
    out = np.zeros_like(t)
    mask = t > 0
    out[mask] = (t[mask] ** (k - 1.0) * np.exp(-t[mask] / theta)
                 / (math.gamma(k) * theta ** k))
    return out


def renewal_density_theory(shape, scale, tau_max, n_bins, pad=3.0):
    """h(tau) = sum_{n>=1} f_n(tau)，f_1 = ISI 密度，f_n = f_{n-1} 卷积 f_1。

    在加长网格（长度 pad 倍）上做线性卷积，避免右端截断污染我们关心的区间。
    返回 (tau, h)，长度都是 n_bins。
    """
    d_tau = tau_max / n_bins
    n_pad = int(n_bins * (1.0 + pad))
    tau_pad = (np.arange(n_pad) + 0.5) * d_tau
    f1 = gamma_pdf(tau_pad, shape, scale)
    total = f1.sum() * d_tau
    if total <= 0:
        raise ValueError("ISI 密度在给定网格上没有质量，请检查 shape/scale/tau_max")
    f1 = f1 / total                       # 归一化，消掉离散化误差

    mean_isi = float(shape) * float(scale)
    cv = 1.0 / math.sqrt(float(shape))
    n_terms = int(math.ceil(tau_max / mean_isi)) + int(math.ceil(10.0 * cv)) + 20
    h = np.zeros(n_pad)
    f_n = f1.copy()
    for _ in range(n_terms):
        h += f_n
        f_n = np.convolve(f_n, f1)[:n_pad] * d_tau
        if f_n.sum() * d_tau < 1e-12:
            break
    tau = tau_pad[:n_bins]
    return tau, h[:n_bins]


def renewal_psd_theory(shape, scale, freqs, tau_max, n_bins):
    """更新过程的连续理论功率谱（单位 Hz）：

        S(f) = nu + 2*nu * Int_0^inf (h(tau) - nu) cos(2*pi*f*tau) dtau,  f > 0

    其中 nu = 1/E[ISI]。当 h = nu（Poisson）时第二项为 0，得 S(f) = nu，
    也就是一条平坦的水平线（白噪声）。
    """
    tau, h = renewal_density_theory(shape, scale, tau_max, n_bins)
    nu = 1.0 / (float(shape) * float(scale))
    g = h - nu
    d_tau = tau[1] - tau[0]
    freqs = np.asarray(freqs, dtype=float)
    cos_mat = np.cos(2.0 * np.pi * np.outer(freqs, tau))
    integral = cos_mat.dot(g) * d_tau
    return nu + 2.0 * nu * integral


def empirical_exponential_survival(values, t_grid, rate):
    """经验生存函数 P(V > t) 与指数分布理论值 exp(-rate*t) 的对比。"""
    values = np.asarray(values, dtype=float)
    empirical = np.array([float((values > t).mean()) for t in t_grid])
    theory = np.exp(-float(rate) * np.asarray(t_grid, dtype=float))
    return empirical, theory


# ---------------------------------------------------------------------------
# 4. Demo：Poisson process
# ---------------------------------------------------------------------------

def print_prediction_prompt(lines):
    """按 B2.3b 的规矩：接受代码前先写预测。"""
    print("  [先写预测] " + lines[0])
    for line in lines[1:]:
        print("             " + line)


def demo_poisson(report, outdir, dt, seed, show):
    print("=" * 74)
    print("Demo 1 | Poisson process：ISI ~ Exp(nu)、计数 ~ Poisson(nu*T)、Fano = 1")
    print("=" * 74)
    print_prediction_prompt([
        "ISI 直方图会贴合 Exp(30) 的密度曲线吗？",
        "1 秒窗口内的计数分布，均值和方差会各是多少？",
        "把窗口从 50 ms 拉到 10 s，Fano factor 会怎么变？",
    ])

    rng = make_rng(seed)
    rate = 30.0
    duration = 200.0
    times = simulate_poisson_spike_times(rate, duration, rng)
    stats = isi_statistics(times)
    measured_rate = times.size / duration
    print(f"  参数：nu = {rate} Hz，时长 = {duration} s，dt = {dt} s")
    print(f"  实测：发放数 = {times.size}，速率 = {measured_rate:.4g} Hz")

    # (1) 速率与 ISI 的三个矩
    report.check("速率 = 发放数/时长 ≈ nu", relative_error(measured_rate, rate) < 0.03,
                 measured_rate, rate, 0.03)
    report.check("E[ISI] = 1/nu", relative_error(stats["mean"], 1 / rate) < 0.03,
                 stats["mean"], 1 / rate, 0.03)
    report.check("Var[ISI] = 1/nu^2", relative_error(stats["var"], 1 / rate ** 2) < 0.10,
                 stats["var"], 1 / rate ** 2, 0.10)
    report.check("CV[ISI] = 1", abs(stats["cv"] - 1.0) < 0.03,
                 stats["cv"], 1.0, 0.03)

    # (2) 固定窗口内的计数 ~ Poisson(nu*T)，于是 Fano = 1
    #     要把 Fano 估准，就用一段很长的记录切成互不重叠的窗口
    #     （对 Poisson 过程，这些窗口里的计数天然独立）
    long_duration = 20000.0
    long_times = simulate_poisson_spike_times(rate, long_duration,
                                              make_rng(seed + 7))
    window_sizes_s = [0.05, 0.5, 1.0, 2.0, 10.0]
    fano_by_window = []
    for window_s in window_sizes_s:
        fano_by_window.append(
            fano_factor(counts_in_windows(long_times, window_s, long_duration)))
    for window_s, fano in zip(window_sizes_s, fano_by_window):
        print(f"  窗口 {window_s:>5.2f} s：Fano = {fano:.4f}（理论 1）")
    report.check("Fano factor 与窗口长度无关，恒为 1",
                 max(abs(f - 1.0) for f in fano_by_window) < 0.08,
                 max(abs(f - 1.0) for f in fano_by_window), 0.0, 0.08)

    counts_1s = counts_in_windows(long_times, 1.0, long_duration)
    report.check("1 s 窗口计数的均值 = nu*T",
                 relative_error(counts_1s.mean(), rate * 1.0) < 0.05,
                 float(counts_1s.mean()), rate * 1.0, 0.05)
    report.check("1 s 窗口计数的方差 = nu*T（Poisson 等均值等方差）",
                 relative_error(counts_1s.var(ddof=0), rate * 1.0) < 0.15,
                 float(counts_1s.var(ddof=0)), rate * 1.0, 0.15)

    # (3) bin 内的计数本身也是 Poisson(nu*dt)
    binned = spike_times_to_counts(times, dt, int(round(duration / dt)))
    report.check("bin 内计数均值 = nu*dt",
                 relative_error(binned.mean(), rate * dt) < 0.03,
                 float(binned.mean()), rate * dt, 0.03)
    report.check("bin 内计数方差 = nu*dt（等价于 Fano = 1）",
                 relative_error(binned.var(ddof=0), rate * dt) < 0.05,
                 float(binned.var(ddof=0)), rate * dt, 0.05)

    # (4) 叠加：两个独立 Poisson 的并集仍是 Poisson，速率相加
    #     这里改用长记录，把速率估计的随机误差压到 < 1%
    rate_b = 20.0
    long_times_b = simulate_poisson_spike_times(rate_b, long_duration,
                                                make_rng(seed + 1))
    merged = np.sort(np.concatenate([long_times, long_times_b]))
    merged_stats = isi_statistics(merged)
    merged_rate = merged.size / long_duration
    print(f"  叠加 nu_a = {rate:g} + nu_b = {rate_b:g} -> 实测 {merged_rate:.4g} Hz")
    report.check("叠加后速率 = nu_a + nu_b",
                 relative_error(merged_rate, rate + rate_b) < 0.01,
                 merged_rate, rate + rate_b, 0.01)
    report.check("叠加后仍是 Poisson（CV = 1）",
                 abs(merged_stats["cv"] - 1.0) < 0.01,
                 merged_stats["cv"], 1.0, 0.01)

    # (5) 稀疏化：以概率 p 独立保留，速率变成 p*nu，且仍是 Poisson
    keep_probability = 0.3
    thinned = long_times[rng.random(long_times.size) < keep_probability]
    thinned_stats = isi_statistics(thinned)
    thinned_rate = thinned.size / long_duration
    print(f"  稀疏化 p = {keep_probability:g} -> 实测 {thinned_rate:.4g} Hz")
    report.check("稀疏化后速率 = p*nu",
                 relative_error(thinned_rate, keep_probability * rate) < 0.01,
                 thinned_rate, keep_probability * rate, 0.01)
    report.check("稀疏化后仍是 Poisson（CV = 1）",
                 abs(thinned_stats["cv"] - 1.0) < 0.02,
                 thinned_stats["cv"], 1.0, 0.02)

    # ---- 图 1：ISI / 计数 / Fano（图内文字用英文，与 projects/lif、projects/dynamics 一致）----
    tau_grid = np.linspace(0, 4.0 / rate, 200)
    fig, axes = make_axis_grid(3)
    axes[0].hist(stats["isi"], bins=60, density=True, alpha=0.65,
                 label="ISI histogram")
    axes[0].plot(tau_grid, rate * np.exp(-rate * tau_grid), "r-", lw=2,
                 label=f"Exp(nu={rate:g})")
    axes[0].set_xlabel("ISI (s)")
    axes[0].set_ylabel("density")
    axes[0].set_ylim(bottom=0)
    axes[0].set_title("ISI ~ exponential")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    k_max = int(counts_1s.max())
    k_grid = np.arange(0, k_max + 1)
    axes[1].hist(counts_1s, bins=np.arange(-0.5, k_max + 1.5), density=True,
                 alpha=0.65, label="counts / 1 s")
    axes[1].plot(k_grid, poisson_pmf(k_grid, rate), "r-o", ms=3,
                 label=f"Poisson({rate:g})")
    axes[1].set_xlabel("counts / 1 s")
    axes[1].set_ylabel("probability")
    axes[1].set_title("window counts ~ Poisson")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].semilogx(np.array(window_sizes_s) * 1e3, fano_by_window, "o-",
                     label="measured Fano")
    axes[2].axhline(1.0, color="r", ls="--", label="theory = 1")
    axes[2].set_xlabel("window length (ms)")
    axes[2].set_ylabel("Fano factor")
    axes[2].set_title("Fano is flat in window size")
    axes[2].legend()
    axes[2].grid(alpha=0.3, which="both")

    fig.suptitle("Demo 1 | Poisson process", fontsize=13)
    fig.tight_layout()
    save_figure(fig, outdir, "poisson_isi_counts_fano.png", show)

    # ---- 图 2：噪声谱。Poisson 的噪声谱是平的 nu（白噪声，双边密度约定）----
    n_trials = 40
    trial_duration = duration / n_trials
    trial_bins = int(round(trial_duration / dt))
    freqs = rfft_frequencies(trial_bins, dt)
    psd_stack = np.empty((n_trials, freqs.size))
    for trial in range(n_trials):
        trial_times = simulate_poisson_spike_times(
            rate, trial_duration, make_rng(seed + 100 + trial))
        trial_counts = spike_times_to_counts(trial_times, dt, trial_bins)
        psd_stack[trial] = positive_half(
            freqs, periodogram_density_twosided(trial_counts, dt))
    psd_mean = psd_stack.mean(axis=0)
    report.check("Poisson 的噪声谱是平的，水平 = nu（白噪声）",
                 relative_error(psd_mean[1:].mean(), rate) < 0.05,
                 float(psd_mean[1:].mean()), rate, 0.05)

    # 顺带把「双边密度」和「单边折叠」两种约定的区别验证清楚，
    # 免得以后看别人的功率谱图把高度读错一倍
    probe_counts = spike_times_to_counts(
        simulate_poisson_spike_times(rate, trial_duration, make_rng(seed + 999)),
        dt, trial_bins)
    density_two_sided = periodogram_density_twosided(probe_counts, dt)
    density_one_sided = fold_to_onesided(density_two_sided)
    report.check("单边谱 = 双边谱在内部频点乘 2（f=0 与 Nyquist 不乘）",
                 np.allclose(density_one_sided[1:-1],
                             2.0 * density_two_sided[1: density_one_sided.size - 1]),
                 "见公式", "x2", None)

    fig, axes = make_axis_grid(2)
    for trial in range(6):
        axes[0].semilogy(freqs[1:], psd_stack[trial][1:], lw=0.8, alpha=0.5)
    axes[0].semilogy(freqs[1:], psd_mean[1:], "k-", lw=2,
                     label=f"mean of {n_trials} trials")
    axes[0].axhline(rate, color="r", ls="--", label=f"theory nu = {rate:g} Hz")
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("PSD (Hz)")
    axes[0].set_title("single periodogram is noisy; average it")
    axes[0].legend()
    axes[0].grid(alpha=0.3, which="both")

    axes[1].plot(freqs[1:], psd_stack[0][1:], lw=0.8, alpha=0.6,
                 label="single realization")
    axes[1].axhline(rate, color="r", ls="--", label="theory nu")
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("PSD (Hz)")
    axes[1].set_title("white noise: no characteristic frequency")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")
    fig.suptitle("Demo 1 | Poisson noise spectrum", fontsize=13)
    fig.tight_layout()
    save_figure(fig, outdir, "poisson_power_spectrum.png", show)

# ---------------------------------------------------------------------------
# 5. Demo：Bernoulli process 与 Bernoulli -> Poisson 极限
# ---------------------------------------------------------------------------

def binomial_pmf(k, n, p):
    """Binomial(n, p) 的概率质量函数（用 lgamma 避免大组合数溢出）。"""
    k = np.asarray(k, dtype=float)
    n = float(n)
    p = float(p)
    out = np.zeros_like(k)
    valid = (k >= 0) & (k <= n)
    kk = k[valid]
    log_c = (np.array([math.lgamma(n + 1.0)]) - np.array(
        [math.lgamma(i + 1.0) for i in kk])
        - np.array([math.lgamma(n - i + 1.0) for i in kk]))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = log_c + kk * np.log(p) + (n - kk) * np.log1p(-p)
    out[valid] = np.exp(log_p)
    return out


def geometric_pmf(k, p):
    """Geometric(p)：等待时间 = 到第一次成功为止的试验次数 (k = 1, 2, ...)。"""
    k = np.asarray(k, dtype=float)
    out = np.zeros_like(k)
    valid = k >= 1
    out[valid] = (1.0 - p) ** (k[valid] - 1.0) * p
    return out


def success_positions(trials):
    """Bernoulli 0/1 序列中成功所在的试验编号（0-based）。"""
    return np.flatnonzero(np.asarray(trials) > 0.5)


def demo_bernoulli(report, outdir, dt, seed, show):
    print("=" * 74)
    print("Demo 2 | Bernoulli process：Binomial / Geometric / Bernoulli -> Poisson")
    print("=" * 74)
    print_prediction_prompt([
        "每块试验的成功次数，Fano factor 会是 1 还是 1-p？",
        "两次成功之间等待的试验次数，CV 会是 1 还是 sqrt(1-p)？",
        "让 p = nu*dt 越来越小，这两条会往 Poisson 的 1 收敛吗？",
    ])

    rng = make_rng(seed)
    p = 0.05
    n_trials = 2_000_000
    trials = simulate_bernoulli_trials(p, n_trials, rng)

    report.check("单步均值 = p", relative_error(trials.mean(), p) < 0.02,
                 float(trials.mean()), p, 0.02)
    report.check("单步方差 = p*(1-p)",
                 relative_error(trials.var(ddof=0), p * (1 - p)) < 0.05,
                 float(trials.var(ddof=0)), p * (1 - p), 0.05)

    # (1) 每块 n_block 次试验的成功次数 ~ Binomial(n_block, p)
    n_block = 200
    n_blocks = n_trials // n_block
    block_counts = trials[: n_blocks * n_block].reshape(n_blocks, n_block).sum(axis=1)
    report.check("分块计数的均值 = n_block*p",
                 relative_error(block_counts.mean(), n_block * p) < 0.03,
                 float(block_counts.mean()), n_block * p, 0.03)
    report.check("分块计数的方差 = n_block*p*(1-p)",
                 relative_error(block_counts.var(ddof=0), n_block * p * (1 - p)) < 0.05,
                 float(block_counts.var(ddof=0)), n_block * p * (1 - p), 0.05)
    report.check("分块计数的 Fano = 1-p（不是 1！）",
                 abs(fano_factor(block_counts) - (1 - p)) < 0.05,
                 fano_factor(block_counts), 1 - p, 0.05)

    # (2) 两次成功之间的等待试验次数 ~ Geometric(p)
    positions = success_positions(trials)
    isi_trials = np.diff(positions).astype(float)
    report.check("Geometric 的均值 = 1/p",
                 relative_error(isi_trials.mean(), 1 / p) < 0.03,
                 float(isi_trials.mean()), 1 / p, 0.03)
    report.check("Geometric 的方差 = (1-p)/p^2",
                 relative_error(isi_trials.var(ddof=0), (1 - p) / p ** 2) < 0.05,
                 float(isi_trials.var(ddof=0)), (1 - p) / p ** 2, 0.05)
    report.check("Geometric 的 CV = sqrt(1-p)",
                 abs(float(isi_trials.std() / isi_trials.mean()) - math.sqrt(1 - p)) < 0.05,
                 float(isi_trials.std() / isi_trials.mean()),
                 math.sqrt(1 - p), 0.05)

    # (3) Bernoulli -> Poisson 极限：固定 nu，让 p = nu*dt -> 0
    rate = 20.0
    total_time = 400.0
    dt_list = [1e-2, 1e-3, 1e-4]
    t_grid = np.linspace(0.0, 3.0 / rate, 60)
    print(f"  固定 nu = {rate:g} Hz、T = {total_time:g} s，令 p = nu*dt 逐步变小：")
    print(f"  {'dt':>8} {'p':>11} {'Fano':>9} {'1-p':>9} "
          f"{'CV':>9} {'sqrt(1-p)':>10} {'max|S-exp|':>11}")
    p_values, fano_measured, cv_measured, survival_deviation = [], [], [], []
    survival_curves = {}
    for dt_b in dt_list:
        p_b = rate * dt_b
        n_b = int(round(total_time / dt_b))
        trials_b = simulate_bernoulli_trials(p_b, n_b, make_rng(seed + 500))
        block_bins = max(1, int(round(0.1 / dt_b)))       # 0.1 s 一块
        n_blocks_b = n_b // block_bins
        block_counts_b = trials_b[: n_blocks_b * block_bins].reshape(
            n_blocks_b, block_bins).sum(axis=1)
        fano_b = fano_factor(block_counts_b)
        isi_b = np.diff(success_positions(trials_b)).astype(float) * dt_b
        cv_b = float(isi_b.std() / isi_b.mean())
        empirical, theory = empirical_exponential_survival(isi_b, t_grid, rate)
        dev = float(np.max(np.abs(empirical - theory)))
        if dt_b == dt_list[-1]:
            survival_curves["empirical"] = empirical
            survival_curves["theory"] = theory
        p_values.append(p_b)
        fano_measured.append(fano_b)
        cv_measured.append(cv_b)
        survival_deviation.append(dev)
        print(f"  {dt_b:>8.0e} {p_b:>11.4g} {fano_b:>9.4f} {1 - p_b:>9.4f} "
              f"{cv_b:>9.4f} {math.sqrt(1 - p_b):>10.4f} {dev:>11.4g}")

    report.check("Fano = 1-p（Binomial 的签名）",
                 max(relative_error(f, 1 - p_b)
                     for f, p_b in zip(fano_measured, p_values)) < 0.05,
                 "见上表", "1-p", 0.05)
    report.check("dt*ISI 的 CV = sqrt(1-p)",
                 max(abs(c - math.sqrt(1 - p_b))
                     for c, p_b in zip(cv_measured, p_values)) < 0.03,
                 "见上表", "sqrt(1-p)", 0.03)
    report.check("两个偏差都随 p -> 0 单调递减（收敛到 Poisson）",
                 all(fano_measured[i + 1] > fano_measured[i]
                     for i in range(len(fano_measured) - 1))
                 and all(cv_measured[i + 1] > cv_measured[i]
                         for i in range(len(cv_measured) - 1)),
                 [round(f, 4) for f in fano_measured], 1.0, None)
    report.check("dt*ISI 的分布收敛到 Exp(nu)（生存函数最大偏差递减）",
                 all(survival_deviation[i + 1] <= survival_deviation[i] + 0.01
                     for i in range(len(survival_deviation) - 1)),
                 [round(d, 4) for d in survival_deviation], 0.0, None)

    # ---- 图（图内文字用英文）----
    fig, axes = make_axis_grid(4)
    k_max = int(block_counts.max())
    k_grid = np.arange(0, k_max + 1)
    axes[0].hist(block_counts, bins=np.arange(-0.5, k_max + 1.5), density=True,
                 alpha=0.65, label=f"counts per {n_block} trials")
    axes[0].plot(k_grid, binomial_pmf(k_grid, n_block, p), "r-o", ms=3,
                 label=f"Binomial({n_block}, {p:g})")
    axes[0].set_xlabel("successes per block")
    axes[0].set_ylabel("probability")
    axes[0].set_title("block counts ~ Binomial")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    m_max = int(min(isi_trials.max(), 8 / p))
    m_grid = np.arange(1, m_max + 1)
    axes[1].hist(isi_trials, bins=np.arange(0.5, m_max + 1.5), density=True,
                 alpha=0.65, label="waiting time")
    axes[1].plot(m_grid, geometric_pmf(m_grid, p), "r-o", ms=3,
                 label=f"Geometric({p:g})")
    axes[1].set_xlabel("trials until next success")
    axes[1].set_ylabel("probability")
    axes[1].set_title("waiting time ~ Geometric")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].loglog(p_values, np.abs(np.array(fano_measured) - 1.0), "o-",
                   label="|Fano - 1|")
    axes[2].loglog(p_values, np.abs(np.array(cv_measured) - 1.0), "s-",
                   label="|CV - 1|")
    axes[2].loglog(p_values, p_values, "k--", lw=1, label="slope 1 guide")
    axes[2].set_xlabel("p = nu*dt")
    axes[2].set_ylabel("deviation from Poisson")
    axes[2].set_title("error vanishes linearly in p")
    axes[2].legend()
    axes[2].grid(alpha=0.3, which="both")

    axes[3].plot(t_grid * 1e3, survival_curves["empirical"], "o-",
                 label=f"measured, dt = {dt_list[-1]:g} s")
    axes[3].plot(t_grid * 1e3, survival_curves["theory"], "r--",
                 label="Exp(nu) theory")
    axes[3].set_xlabel("t (ms)")
    axes[3].set_ylabel("P(ISI > t)")
    axes[3].set_title("matches exponential for small dt")
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    fig.suptitle("Demo 2 | Bernoulli process and its Poisson limit", fontsize=13)
    fig.tight_layout()
    save_figure(fig, outdir, "bernoulli_binomial_geometric.png", show)

# ---------------------------------------------------------------------------
# 6. Demo：Renewal process
# ---------------------------------------------------------------------------

def simulate_alternating_burst_spike_times(n_pairs, short_isi, long_isi):
    """非更新过程的对照：短-长交替（burst），ISI 之间存在强负相关。

    更新过程的定义要求 ISI 独立同分布。这个对照用来说明「更新性」并不是
    自动成立的：只要去测相邻 ISI 的 lag-1 相关，马上就露馅。
    """
    isi = np.empty(2 * int(n_pairs))
    isi[0::2] = float(short_isi)
    isi[1::2] = float(long_isi)
    return np.cumsum(isi)


def lag1_autocorrelation(values):
    """相邻样本的 Pearson 相关系数（更新过程应接近 0）。

    用整段序列的均值中心化，这是 lag-1 自相关的标准定义；
    对长度 n 的序列有约 -1/n 的负偏差，n 很大时可忽略。
    """
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return np.nan
    centered = values - values.mean()
    a = centered[:-1]
    b = centered[1:]
    denominator = np.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    return float(np.dot(a, b) / denominator) if denominator > 0 else np.nan


def demo_renewal(report, outdir, dt, seed, show):
    print("=" * 74)
    print("Demo 3 | Renewal process：CV = 1/sqrt(k)、更新性、h(tau) = sum f_n(tau)")
    print("=" * 74)
    print_prediction_prompt([
        "Gamma(k) 的 CV 会等于 1/sqrt(k) 吗？k=1 是不是特殊情形？",
        "相邻两个 ISI 之间会有相关吗（更新性到底约束了什么）？",
        "ISI 越规则，全间隔直方图会不会在 tau ~ 1/nu 处冒出峰？",
    ])

    rate = 20.0
    duration = 2000.0
    shapes = [0.5, 1.0, 2.0, 5.0, 20.0]
    mean_isi = 1.0 / rate
    print(f"  固定速率 nu = {rate:g} Hz（E[ISI] = {mean_isi:g} s），改变 Gamma 的 shape k：")
    print(f"  {'k':>6} {'CV 实测':>9} {'1/sqrt(k)':>10} {'速率实测':>10} "
          f"{'lag-1 corr':>11}")
    measured_cv, measured_rate, lag1_values = [], [], []
    spike_times_by_shape = {}
    for k in shapes:
        scale = mean_isi / k                     # 使 E[ISI] = k*scale = 1/rate
        times = simulate_renewal_spike_times(gamma_isi_sampler(k, scale),
                                             duration, make_rng(seed + int(k * 10)))
        stats = isi_statistics(times)
        rate_k = times.size / duration
        lag1 = lag1_autocorrelation(stats["isi"])
        measured_cv.append(stats["cv"])
        measured_rate.append(rate_k)
        lag1_values.append(lag1)
        spike_times_by_shape[k] = times
        print(f"  {k:>6.1f} {stats['cv']:>9.4f} {1 / math.sqrt(k):>10.4f} "
              f"{rate_k:>10.4f} {lag1:>11.4f}")

    report.check("CV = 1/sqrt(k)",
                 max(relative_error(cv, 1 / math.sqrt(k))
                     for cv, k in zip(measured_cv, shapes)) < 0.05,
                 [round(c, 4) for c in measured_cv],
                 [round(1 / math.sqrt(k), 4) for k in shapes], 0.05)
    report.check("基本更新定理：速率 = 1/E[ISI]，与 ISI 分布形状无关",
                 max(relative_error(r, rate) for r in measured_rate) < 0.02,
                 [round(r, 4) for r in measured_rate], rate, 0.02)
    report.check("更新性：相邻 ISI 的 lag-1 相关 ≈ 0",
                 max(abs(v) for v in lag1_values) < 0.02,
                 [round(v, 4) for v in lag1_values], 0.0, 0.02)

    # 对照：短-长交替的 burst 过程不是更新过程
    burst_times = simulate_alternating_burst_spike_times(
        20000, 0.01, 0.09)
    burst_lag1 = lag1_autocorrelation(isi_statistics(burst_times)["isi"])
    report.check("对照：短-长交替的 burst 过程 lag-1 相关 ≈ -1（非更新）",
                 burst_lag1 < -0.9, burst_lag1, -1.0, None)

    # ---- 全间隔直方图 vs 理论 h(tau) = sum_{n>=1} f_n(tau) ----
    tau_max = 6.0 * mean_isi
    n_tau_bins = 600
    d_tau = tau_max / n_tau_bins
    print(f"  条件强度 h(tau)：在 tau in [{mean_isi:g}, {tau_max:g}] s 上比较"
          f"（跳过 tau -> 0 的发散区）")
    print(f"  {'k':>6} {'相对RMS误差':>12} {'h(tau_max) 实测':>16} "
          f"{'h(tau_max) 理论':>16} {'nu':>8}")
    interval_errors, tail_empirical, tail_theory = [], [], []
    interval_curves = {}
    for k in shapes:
        scale = mean_isi / k
        tau_emp, h_emp = all_interval_histogram(spike_times_by_shape[k],
                                                tau_max, n_tau_bins)
        tau_theory, h_theory = renewal_density_theory(k, scale, tau_max, n_tau_bins)
        mask = tau_theory >= mean_isi
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = np.abs(h_emp[mask] - h_theory[mask]) / h_theory[mask]
        interval_errors.append(float(np.sqrt(np.mean(relative ** 2))))
        tail_empirical.append(float(h_emp[-5:].mean()))
        tail_theory.append(float(h_theory[-5:].mean()))
        interval_curves[k] = (tau_emp, h_emp, tau_theory, h_theory)
        print(f"  {k:>6.1f} {interval_errors[-1]:>12.4f} {tail_empirical[-1]:>16.4f} "
              f"{tail_theory[-1]:>16.4f} {rate:>8.4f}")

    report.check("全间隔直方图 = 理论 h(tau)（相对 RMS 误差 < 15%）",
                 max(interval_errors) < 0.15,
                 [round(e, 4) for e in interval_errors], 0.0, 0.15)
    report.check("tau 很大时 h(tau) -> nu（不规则过程收敛更慢）",
                 max(relative_error(value, rate)
                     for value in tail_theory) < 0.30,
                 [round(v, 4) for v in tail_theory], rate, 0.30)
    report.check("h(tau) 的实测和理论在尾部一致",
                 max(relative_error(e, t)
                     for e, t in zip(tail_empirical, tail_theory)) < 0.20,
                 [round(e, 4) for e in tail_empirical],
                 [round(t, 4) for t in tail_theory], 0.20)

    # ---- 噪声谱：越规则（CV 越小），f ~ nu 处的峰越明显 ----
    n_trials = 60
    trial_duration = duration / n_trials
    trial_bins = int(round(trial_duration / dt))
    freqs = rfft_frequencies(trial_bins, dt)
    psd_shapes = [1.0, 5.0, 20.0]
    psd_by_shape = {}
    for k in psd_shapes:
        scale = mean_isi / k
        stack = np.empty((n_trials, freqs.size))
        for trial in range(n_trials):
            times_trial = simulate_renewal_spike_times(
                gamma_isi_sampler(k, scale), trial_duration,
                make_rng(seed + 3000 + 7 * trial + int(10 * k)))
            counts_trial = spike_times_to_counts(times_trial, dt, trial_bins)
            stack[trial] = positive_half(
                freqs, periodogram_density_twosided(counts_trial, dt))
        psd_by_shape[k] = stack.mean(axis=0)

    comparison = (freqs >= 1.0) & (freqs <= 100.0)
    freqs_compare = freqs[comparison]
    print(f"  功率谱：比较 f in [1, 100] Hz 的理论 S(f) = nu + 2*nu*Int(h - nu)cos")
    print(f"  {'k':>6} {'实测/理论 均值比':>16} {'相关系数':>10} {'谱峰位置(Hz)':>13}"
          f" {'理论 max/mean':>13} {'实测 max/median':>15}")
    scale_ratios, correlations, peak_frequencies = [], [], []
    flatness_by_shape, prominences = {}, []
    theory_by_shape = {}
    for k in psd_shapes:
        theory = renewal_psd_theory(k, mean_isi / k, freqs_compare,
                                    tau_max, n_tau_bins)
        theory_by_shape[k] = theory
        empirical = psd_by_shape[k][comparison]
        scale_ratios.append(float(empirical.mean() / theory.mean()))
        correlations.append(float(np.corrcoef(empirical, theory)[0, 1]))
        peak_index = 1 + int(np.argmax(psd_by_shape[k][1:]))
        peak_frequencies.append(float(freqs[peak_index]))
        flatness_by_shape[k] = float(theory.max() / theory.mean())
        prominences.append(float(empirical.max() / np.median(empirical)))
        print(f"  {k:>6.1f} {scale_ratios[-1]:>16.4f} {correlations[-1]:>10.4f}"
              f" {peak_frequencies[-1]:>13.4f} {flatness_by_shape[k]:>13.4f}"
              f" {prominences[-1]:>15.4f}")
    print("  （注：实测 max/median 即使对白噪声也会 > 1，这是有限样本的极值效应；")
    print("    3300 个频点、60 次平均时，白噪声的噪声底约在 1.5 左右）")

    report.check("实测谱的绝对水平与理论一致（均值比 ≈ 1）",
                 max(abs(r - 1.0) for r in scale_ratios) < 0.10,
                 [round(r, 4) for r in scale_ratios], 1.0, 0.10)
    # k=1 的理论谱是常数（方差为 0），此时「相关系数」没有意义，
    # 所以只对真的有谱结构的 k > 1 计算相关系数。
    structured = [c for k, c in zip(psd_shapes, correlations) if k > 1.0]
    report.check("实测谱的形状与理论一致（k>1 时的相关系数 > 0.8）",
                 min(structured) > 0.8,
                 [round(c, 4) for c in structured], 1.0, 0.8)
    report.check("k=1（Poisson）的理论谱是平的：max/mean < 1.05",
                 flatness_by_shape[1.0] < 1.05,
                 round(flatness_by_shape[1.0], 4), 1.0, 1.05)
    report.check("k=20（规则放电）的理论谱有真峰：max/mean > 1.5",
                 flatness_by_shape[20.0] > 1.5,
                 round(flatness_by_shape[20.0], 4), 1.5, None)
    report.check("k=20（规则放电）的谱峰出现在 f ≈ nu",
                 relative_error(peak_frequencies[-1], rate) < 0.15,
                 peak_frequencies[-1], rate, 0.15)
    report.check("谱峰的显著程度随 ISI 规则性单调上升（max/median）",
                 prominences[0] < prominences[1] < prominences[2],
                 [round(p, 4) for p in prominences], 0.0, None)

    # ---- 图 1：CV / 全间隔直方图 / 更新性 ----
    fig, axes = make_axis_grid(3)
    shape_grid = np.linspace(0.2, 25.0, 300)
    axes[0].plot(shape_grid, 1.0 / np.sqrt(shape_grid), "r-",
                 label="theory 1/sqrt(k)")
    axes[0].plot(shapes, measured_cv, "ko", ms=7, label="measured")
    axes[0].axhline(1.0, color="gray", ls=":", label="Poisson level CV = 1")
    axes[0].set_xlabel("Gamma shape k")
    axes[0].set_ylabel("CV of ISI")
    axes[0].set_title("CV = 1/sqrt(k)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for k, color in zip(psd_shapes, ["C0", "C1", "C2"]):
        tau_emp, h_emp, tau_theory, h_theory = interval_curves[k]
        axes[1].plot(tau_emp * 1e3, h_emp, color=color, lw=1.6,
                     label=f"k={k:g} measured")
        axes[1].plot(tau_theory * 1e3, h_theory, color=color, ls="--", lw=1.2,
                     label=f"k={k:g} theory")
    axes[1].axhline(rate, color="gray", ls=":", label="nu")
    axes[1].set_xlabel("tau (ms)")
    axes[1].set_ylabel("h(tau) (Hz)")
    axes[1].set_title("all-interval histogram = sum f_n(tau)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    bar_labels = [f"k={k:g}" for k in shapes] + ["burst\n(non-renewal)"]
    bar_values = list(lag1_values) + [burst_lag1]
    axes[2].bar(range(len(bar_values)), bar_values,
                color=["C0"] * len(shapes) + ["C3"])
    axes[2].set_xticks(range(len(bar_values)))
    axes[2].set_xticklabels(bar_labels)
    axes[2].axhline(0.0, color="gray")
    axes[2].set_ylabel("lag-1 corr of ISI")
    axes[2].set_title("renewal: ~0; burst: strongly negative")
    axes[2].grid(alpha=0.3, axis="y")

    fig.suptitle("Demo 3 | Renewal statistics", fontsize=13)
    fig.tight_layout()
    save_figure(fig, outdir, "renewal_cv_intervals_renewalness.png", show)

    # ---- 图 2：噪声谱 ----
    fig, axes = make_axis_grid(2)
    for k, color in zip(psd_shapes, ["C0", "C1", "C2"]):
        axes[0].plot(freqs_compare, psd_by_shape[k][comparison], color=color,
                     lw=1.4, label=f"k={k:g} measured")
        axes[0].plot(freqs_compare, theory_by_shape[k], color=color, ls="--",
                     lw=1.2, label=f"k={k:g} theory")
    axes[0].axhline(rate, color="gray", ls=":", label="nu (white floor)")
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("PSD (Hz)")
    axes[0].set_title("regular firing gives a spectral peak")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    for k, color in zip(psd_shapes, ["C0", "C1", "C2"]):
        axes[1].plot(freqs_compare / rate, psd_by_shape[k][comparison] / rate,
                     color=color, lw=1.4,
                     label=f"k={k:g} (CV={1 / math.sqrt(k):.2f})")
    axes[1].axhline(1.0, color="gray", ls=":", label="white-noise floor")
    axes[1].axvline(1.0, color="k", ls=":", label="f = nu")
    axes[1].set_xlabel("f / nu")
    axes[1].set_ylabel("S(f) / nu")
    axes[1].set_xlim(0, 4)
    axes[1].set_title("peak sits at f = nu")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle("Demo 3 | Renewal noise spectrum", fontsize=13)
    fig.tight_layout()
    save_figure(fig, outdir, "renewal_power_spectrum.png", show)

# ---------------------------------------------------------------------------
# 7. Demo：Autocorrelation <-> Power spectrum（Wiener-Khinchin）
# ---------------------------------------------------------------------------

def demo_acf_psd(report, outdir, dt, seed, show):
    print("=" * 74)
    print("Demo 4 | Autocorrelation <-> Power spectrum（Wiener-Khinchin）")
    print("=" * 74)
    print_prediction_prompt([
        "「用 FFT 算自协方差」和「按定义逐点算」会得到同一个数吗？",
        "周期图的逆变换会不会正好还原自协方差（这是定理还是近似）？",
        "Poisson 的 ACF 除了 tau=0 的尖峰以外，其它 lag 上是什么？",
    ])

    n_bins = 1 << 13                      # 8192 个 bin
    rate = 40.0
    short_duration = n_bins * dt
    counts = spike_times_to_counts(
        simulate_poisson_spike_times(rate, short_duration, make_rng(seed)),
        dt, n_bins)
    centered = counts - counts.mean()

    # ---- (1) 三个精确恒等式（可以验证到机器精度）----
    circular_acf = circular_autocovariance(counts)
    two_sided = periodogram_variance_twosided(counts)

    difference = float(np.max(np.abs(np.fft.fft(circular_acf) - two_sided)))
    report.check("Wiener-Khinchin 恒等式：fft(循环自协方差) == |X|^2/N",
                 difference < 1e-12 * max(1.0, float(two_sided.max())),
                 difference, 0.0, None)

    n_lags = 200
    acf_fft = autocovariance_fft(counts, n_lags)
    acf_direct = autocovariance_direct(counts, n_lags)
    difference = float(np.max(np.abs(acf_fft - acf_direct)))
    report.check("FFT 版自协方差 == 按定义逐点算（两种算法同一个量）",
                 difference < 1e-12 * max(1.0, float(np.abs(acf_direct).max())),
                 difference, 0.0, None)

    report.check("Parseval：sum_m P[m] == sum_n xc[n]^2",
                 relative_error(two_sided.sum(), float(np.sum(centered ** 2))) < 1e-9,
                 float(two_sided.sum()), float(np.sum(centered ** 2)), 1e-9)

    report.check("循环自协方差是实、偶函数（c[k] == c[N-k]）",
                 np.allclose(circular_acf[1:], circular_acf[:0:-1]),
                 "见公式", "c[k]=c[N-k]", None)

    # ---- (2) Poisson：ACF 只有 tau = 0 的尖峰，谱是平的 ----
    n_bins_long = 1 << 20                 # 约 1048 s
    long_duration = n_bins_long * dt
    long_counts = spike_times_to_counts(
        simulate_poisson_spike_times(rate, long_duration, make_rng(seed + 11)),
        dt, n_bins_long)
    poisson_acf = autocovariance_fft(long_counts, n_lags)
    expected_lag0 = rate * dt             # bin 内计数是 Poisson(nu*dt)，方差 = nu*dt
    report.check("Poisson：R[0] = nu*dt（但把它当成 delta 函数的离散化）",
                 relative_error(poisson_acf[0], expected_lag0) < 0.05,
                 float(poisson_acf[0]), expected_lag0, 0.05)
    report.check("Poisson：R[k>=1] ≈ 0（只有噪声水平，没有记忆）",
                 float(np.max(np.abs(poisson_acf[1:]))) / poisson_acf[0] < 0.05,
                 float(np.max(np.abs(poisson_acf[1:])) / poisson_acf[0]), 0.0, 0.05)

    # ---- (3) 规则更新过程：ACF 出现周期峰 -> 功率谱出现峰（WK 的推论）----
    mean_isi = 1.0 / rate
    tau_max_acf = n_lags * dt
    n_trials = 40
    trial_duration = 50.0
    trial_bins = int(round(trial_duration / dt))
    freqs = rfft_frequencies(trial_bins, dt)
    shapes_of_interest = [1.0, 20.0]
    acf_cases, psd_cases, theory_acf_cases = {}, {}, {}
    for k in shapes_of_interest:
        scale = mean_isi / k
        # ACF 本身已经是对大量 spike 对求平均，用一段长记录就足够光滑
        long_times = simulate_renewal_spike_times(
            gamma_isi_sampler(k, scale), long_duration,
            make_rng(seed + 21 + int(k)))
        long_counts_k = spike_times_to_counts(long_times, dt, n_bins_long)
        acf_cases[k] = autocovariance_fft(long_counts_k, n_lags)

        # 理论 ACF（离散化）：R[k] = nu*(h(k*dt) - nu)*dt，k >= 1
        tau_theory, h_theory = renewal_density_theory(k, scale, tau_max_acf, n_lags)
        theory_acf_cases[k] = (tau_theory, rate * (h_theory - rate) * dt)

        # PSD 需要多 trial 平均才能压住周期图的涨落
        stack = np.empty((n_trials, freqs.size))
        for trial in range(n_trials):
            times_trial = simulate_renewal_spike_times(
                gamma_isi_sampler(k, scale), trial_duration,
                make_rng(seed + 5000 + 13 * trial + int(k)))
            counts_trial = spike_times_to_counts(times_trial, dt, trial_bins)
            stack[trial] = positive_half(
                freqs, periodogram_density_twosided(counts_trial, dt))
        psd_cases[k] = stack.mean(axis=0)

    # ACF 的峰位置 <-> 谱的峰位置：f = 1/tau
    regular = shapes_of_interest[-1]
    regular_acf = acf_cases[regular]
    tau_peak = float((1 + int(np.argmax(regular_acf[1:]))) * dt)
    peak_index = 1 + int(np.argmax(psd_cases[regular][1:]))
    frequency_peak = float(freqs[peak_index])
    print(f"  k={regular:g}：ACF 在 tau = {tau_peak * 1e3:.2f} ms 出现峰，"
          f"谱峰在 f = {frequency_peak:.2f} Hz")
    report.check("ACF 的峰位置 tau* ≈ E[ISI] = 1/nu",
                 relative_error(tau_peak, mean_isi) < 0.20,
                 tau_peak, mean_isi, 0.20)
    report.check("谱的峰位置 f* ≈ 1/tau*（这正是 WK 的推论）",
                 relative_error(frequency_peak, 1.0 / tau_peak) < 0.05,
                 frequency_peak, 1.0 / tau_peak, 0.05)

    # 理论 ACF vs 实测 ACF（只对 k>1 检验，k=1 的理论值恒为 0，比值没有意义）
    lags = np.arange(1, n_lags + 1) * dt
    tau_theory, theory_acf = theory_acf_cases[regular]
    theory_on_lags = np.interp(lags, tau_theory, theory_acf)
    residual = float(np.sqrt(np.mean((regular_acf[1:] - theory_on_lags) ** 2))
                     / np.max(np.abs(theory_on_lags)))
    report.check("k=20 的实测 ACF = 理论 nu*(h(tau) - nu)*dt（相对峰值 < 0.1）",
                 residual < 0.10, residual, 0.0, 0.10)

    # 理论 PSD vs 实测 PSD
    comparison = (freqs >= 1.0) & (freqs <= 100.0)
    freqs_compare = freqs[comparison]
    theory_psd = renewal_psd_theory(regular, mean_isi / regular, freqs_compare,
                                    6.0 * mean_isi, 600)
    empirical_psd = psd_cases[regular][comparison]
    scale_ratio = float(empirical_psd.mean() / theory_psd.mean())
    correlation = float(np.corrcoef(empirical_psd, theory_psd)[0, 1])
    report.check("k=20 的实测谱与理论 S(f) 的绝对水平一致（均值比 ≈ 1）",
                 abs(scale_ratio - 1.0) < 0.10, scale_ratio, 1.0, 0.10)
    report.check("k=20 的实测谱与理论 S(f) 的形状一致（corr > 0.8）",
                 correlation > 0.8, correlation, 1.0, 0.8)

    # ---- 图 1：ACF / 谱 / 恒等式 ----
    lags_ms = np.arange(n_lags + 1) * dt * 1e3
    fig, axes = make_axis_grid(4)
    axes[0].plot(lags_ms, poisson_acf, "C0-", lw=1.2)
    axes[0].axhline(0.0, color="gray", lw=0.8)
    axes[0].set_xlabel("lag tau (ms)")
    axes[0].set_ylabel("R(tau)  (counts^2)")
    axes[0].set_title("Poisson: ACF has a spike only at tau = 0")
    axes[0].grid(alpha=0.3)

    axes[1].plot(lags_ms, regular_acf, "C3-", lw=1.4, label="measured")
    axes[1].plot(tau_theory * 1e3, theory_acf, "k--", lw=1.2,
                 label="theory nu*(h-nu)*dt")
    axes[1].axhline(0.0, color="gray", lw=0.8)
    axes[1].set_xlabel("lag tau (ms)")
    axes[1].set_ylabel("R(tau)  (counts^2)")
    axes[1].set_title(f"regular renewal (k={regular:g}): periodic peaks")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(freqs_compare, psd_cases[1.0][comparison], "C0-", lw=1.2,
                 label="k=1 (Poisson)")
    axes[2].plot(freqs_compare, psd_cases[regular][comparison], "C3-", lw=1.4,
                 label=f"k={regular:g} (regular)")
    axes[2].plot(freqs_compare, theory_psd, "k--", lw=1.2, label="theory S(f)")
    axes[2].axhline(rate, color="gray", ls=":", label="nu")
    axes[2].set_xlabel("frequency (Hz)")
    axes[2].set_ylabel("PSD (Hz)")
    axes[2].set_title("flat spectrum vs peaked spectrum")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)

    axes[3].plot(np.arange(n_bins) * dt, two_sided, "C0-", lw=3.0, alpha=0.6,
                 label="|X|^2 / N")
    axes[3].plot(np.arange(n_bins) * dt, np.fft.fft(circular_acf), "r--", lw=1.0,
                 label="fft(circular ACF)")
    axes[3].set_xlabel("frequency (Hz)")
    axes[3].set_ylabel("P  (counts^2)")
    axes[3].set_title("the two curves coincide (max diff ~ 1e-17)")
    axes[3].legend(fontsize=7)
    axes[3].grid(alpha=0.3)

    fig.suptitle("Demo 4 | Autocorrelation and power spectrum", fontsize=13)
    fig.tight_layout()
    save_figure(fig, outdir, "acf_psd_wiener_khinchin.png", show)

    # ---- 图 2：ACF 的峰与谱的峰互为倒数 ----
    fig, axes = make_axis_grid(2)
    axes[0].plot(lags_ms, regular_acf, "C3-", lw=1.4)
    axes[0].axvline(tau_peak * 1e3, color="k", ls=":")
    axes[0].annotate(f"tau* = {tau_peak * 1e3:.1f} ms",
                     xy=(tau_peak * 1e3, float(np.max(regular_acf))),
                     xytext=(tau_peak * 1e3 + 30.0, float(np.max(regular_acf)) * 0.8),
                     arrowprops=dict(arrowstyle="->"))
    axes[0].set_xlabel("lag tau (ms)")
    axes[0].set_ylabel("R(tau)  (counts^2)")
    axes[0].set_title("autocorrelation peaks at tau*")
    axes[0].grid(alpha=0.3)

    axes[1].plot(freqs_compare, psd_cases[regular][comparison], "C3-", lw=1.4)
    axes[1].axvline(frequency_peak, color="k", ls=":")
    axes[1].annotate(f"f* = {frequency_peak:.1f} Hz",
                     xy=(frequency_peak, float(np.max(empirical_psd))),
                     xytext=(frequency_peak + 12.0, float(np.max(empirical_psd)) * 0.85),
                     arrowprops=dict(arrowstyle="->"))
    axes[1].axhline(rate, color="gray", ls=":", label="nu (white floor)")
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("PSD (Hz)")
    axes[1].set_title("power spectrum peaks at f* = 1/tau*")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle("Demo 4 | ACF peak and spectrum peak are reciprocal", fontsize=13)
    fig.tight_layout()
    save_figure(fig, outdir, "acf_psd_reciprocal_peaks.png", show)

# === END OF FILE ===














