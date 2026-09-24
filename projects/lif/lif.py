import numpy as np
import matplotlib.pyplot as plt

#LIF(leaky integrate-and-fire) Model
#tau*du/dt=I(t)R-u+u_rest
#tau---time constant,dt---a small period of time,u---the voltage of membrane,I--the current past through the mambrane,u_rest--resting potential

class LIF:
  def __init__(self, n=1, t_max=150e-3, #
    dt=1e-3,
    tau=20e-3,
    u_rest=-70e-3,
    u_th=-50e-3,
    R=100e6,
    u_start=-70e-3,
    u_reset=-70e-3,
    noise_amplitude=0.0,
    seed=None
    ):
    #number of neuron(s)
    self.n=n
    #property parameters
    self.t_max=t_max#
    self.dt=dt
    self.tau=tau
    self.u_rest=u_rest
    self.u_th=u_th #threshold for firing
    self.R=R #resistor
    self.u_reset=u_reset
    
    #state parameters
    self.steps=int(self.t_max/self.dt)
    self.u_start=u_start
    self.I=np.zeros((n,self.steps))
    self.u=np.zeros((n,self.steps))
    self.fire=np.zeros((n,self.steps))

    #noise
    self.noise_amplitude=noise_amplitude
    self.rng = np.random.default_rng(seed)
    standard_normal_noise = self.rng.normal(
        loc=0.0,
        scale=1.0,
        size=(self.n, self.steps)
    )

    self.I_noise = (
        self.noise_amplitude
        * np.sqrt(self.dt)
        * standard_normal_noise
    )

  def Euler_integrate_fire_process(self):
    for i in range (self.n):
      self.u[i,0]=self.u_start
    for i in range(1,self.steps):
      for j in range(self.n):
        # #without noise
        # self.u[j,i]=self.u[j,i-1]+self.dt*(self.I[j,i-1]*self.R-self.u[j,i-1]+self.u_rest)/self.tau

        #with noise(I_noise)
        #with background stochastic spike arrival
        
        external_current = self.I[j, i - 1]

        background_current = self.I_background[j, i - 1]

        self.u[j, i] = (
            self.u[j, i - 1]
            + self.dt
            / self.tau
            * (self.R* (external_current+ background_current )- (self.u[j, i - 1] - self.u_rest
                        )
             )
            + self.I_noise[j, i - 1] * self.R/ self.tau
            )


        if self.u[j,i]>=self.u_th:
          self.u[j,i]=self.u_reset
          self.fire[j,i]=1

  def plot_process(self):
    time = np.arange(self.steps) * self.dt
    time_ms = time * 1e3
    voltage_mV = self.u * 1e3
    input_nA = self.I * 1e9

    axes_count = self.n + 2

    fig, axes = plt.subplots(
        axes_count,
        1,
        figsize=(10, max(7, 2 * axes_count)),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1] + [2] * self.n + [1]
        }
    )

    # Input current
    for neuron_index in range(self.n):
        axes[0].plot(
            time_ms,
            input_nA[neuron_index],
            label=f"neuron {neuron_index}"
        )

    axes[0].set_ylabel("Input\n(nA)")
    axes[0].set_title("Input current")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # One membrane-potential panel per neuron
    for neuron_index in range(self.n):
        axis = axes[neuron_index + 1]

        axis.plot(
            time_ms,
            voltage_mV[neuron_index],
            color=f"C{neuron_index % 10}",
            label=f"neuron {neuron_index}"
        )

        axis.axhline(
            self.u_th * 1e3,
            color="red",
            linestyle="--",
            linewidth=1,
            label="threshold"
        )

        axis.set_ylabel("u (mV)")
        axis.set_title(f"Neuron {neuron_index}")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")

    # Spike raster
    raster_axis = axes[-1]

    for neuron_index in range(self.n):
        spike_times = time_ms[self.fire[neuron_index] == 1]

        raster_axis.scatter(
            spike_times,
            np.full(spike_times.shape, neuron_index),
            marker="|",
            s=120,
            color=f"C{neuron_index % 10}"
        )

    raster_axis.set_xlabel("Time (ms)")
    raster_axis.set_ylabel("Neuron")
    raster_axis.set_yticks(range(self.n))
    raster_axis.set_title("Spike raster")
    raster_axis.grid(True, alpha=0.3)

    fig.suptitle("LIF population simulation", fontsize=14)
    fig.tight_layout()

    plt.show()
    return fig, axes

  #background spike---stochastic spike arrival
  def generate_background_spikes(
    self,
    rate_hz=100.0,
    amplitude=0.1e-9,
    duration=5e-3
):
    probability = rate_hz * self.dt

    if probability > 1:
        raise ValueError(
            "rate_hz * dt must be <= 1 for the step-function implementation."
        )

    self.background_spikes = (
        self.rng.random((self.n, self.steps)) < probability
    ).astype(float)

    self.I_background = np.zeros((self.n, self.steps))

    duration_steps = max(1, int(duration / self.dt))

    for neuron_index in range(self.n):
        spike_indices = np.flatnonzero(
            self.background_spikes[neuron_index]
        )

        for spike_index in spike_indices:
            end_index = min(
                spike_index + duration_steps,
                self.steps
            )

            self.I_background[
                neuron_index,
                spike_index:end_index
            ] += amplitude

def main():
    lif = LIF(t_max=1,n=3,noise_amplitude=1e-12,seed=42)

    #sin input
    time = np.arange(lif.steps) * lif.dt
    frequency = 4.0
    baseline_current = 0.1e-9
    amplitude = 0.15e-9
    input_current = (
        baseline_current
        + amplitude * np.sin(2 * np.pi * frequency * time)
    )


    lif.I[:, :] = input_current

    #background spike
    lif.generate_background_spikes(
        rate_hz=100.0,
        amplitude=0.1e-9,
        duration=5e-3
    )


    lif.Euler_integrate_fire_process()
    lif.plot_process()

if __name__ == "__main__":
    main()