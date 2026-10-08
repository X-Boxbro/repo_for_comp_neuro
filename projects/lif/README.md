# Leaky Integrate-and-Fire Model

This project implements a minimal LIF neuron population model. The membrane potential integrates input current, leaks toward the resting potential, emits a spike at threshold, and resets.

## Files

- [`lif.py`](./lif.py) - LIF simulation, Gaussian noise, background spike arrivals, and plotting.

## Model

The subthreshold dynamics are represented by:

```text
tau * du/dt = R * I(t) - (u - u_rest)
```

Euler integration is used for the current implementation. When `u >= u_th`, a spike is recorded and the membrane potential is reset.

## Language

- [中文说明](./README.zh-CN.md)
