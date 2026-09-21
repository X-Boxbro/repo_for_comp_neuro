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
    u_reset=-70e-3
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

  def integrate_fire_process(self):
    for i in range (self.n):
      self.u[i,0]=self.u_start
    for i in range(1,self.steps):
      for j in range(self.n):
        self.u[j,i]=self.u[j,i-1]+self.dt*(self.I[j,i-1]*self.R-self.u[j,i-1]+self.u_rest)/self.tau
        if self.u[j,i]>=self.u_th:
          self.u[j,i]=self.u_reset
          self.fire[j,i]=1

  def plot_process(self):
    time = np.arange(self.steps) * self.dt

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # Convert SI units to readable units.
    time_ms = time * 1e3
    voltage_mV = self.u * 1e3

    # Membrane potential.
    for neuron_index in range(self.n):
        axes[0].plot(
            time_ms,
            voltage_mV[neuron_index],
            label=f"neuron {neuron_index}"
        )

    axes[0].axhline(
        self.u_th * 1e3,
        color="red",
        linestyle="--",
        label="threshold"
    )
    axes[0].set_ylabel("Membrane potential (mV)")
    axes[0].set_title("LIF membrane potential")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Spike raster.
    for neuron_index in range(self.n):
        spike_times = time_ms[self.fire[neuron_index] == 1]
        axes[1].scatter(
            spike_times,
            np.full_like(spike_times, neuron_index),
            marker="|",
            s=100
        )

    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Neuron")
    axes[1].set_yticks(range(self.n))
    axes[1].set_title("Spike times")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
    lif = LIF()

    #constant input
    lif.I[:, :] = 0.25e-9


    lif.integrate_fire_process()
    lif.plot_process()


if __name__ == "__main__":
    main()