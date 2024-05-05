import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .particles import Particles
from numba import jit, njit, prange, set_num_threads

"""
The N-Body Simulator class is responsible for simulating the motion of N bodies



"""

class NBodySimulator:

    def __init__(self, particles: Particles):
        
        # TODO
        self.particles = particles
        self.setup()    # set up the default simulation enviroments

        return

    def setup(self, G=1,
                    rsoft=0.01,
                    method="RK4",
                    io_freq=10,
                    io_header="nbody",
                    io_screen=True,
                    visualization=False):
        """
        Customize the simulation enviroments.

        :param G: the graivtational constant
        :param rsoft: float, a soften length
        :param meothd: string, the numerical scheme
                       support "Euler", "RK2", and "RK4"

        :param io_freq: int, the frequency to outupt data.
                        io_freq <=0 for no output. 
        :param io_header: the output header
        :param io_screen: print message on screen or not.
        :param visualization: on the fly visualization or not. 
        """
        
        # TODO
        self.G = G
        self.rsoft = rsoft
        method = method.lower()
        self.method = method
        if io_freq <= 0: io_freq = np.inf
        self.io_freq = io_freq
        self.io_header = io_header
        self.io_screen = io_screen
        self.visualization = visualization

        if method == 'euler':
            self._advance_particles = self._advance_particles_Euler
        elif method == 'rk2':
            self._advance_particles = self._advance_particles_RK2
        elif method == 'rk4':
            self._advance_particles = self._advance_particles_RK4
        elif method == 'leapfrog':
            self._advance_particles = self._advance_particles_Leapfrog
        else:
            raise ValueError(f"Invalid method {method}!")

        return

    def evolve(self, dt:float, tmax:float):
        """
        Start to evolve the system

        :param dt: float, the time step
        :param tmax: float, the total time to evolve
        
        """

        # TODO
        time = self.particles.time
        nsteps = int(np.ceil((tmax-time)/dt))
        particles = self.particles

        # check output directory
        folder = f"data_{self.io_header}"
        Path(folder).mkdir(parents=True, exist_ok=True)
        # simulation loop
        print (nsteps)

        # record energy for plot
        potential_energys = []
        kenetic_energys = []
        time_array = []
        for n in range(nsteps):
            if time + dt > tmax:
                dt = tmax - time    # prevent overshooting

            # visualization
            if self.visualization and n % 200 == 0:
                particles.draw()

            # update physics
            self._advance_particles(dt, self.particles)
            time+=dt

            # check IO
            if n % self.io_freq == 0:
                if self.io_screen:
                    print(f"Time: {time}, dt: {dt}")
                file_name = f"{self.io_header}_{n:06d}.dat"
                file_name = folder + '/' + file_name
                self.particles.time = time
                self.particles.output(file_name)

                potential_energys.append(self.particles.potential_energy)
                kenetic_energys.append(self.particles.kenetic_energy)
                time_array.append(time)

        # plot energy
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(time_array, potential_energys, label='Potential Energy')
        ax.plot(time_array, kenetic_energys, label='Kenetic Energy')
        ax.plot(time_array, np.array(potential_energys) + np.array(kenetic_energys), label='Total Energy')
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy')
        ax.legend()
        plt.show()
        plt.close()
        print("Simulation is done!")
        return

    def _calculate_acceleration(self, nparticles, masses, positions):
        """
        Calculate the acceleration of the particles
        """
        accelerations = np.zeros_like(positions)
        

        # TODO
        rsoft = self.rsoft
        G     = self.G
        
        # use multi-threading
        accelerations = _calculate_acceleration_kernel(nparticles, masses, positions, accelerations, G, rsoft)

        return accelerations
    
    # def _calculate_acceleration_energy(self, nparticles, masses, positions):
    #     """
    #     Calculate the acceleration of the particles and the energy of the system
    #     """
    #     accelerations = np.zeros_like(positions)
    #     velocities = self.particles.velocities

    #     # TODO
    #     rsoft = self.rsoft
    #     G     = self.G
        
    #     # use multi-threading
    #     accelerations, potential_energy, kenetic_energy = _calculate_acceleration_energy_kernel(nparticles, masses, positions, velocities, accelerations, G, rsoft)

    #     return accelerations, potential_energy, kenetic_energy
        
    def _advance_particles_Euler(self, dt, particles):

        #TODO
        # print("Euler")
        nparticles = particles.nparticles
        mass = particles.masses

        pos = particles.positions
        vel = particles.velocities
        acc = particles.accelerations

        pos = pos + vel*dt
        vel = vel + acc*dt
        acc = self._calculate_acceleration(nparticles, mass, pos)
        # acc, pe, ke = self._calculate_acceleration_energy(nparticles, mass, pos)

        particles.set_particles(pos, vel, acc)
        # particles.set_particles(pos, vel, acc, pe, ke)

        return particles

    def _advance_particles_RK2(self, dt, particles):

        # TODO
        nparticles = particles.nparticles
        mass = particles.masses

        pos = particles.positions
        vel = particles.velocities
        acc = particles.accelerations

        pos2 = pos + vel*dt
        vel2 = vel + acc*dt
        acc2 = self._calculate_acceleration(nparticles, mass, pos2)

        pos2 = pos2 + vel2*dt
        vel2 = vel2 + acc2*dt

        pos = 0.5*(pos + pos2)
        vel = 0.5*(vel + vel2)
        acc = self._calculate_acceleration(nparticles, mass, pos)
        # acc, pe, ke = self._calculate_acceleration_energy(nparticles, mass, pos)

        particles.set_particles(pos, vel, acc)
        # particles.set_particles(pos, vel, acc, pe, ke)

        return particles

    def _advance_particles_RK4(self, dt, particles):
        
        #TODO
        nparticles = particles.nparticles
        mass = particles.masses

        pos = particles.positions
        vel = particles.velocities
        acc = particles.accelerations


        pos2 = pos + vel*dt/2
        vel2 = vel + acc*dt/2
        acc2 = self._calculate_acceleration(nparticles, mass, pos2)

        pos3 = pos + vel2*dt/2
        vel3 = vel + acc2*dt/2
        acc3 = self._calculate_acceleration(nparticles, mass, pos3)

        pos4 = pos + vel3*dt
        vel4 = vel + acc3*dt
        acc4 = self._calculate_acceleration(nparticles, mass, pos4)

        pos = pos + (vel + 2*vel2 + 2*vel3 + vel4)*dt/6
        vel = vel + (acc + 2*acc2 + 2*acc3 + acc4)*dt/6
        acc = self._calculate_acceleration(nparticles, mass, pos)
        # acc, pe, ke = self._calculate_acceleration_energy(nparticles, mass, pos)

        particles.set_particles(pos, vel, acc)
        # particles.set_particles(pos, vel, acc, pe, ke)
        return particles

    def _advance_particles_Leapfrog(self, dt, particles):
        
        #TODO
        nparticles = particles.nparticles
        mass = particles.masses

        pos = particles.positions
        vel = particles.velocities
        acc = particles.accelerations

        vel2 = vel + acc*dt/2
        pos = pos + vel2*dt
        acc = self._calculate_acceleration(nparticles, mass, pos)
        # acc, pe, ke = self._calculate_acceleration_energy(nparticles, mass, pos)
        vel = vel2 + acc*dt/2

        particles.set_particles(pos, vel, acc)
        # particles.set_particles(pos, vel, acc, pe, ke)
        return particles


@njit(parallel=True)
def _calculate_acceleration_kernel(nparticles, masses, positions, accelerations, G, rsoft):
    """
    Calculate the acceleration of the particles

    :param particles: Particles, the particles to calculate the acceleration
    """

    # kernel for acceleration calculation
    for i in prange(nparticles):
        for j in prange(nparticles):
            if (j>i): 
                rij = positions[i,:] - positions[j,:]
                r = np.sqrt(np.sum(rij**2) + rsoft**2)
                force = - G * masses[i,0] * masses[j,0] * rij / r**3
                accelerations[i,:] += force[:] / masses[i,0]
                accelerations[j,:] -= force[:] / masses[j,0]

    return accelerations

# @njit(parallel=True)
# def _calculate_acceleration_energy_kernel(nparticles, masses, positions, velocities ,accelerations, G, rsoft):
#     """
#     Calculate the acceleration of the particles

#     :param particles: Particles, the particles to calculate the acceleration
#     """

#     # kernel for acceleration calculation
#     potential_energy = 0
#     kenetic_energy = 0
#     for i in prange(nparticles):
#         for j in prange(nparticles):
#             if (j>i): 
#                 rij = positions[i,:] - positions[j,:]
#                 r = np.sqrt(np.sum(rij**2) + rsoft**2)
#                 force = - G * masses[i,0] * masses[j,0] * rij / r**3
#                 accelerations[i,:] += force[:] / masses[i,0]
#                 accelerations[j,:] -= force[:] / masses[j,0]
#                 potential_energy -= G * masses[i,0] * masses[j,0] / r
#         kenetic_energy += 0.5 * masses[i,0] * np.sum(velocities[i,:]**2)

#     return accelerations, potential_energy, kenetic_energy

if __name__ == "__main__":
    
    pass