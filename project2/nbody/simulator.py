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
        self.dt = dt
        self.tmax = tmax
        time = self.particles.time
        nsteps = int(np.ceil((tmax-time)/dt))

        # check output directory
        folder = f"data_{self.io_header}"
        Path(folder).mkdir(parents=True, exist_ok=True)
        # simulation loop
        print (nsteps)
        for n in range(nsteps):
            if time + dt > tmax:
                dt = tmax - time    # prevent overshooting

            # update physics
            self._advance_particles(dt, self.particles)
            time+=dt

            # check IO
            if n % self.io_freq == 0:
                if self.io_screen:
                    print(f"Time: {time}, dt: {dt}")
                file_name = f"{self.io_header}_{n:06d}.dat"
                file_name = folder + '/' + file_name
                self.particles.output(file_name)

        print("Simulation is done!")
        return

    def _calculate_acceleration(self, nparticles, masses, positions):
        """
        Calculate the acceleration of the particles
        """
        accelerations = np.zeros_like(positions)
        

        # TODO




        return accelerations
        
    def _advance_particles_Euler(self, dt, particles):

        #TODO
        # print("Euler")






        return particles

    def _advance_particles_RK2(self, dt, particles):

        # TODO





        return particles

    def _advance_particles_RK4(self, dt, particles):
        
        #TODO








        return particles



if __name__ == "__main__":
    
    pass