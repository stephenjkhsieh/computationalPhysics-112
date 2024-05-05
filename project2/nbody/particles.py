import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit, prange, set_num_threads

class Particles:
    """
    Particle class to store particle properties
    """
    def __init__(self, N:int=0) -> None:
        """
        Allocate memories for particle properties

        :param N: number of particles
        """
        self.nparticles = N
        self._masses = np.ones((N,1))
        self._positions = np.zeros((N,3))
        self._velocities = np.zeros((N,3))
        self._accelerations = np.zeros((N,3))
        self._tags = np.arange(N)
        self._time = 0.0
        pe, ke = self.calc_energy()
        self._potential_energy = pe
        self._kenetic_energy = ke
        return
    
    @property
    def kenetic_energy(self) -> float:
        return self._kenetic_energy
    
    @kenetic_energy.setter
    def kenetic_energy(self, ke: float) -> None:
        self._kenetic_energy = ke
        return
    
    @property
    def potential_energy(self) -> float:
        return self._potential_energy
    
    @potential_energy.setter
    def potential_energy(self, pe: float) -> None:
        self._potential_energy = pe
        return

    @property
    def masses(self) -> np.ndarray:
        return self._masses
    
    @masses.setter
    def masses(self, m: np.ndarray) -> None:
        if m.shape != np.ones((self.nparticles,1)).shape:
            print("Number of particles does not match!")
            raise ValueError
        
        self._masses = m
        return
    
    @property
    def positions(self) -> np.ndarray:
        return self._positions
    
    @positions.setter
    def positions(self, pos: np.ndarray) -> None:
        if pos.shape != np.zeros((self.nparticles,3)).shape:
            print("Number of particles does not match!")
            raise ValueError
        
        self._positions = pos
        return
    
    @property
    def velocities(self) -> np.ndarray:
        return self._velocities
    
    @velocities.setter
    def velocities(self, vel: np.ndarray) -> None:
        if vel.shape != np.zeros((self.nparticles,3)).shape:
            print("Number of particles does not match!")
            raise ValueError
        
        self._velocities = vel
        return
    
    @property
    def accelerations(self) -> np.ndarray:
        return self._accelerations
    
    @accelerations.setter
    def accelerations(self, acc: np.ndarray) -> None:
        if acc.shape != np.zeros((self.nparticles,3)).shape:
            print("Number of particles does not match!")
            raise ValueError
        
        self._accelerations = acc
        return
    
    @property
    def tags(self) -> np.ndarray:
        return self._tags
    
    @tags.setter
    def tags(self, t: np.ndarray) -> None:
        if t.shape != np.arange(self.nparticles).shape:
            print("Number of particles does not match!")
            raise ValueError
        
        self._tags = t
        return
    
    @property
    def time(self) -> float:
        return self._time
    
    @time.setter
    def time(self, t: float) -> None:
        self._time = t
        return
    
    def add_particles(self, mass, pos, vel, acc):
        """
        Add particles to the system

        :param mass: mass of the particles
        :param pos: positions of the particles
        :param vel: velocities of the particles
        :param acc: accelerations of the particles
        """
        self.nparticles += 1
        self._masses = np.vstack((self._masses, mass))
        self._positions = np.vstack((self._positions, pos))
        self._velocities = np.vstack((self._velocities, vel))
        self._accelerations = np.vstack((self._accelerations, acc))
        self._tags = np.arange(self.nparticles)
        return

    def set_particles(self, pos, vel, acc):
        self.positions = pos
        self.velocities = vel
        self.accelerations = acc
        return
    
    def set_energy(self, ke, pe):
        self.kenetic_energy = ke
        self.potential_energy = pe
        return
    
    def output(self, filename):
        """
        Output the particle properties to a file

        :param filename: output file name
        """
        mass = self._masses
        pos = self._positions
        vel = self._velocities
        acc = self._accelerations
        tags = self._tags
        time = self._time
        pe, ke = self.calc_energy()
        self._potential_energy = pe
        self._kenetic_energy = ke
        header = f"""
                ----------------------------------------------------
                Data from a 3D direct N-body simulation. 

                rows are i-particle; 
                coumns are :tag, mass, x ,y, z, vx, vy, vz, ax, ay, az

                NTHU, Computational Physics 

                ----------------------------------------------------
                Time = {time}
                Potential Energy = {pe}
                Kenetic Energy = {ke}
                Total Energy = {ke+pe}"""
        np.savetxt(filename,(tags[:],mass[:,0],pos[:,0],pos[:,1],pos[:,2],
                            vel[:,0],vel[:,1],vel[:,2],
                            acc[:,0],acc[:,1],acc[:,2]),header=header)
        return
    
    def draw(self, dim=2):
        """
        Draw the particles in 2D or 3D

        :param dim: dimension of the plot
        """
        fig = plt.figure()
        if dim == 2:
            ax = fig.add_subplot(111)
            ax.scatter(self._positions[:,0], self._positions[:,1], s=1)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        elif dim == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self._positions[:,0], self._positions[:,1], self._positions[:,2])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
        else:
            print("Invalid dimension!")
            raise ValueError
        ax.set_aspect('equal')
        plt.xlim(-4,4)
        plt.ylim(-4,4)
        plt.title(f'Time = {self._time:.0f}')
        # plt.tight_layout()
        plt.show()
        plt.close()
    
    def calc_energy(self):
        """
        Calculate the total energy of the system
        """
        nparticles = self.nparticles
        masses = self.masses
        positions = self.positions
        velocities = self.velocities
        pe, ke = _calculate_energy_kernel(nparticles, masses, positions, velocities)

        return pe, ke


@njit(parallel=True)
def _calculate_energy_kernel(nparticles, masses, positions, velocities, G=1, rsoft=0.01):
    """
    Calculate the energy of the particles
    """

    # kernel for acceleration calculation
    potential_energy = 0
    kenetic_energy = 0
    for i in prange(nparticles):
        for j in prange(nparticles):
            if (j>i): 
                rij = positions[i,:] - positions[j,:]
                r = np.sqrt(np.sum(rij**2) + rsoft**2)
                potential_energy -= G * masses[i,0] * masses[j,0] / r
        kenetic_energy += 0.5 * masses[i,0] * np.sum(velocities[i,:]**2)

    return potential_energy, kenetic_energy

if __name__ == '__main__':
    # Test the Particles class
    N = 100
    particles = Particles(N)
    print(particles.nparticles)
    print(particles._masses.shape)
    print(particles._positions.shape)
    print(particles._velocities.shape)
    print(particles._accelerations.shape)
    print(particles._tags.shape)
    print(particles._time)
    plt.show()
    num_particles = N

    # particles.masses = np.ones((num_particles,1))
    # particles.positions = np.random.rand(num_particles, 3)
    # particles.velocities = np.random.rand(num_particles, 3)
    # particles.accelerations = np.random.rand(num_particles, 3)
    # particles.tags = np.linspace(1,num_particles,num_particles)

    particles.masses = np.ones(num_particles)

