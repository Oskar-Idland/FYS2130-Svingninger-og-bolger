import numpy as np
import sys
import matplotlib.pyplot as plt
import numba as nb
from numba import njit as func_go_brrr
from dataclasses import dataclass
from time import time
from random import random, seed
seed(123)

def jitdataclass(cls=None, *, extra_spec=[]):
    """
    Helper decorator to make it easier to numba jitclass dataclasses

    Inspired by https://github.com/numba/numba/issues/4037#issuecomment-907523015
    """
    def _jitdataclass(cls):
        dc_cls = dataclass(cls, eq=False)
        del dc_cls.__dataclass_params__
        del dc_cls.__dataclass_fields__
        return nb.experimental.jitclass(dc_cls, spec=extra_spec)
    
    if cls is not None:
        # We've been called without additional args - invoke actual decorator immediately
        return _jitdataclass(cls)
    # We've been called with additional args - so return actual decorator which python calls for us
    return _jitdataclass


@jitdataclass
class Particle:
    x: nb.float32
    y: nb.float32
    vx: nb.float32
    vy: nb.float32
    # m: nb.float32
    # r: nb.float32
    
@dataclass
class nojit_Particle:
    x: float
    y: float
    vx: float
    vy: float
    # m: float
    # r: float
    
    
def collision_simulator(pos_p, vel_p, dt, r):
    # Write function which takes in the position and velocities of particles in the form of a matrix, and calculates collisions between them if their distance is less than r. If a collision occurs, update the velocities of the particles involved in the collision. If no collision, keep going in the same direction. 
    
    # Calculate distances between all particles
    dist = np.sqrt((pos_p[:, 0, None] - pos_p[:, 0])**2 + (pos_p[:, 1, None] - pos_p[:, 1])**2)
    
    # Find all particles which are closer than r
    close_particles = np.argwhere(dist < r)
    print(close_particles)
    
    # Calculate the new velocities of the particles involved in the collision
    for i, j in close_particles:
        # Calculate the unit vector between the two particles
        n = (pos_p[i] - pos_p[j])/dist[i, j]
        # Calculate the relative velocity of the particles
        rel_vel = vel_p[i] - vel_p[j]
        # Calculate the normal and tangential components of the relative velocity
        rel_vel_n = np.dot(rel_vel, n)
        rel_vel_t = rel_vel - rel_vel_n*n
        # Calculate the new normal and tangential components of the relative velocity
        rel_vel_n_new = -rel_vel_n
        rel_vel_t_new = rel_vel_t
        # Calculate the new relative velocity
        rel_vel_new = rel_vel_n_new*n + rel_vel_t_new
        # Calculate the new velocities of the particles
        vel_p[i] = rel_vel_new + vel_p[j]
        vel_p[j] = -rel_vel_new + vel_p[i]
        
    # Update the position of the particles
    pos_p += vel_p*dt
    
    return pos_p, vel_p

        

    
# Tester måter å lage en liste med partikler på
N = int(1E7)
# start = time()
# particles = [Particle(random(), random(), random(), random()) for _ in range(N)]

# print(f'Numba dataclass:    {time() - start:.2g} s')
# start = time()
# nojit_particles = [nojit_Particle(random(), random(), random(), random()) for _ in range(N)]
# print(f'Vanlig dataclass:   {time() - start:.2g} s')

# Tester å bare lagre alt i en matrise
start = time()
pos_matrix = np.random.rand(N, 2)
vel_matrix = np.random.rand(N, 2)
print(f'Matrise:            {time() - start:.2g} s')

# Henter ut posisjonene til partiklene og lagrer som matrise for plotting
# start = time()
# positions = np.array([[p.x, p.y] for p in particles])
# print(f'Tid for å hente ut posisjonene: {time() - start:.2g} s')

# start = time()
# plt.scatter(positions[:, 0], positions[:, 1])
# print(f'Tid for å plotte posisjonene:   {time() - start:.2g} s')
# plt.close()

# @func_go_brrr
# def update_positions(particles, dt):
#     for p in particles:
#         p.x += p.vx * dt    
#         p.y += p.vy * dt
        
def nojit_update_positions(test_particles, dt, n):
    for _ in range(n):
        for p in test_particles:
            p.x += p.vx * dt
            p.y += p.vy * dt

# @func_go_brrr()
def update_positions_mat(pos_matrix, dt, n):
    for _ in range(n):
        pos_matrix[:, 0] += vel_matrix[:, 0]*dt
        pos_matrix[:, 1] += vel_matrix[:, 1]*dt
        
        



# Tester å oppdatere posisjonene til partiklene
dt = 0.1
n = int(1E2)
# start = time()
# update_positions(particles, dt)
# print(f'Tid for å oppdatere posisjonene:         {time() - start:.2g} s')
# start = time()
# nojit_update_positions(nojit_particles, dt, n)
# print(f'Tid for å oppdatere nojit posisjonene:   {time() - start:.2g} s')
start = time()
update_positions_mat(pos_matrix, dt, n)
print(f'Tid for å oppdatere matrise posisjonene: {time() - start:.2} s')
# r = 0.5
# pos_p, vel_p = collision_simulator(pos_matrix, vel_matrix, dt, r)