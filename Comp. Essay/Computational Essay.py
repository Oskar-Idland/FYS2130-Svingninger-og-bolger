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
    
# Tester to måter å lage en liste med partikler på
N = 1_000_000
start = time()
particles = [Particle(random(), random(), random(), random()) for _ in range(N)]

print(f'Numba dataclass:    {time() - start:.2g} s')
start = time()
nojit_particles = [nojit_Particle(random(), random(), random(), random()) for _ in range(N)]
print(f'Vanlig dataclass:   {time() - start:.2g} s')

# Tester å bare lagre alt i en matrise
start = time()
pos_matrix = np.random.rand(N, 2)
print(f'Matrise:            {time() - start:.2g} s')

# Henter ut posisjonene til partiklene og lagrer som matrise for plotting
start = time()
positions = np.array([[p.x, p.y] for p in particles])
print(f'Tid for å hente ut posisjonene: {time() - start:.2g} s')

start = time()
plt.scatter(positions[:, 0], positions[:, 1])
print(f'Tid for å plotte posisjonene:   {time() - start:.2g} s')
plt.close()

@func_go_brrr
def update_positions(particles, dt):
    for p in particles:
        p.x += p.vx * dt    
        p.y += p.vy * dt
        
def nojit_update_positions(test_particles, dt):
    for p in test_particles:
        p.x += p.vx * dt
        p.y += p.vy * dt


# Tester å oppdatere posisjonene til partiklene
start = time()
update_positions(particles, 0.1)
print(f'Tid for å oppdatere posisjonene:        {time() - start:.2g} s')
start = time()
nojit_update_positions(nojit_particles, 0.1)
print(f'Tid for å oppdatere nojit posisjonene:  {time() - start:.2g} s')