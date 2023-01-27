import matplotlib.pyplot as plt
import numpy as np
import scipy.misc 
import os
real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)
Δ = scipy.misc.derivative
m = 2 # Massen i kg

def v(t: float) -> float:
    '''Regner ut hastighet som en funskjon av tid'''
    return Δ(x, t, dx=1e-6)

def p(t: float) -> float:
    '''Regner ut bevegelsesmengde som en funksjon av tid '''
    return m*v(t)   

def x(t: float) -> float:
    '''Regner ut posisjon som en funksjon av tid'''
    return 1.08*np.cos(2*t + 1.19)


t = np.linspace(0, np.pi, 1000)
plt.plot(x(t), p(t))
plt.xlabel('Posisjon x [m]')
plt.ylabel('Bevegelses mengde p [kg m/s]')
plt.axis('equal')
plt.savefig(os.path.join(dir_path, 'Figures/2_c.pdf'))

plt.clf()

plt.plot(x(t)/x(0), p(t)/((v(0)*m)))
plt.xlabel('Posisjon x/x(0)')
plt.ylabel('Bevegelses mengde p/(v(0)m)')
plt.axis('equal')
plt.savefig(os.path.join(dir_path, 'Figures/2_d.pdf'))
