import numpy as np 
import matplotlib.pyplot as plt
import os 
from numba import njit
filepath = os.path.dirname(__file__)
π = np.pi

## Oppgave 2


# Dimensjonene til sneglehuset er gitt ved
h0      =    0.1 * 10**(-3) # m
h1      =    0.3 * 10**(-3) # m
w0      =    0.3 * 10**(-3) # m
w1      =    0.1 * 10**(-3) # m
l       =    30  * 10**(-3) # m
ρ0      =    1500           # kg/m^3
ρ1      =    2500           # kg/m^3
k0      =    10**-6             # kg/s^2
k1      =    10**-1             # kg/s^2


n       = 3000
height  = np.linspace(h0, h1, n)
width   = np.linspace(w0, w1, n)
length  = l/n
k       = np.linspace(k0, k1, n)
k_ln    = np.logspace(np.log10(k0), np.log10(k1), n)
ρ       = np.linspace(ρ0, ρ1, n)
dV      = height * width * length
m       = ρ * dV
ω       = np.sqrt(k/m)
ω_ln    = np.sqrt(k_ln/m)


def freq(ω):
    return ω/(2*π)


x = np.linspace(0, 30, n)
plt.plot(x, freq(ω), label = 'Frequency')
plt.xlabel('Length [mm]')
plt.ylabel('Frequency [Hz]')
plt.title('Frequency as a function of length inside the snail shell')
plt.savefig(os.path.join(filepath, 'figs/1.b.pdf'))
# plt.show()
plt.clf()


plt.plot(x, freq(ω_ln), label = 'Frequency using logarithmic k-variance')
plt.xlabel('Length [mm]')
plt.ylabel('Frequency [Hz]')
plt.title('Frequency as a function of length inside the snail shell')
plt.savefig(os.path.join(filepath, 'figs/1.b_ln.pdf'))
# plt.show()
plt.clf()


F   = 1         # N
C4  = 261.63    # Hz
C4S = 277.18    # Hz
ωF1 = 2*π*C4    # rad/s
ωF2 = 2*π*C4S   # rad/s
b   = 1e-8   # Dampening constant


def amp(F, ωF, m, b, ω):
    return (F/m) * 1/(np.sqrt((ω**2 - ωF**2)**2 + (b*ωF/m)**2))
           
plt.figure(figsize = (16, 9))
s = slice(10, 80)
plt.plot(x[s], amp(F, ωF1, m, b, ω)[s], label = 'Amplitude for C4')
plt.plot(x[s], amp(F, ωF2, m, b, ω)[s], label = 'Amplitude for C4S') 

size = 22
plt.xlabel('Length [mm]', fontsize = size)
plt.ylabel('Amplitude [m]', fontsize = size)
plt.title(f'Amplitude as a function of length inside the snail shell with b = {b}', fontsize = size)
plt.savefig(os.path.join(filepath, 'figs/2.b.pdf'))
# plt.show()
plt.clf()

Q = m*k/b**2
plt.figure(figsize = (16,9))
plt.plot(x, Q)
plt.xlabel('Length [mm]', fontsize = size)
plt.ylabel('Q-factor', fontsize = size)
plt.title('Q-factor with linear k-variance', fontsize = size)
plt.savefig(os.path.join(filepath, 'figs/2.c.pdf'))
# plt.show()

ω_r = np.sqrt(ω**2 - (b/(2*m))**2)
T = Q/ω_r
plt.figure(figsize = (16,9))
plt.plot(x, T)
plt.xlabel('Length [mm]', fontsize = size)
plt.ylabel('T [s]', fontsize = size)
plt.title('Time for oscillation to decay', fontsize = size)
plt.savefig(os.path.join(filepath, 'figs/2.d.pdf'))
# plt.show()