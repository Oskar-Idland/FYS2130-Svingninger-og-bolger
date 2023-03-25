import numpy as np
import matplotlib.pyplot as plt 
import os
import numpy.fft as fft
from numpy import sin, cos
import scipy.signal as signal
from scipy.io import wavfile
from numba import njit

filepath = os.path.dirname(__file__)

figpath = os.path.join(filepath, 'fig')

π = np.pi

e = np.exp

'''

Filen er delt opp i 3 deler. Hver oppgave er hver sin funksjon bare for å gjøre det letter å hoppe mellom oppgaver. 

Felles variabler og funksjoner som går på tvers av oppgaver er definert først
'''



# Oppgave 2 funskjsoner)


def f(t):

    #    m         m        Hz        Hz         s         s          s         s

    A1 = 1; A2 = 1.6; f1 = 100; f2 = 160; t1 = 0.2; t2 = 0.6; σ1 = 0.05; σ2 = 0.1;
    return  (

            A1*sin(2*π*f1*t)*e(-((t - t1)/σ1)**2) + 

            A2*sin(2*π*f2*t)*e(-((t - t2)/σ2)**2)
            )





def stft(sample: np.ndarray, f_samp: int , overlap: float, window_size: float, filter: bool = False):
    '''

    :param sample: [Hz] The sampled signal 

    :param f_samp: [Hz] The sampling frequency

    :param overlap: [% / 100] Overlap between the windows

    :param window_size: [s] The size of each window

    :param filter: If the signal should be filtered or not  
    '''
    
    '''

    Does not work entirely as intended. The result matrix should not need to be transposed for the correct result, but it must to match the sample and frequency axis. Some slicing and dicing is also needed to get the correct result.
    '''
    
    t_max                   = int(len(sample))                                # [s] Total time of the signal
    points_per_window       = int(window_size*f_samp)                         # [int] Number of points in each window
    num_frequencies         = int(f_samp*points_per_window/2)                                  # [int] Number of frequencies 
    none_overlapping_points = int(points_per_window * (1 - overlap))            # [int] Number of points not shared

                                                                                # between neighboring windows

    num_windows             = int(1/window_size / (1 - overlap))            # [int] Total number of windows
    result                  = np.zeros((points_per_window, num_windows))    # [n,m] Matrix to store results TODO: Litt usikker på denne
    fade_array = signal.windows.hann(points_per_window) 
    sample_pad = np.concatenate((sample, np.zeros(none_overlapping_points))) 

    for i in range(num_windows):

        # Defining start and end indexes for each window and slice out the desired window

        i_start = int(i * none_overlapping_points)
        i_end = int(i_start + points_per_window)
        sliced_signal = sample_pad[i_start:i_end]
        
        if filter:
            sliced_signal = sliced_signal * fade_array
            
        # Computes the FFT on the sliced signal and stores the result in the result matrix

        ft_slice = fft.fft(sliced_signal)
        result[:, i] = np.abs(ft_slice[:num_frequencies])
        

    # Creates and returns the time and frequency matrix together with the positive frequencies 

    t = np.linspace(num_windows/2, num_windows, int(num_windows))

    freqs = fft.fftfreq(int(points_per_window), d = 1/f_samp)
    pos_freqs = slice(int(np.where(freqs >= 0)[0][-1]))     # Slice object with positive frequencies
    
    freqs = freqs[pos_freqs]
    result = result[pos_freqs,:] / len(sample) * 2          # Slicing the result array to get the correct result.
                                                            # This is because I probably made some mistake somewhere else
                                        

    return t, freqs, result


def Oppgave_1():

    def g(A, f, t):

        return A*np.sin(2*π*f*t)


    # a)

    def del_opp_a():

        A   = 1     # m
        f   = 200   # Hz
        T   = 1     # s
        fs  = 1000  # Hz
        N = fs * T  # Number of samples
        t = np.linspace(0, T, N)
        signal = g(A, f, t)

        plt.figure(figsize=(16, 9))
        font_size = 20
        plt.plot(t, signal)
        plt.title('Sampled Time Series', fontsize = font_size)
        plt.xlabel('Time [s]', fontsize = font_size)
        plt.ylabel('Amplitude [m]', fontsize = font_size)

        plt.savefig(os.path.join(figpath, '1.a.1.pdf'))
        plt.show()
        plt.close()        

        discrete_fourier = 2 * fft.fft(signal) / N
        freq = fft.fftfreq(len(signal), d = 1/fs)
        plt.figure(figsize=(16, 9))
        plt.plot(freq, np.abs(discrete_fourier))
        plt.title('Discrete Fourier Transform', fontsize = font_size)
        plt.xlabel('Samples', fontsize = font_size)
        plt.ylabel('Amplitude [m]', fontsize = font_size)
        plt.savefig(os.path.join(figpath, '1.a.2.pdf'))
        plt.show()
        plt.close()    

    # b)

    def del_opp_b():

        fs = 1000   # Hz
        T = 1       # s
        N = fs * T  # Number of samples
        t = np.linspace(0, T, N)
        f1 = 400    # Hz
        f2 = 1200   # Hz
        f3 = 1600   # Hz
        freqs = [f1, f2, f3]
        plt.figure(figsize=(16, 9))
        for i, f in enumerate(freqs):
            N = fs * T # Number of samples
            signal = g(A, f, t)
            discrete_fourier = 2 * fft.fft(signal) / N
            freq = fft.fftfreq(N, d = 1/fs)
            font_size = 12
            plt.subplot(6,1, 2*i + 1)
            plt.plot(t,signal)
            plt.xlim(0, 0.1)
            plt.subplot(6,1, 2*i + 2)
            plt.xlabel('Samples', fontsize = font_size)
            plt.ylabel('Amplitude [m]', fontsize = font_size)
            plt.plot(freq[:int(len(freq)/2)], np.abs(discrete_fourier)[:int(len(freq)/2)], label=f'{f} Hz') 
            plt.legend()        

        plt.suptitle('Discrete Fourier Transform for Different Frequencies', fontsize = font_size)
        plt.tight_layout()
        plt.savefig(os.path.join(figpath, '1.b.pdf'))
        plt.show()        
    

def Oppgave_2():

    

    # a) 

    def del_opp_a():

        T = 1       # s
        fs = 1000    # Hz
        N = fs * T  # Number of samples
        t = np.linspace(0, T, fs)
        signal = f(t)
        freqs = fft.fftfreq(len(signal), d=1/fs)
        ft = 2 * fft.fft(signal) / N
        fontsize = 20
        
        plt.figure(figsize=(16, 9))
        plt.subplot(2, 1, 1)
        plt.plot(t, signal)
        plt.title('Sampled Frequencies', fontsize=fontsize)
        plt.xlabel('Time [s]', fontsize=fontsize)
        plt.ylabel('Amplitude [m]', fontsize=fontsize)
        plt.subplot(2, 1, 2)
        plt.plot(freqs[:int(len(freqs)/2)], np.abs(ft)[:int(len(freqs)/2)])
        plt.title('Discrete Fourier Transform', fontsize=fontsize)
        plt.xlabel('Frequencies [Hz]', fontsize=fontsize)
        plt.ylabel('Amplitude [m]', fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(os.path.join(figpath, '2.a.pdf'))
        plt.show()


        

    # c)

    def del_opp_c(): 

        window_sizes = [0.02, 0.15]; overlap = 0.5; f_samp = 1000; T = 1; t = np.linspace(0, T, f_samp); sample_signal = f(t);
        

        for i, window_size in enumerate(window_sizes):

            plt.figure(figsize=(16, 9)) 
            windows, freqs, result = stft(sample_signal, f_samp, overlap, window_size)
            f_nyquist = 75
            fontsize = 20
            print(windows.shape, freqs.shape, result.shape)

            plt.pcolormesh(windows, freqs, result, shading='auto')
            # plt.hlines(f_nyquist, freqs[0], freqs[-1], color='r', linestyle='--', label='Nyquist Frequency')
            # plt.hlines(160, 50, 100, color='r', linestyle='--')
            plt.xlabel('Windows', fontsize = fontsize)

            plt.ylabel('Frequency (Hz)', fontsize = fontsize)
            # plt.xlim(0,200)
            # plt.ylim(f_nyquist, 100)
            plt.colorbar()
            # plt.xticks(np.linspace(0,T,10))

            # plt.savefig(os.path.join(figpath, f'2.c.{i+1}.pdf'))

            plt.show()
            # break
            
            
            
            
        # freqs, t, Zxx = signal.stft(sample_signal, f_samp, '0.015')
        # plt.pcolormesh(t,freqs, np.abs(Zxx), shading='auto')
        # plt.show()
    # d)

    def del_opp_d():

        window_sizes = [0.02, 0.15]; overlap = 0.5; f_samp = 1000; T = 1; t = np.linspace(0, T, f_samp); sample_signal = f(t);

        for i, window_size in enumerate(window_sizes):

            plt.figure(figsize=(16, 9)) 

            t, freqs, result = stft(sample_signal, f_samp, overlap, window_size, filter = True)

            fontsize = 20

            print(t.shape, freqs.shape, result.shape)

            plt.pcolormesh(freqs, t, result, shading='auto')

            plt.xlabel('Time (s)', fontsize = fontsize)

            plt.ylabel('Frequency (Hz)', fontsize = fontsize)

            plt.title(f'Window size: {window_size} with Hamming Window')

            plt.colorbar()

            plt.savefig(os.path.join(figpath, f'2.d.{i+1}.pdf'))

            plt.show()
    

    # e)

    def del_opp_e():

        window_sizes = [0.02, 0.15]; overlap = 0.5; f_samp = 1000; T = 1; t = np.linspace(0, T, f_samp); sample_signal = f(t);

        for i, window_size in enumerate(window_sizes):

            plt.figure(figsize=(16, 9)) 

            t, freqs, result = stft(sample_signal, f_samp, overlap, window_size)

            fontsize = 20

            print(t.shape, freqs.shape, result.shape)

            plt.pcolormesh(t,freqs, result, shading='gouraud')

            plt.xlabel('Time (s)', fontsize = fontsize)

            plt.ylabel('Frequency (Hz)', fontsize = fontsize)

            plt.title(f'Window size: {window_size} with Gouraud Shading', fontsize = fontsize)

            plt.colorbar()

            plt.tight_layout()

            plt.savefig(os.path.join(figpath, f'2.e.{i+1}.pdf'))

            plt.show()
            
    # del_opp_a()
    del_opp_c()
    # del_opp_d()
            

def Oppgave_3():
    

    samplerate , data = wavfile.read('mistle_thrush.wav') # duetrost

    x_n = data [: , 0] # velg en av to kanaler
    f_samp = samplerate

    T = len(x_n)/f_samp 

    t = np.linspace(0, T, len(x_n))
    

    # a)

    def del_opp_a():

        fontsize = 16

        plt.figure(figsize=(16, 9))

        plt.subplot(2,1,1)

        plt.plot(t,x_n)

        plt.title('Mistle Thrush', fontsize=fontsize)

        plt.xlabel('Time [s]', fontsize=fontsize)

        plt.ylabel('Amplitude [m]', fontsize=fontsize)

        plt.subplot(2,1,2)

        fourier = fft.fft(x_n)

        freqs = fft.fftfreq(len(x_n), d = 1/f_samp)

        plt.plot(freqs[:int(len(freqs)/2)], np.abs(fourier)[:int(len(freqs)/2)])

        plt.title('Discrete Fourier Transform', fontsize=fontsize)

        plt.xlabel('Frequencies [Hz]', fontsize=fontsize)

        plt.ylabel('Amplitude [m]', fontsize=fontsize)

        plt.tight_layout()

        plt.savefig(os.path.join(figpath, '3.a.pdf'))

        plt.show()

        plt.close()
    

    # b)
    '''

    The sample frequency is 48000 Hz

    People can hear frequencies between 20 Hz and 20 000 Hz

    A sample frequency of 48 000 Hz is therefore sufficient to capture sounds we can hear 

    as its Nyquist frequency is 24 000 Hz which is above the highest frequency we can hear.
    '''    
    

    # c)

    def del_opp_c():

        x_n = data [np.int_(t1*samplerate):np.int_(t2*samplerate),0]

            # [s]       [s]

        t1 = 0.1; t2 = 0.4;

        shadings = ['gouraud', 'auto']; window_size = 0.02; overlap = 0.5; 

        t, freqs, result = stft(x_n, f_samp, overlap, window_size)

        fontsize = 20

        for i, shade in enumerate(shadings):

            plt.pcolormesh(freqs, t, result, shading=shade)

            plt.xlabel('Samples', fontsize = fontsize)

            plt.ylabel('Frequency (Hz)', fontsize = fontsize)

            plt.title(f'STFT with {shade} Shading')

            plt.colorbar()

            plt.savefig(os.path.join(figpath, f'3.d.{i+1}.pdf'))

            plt.show()
            


if __name__ == '__main__':

    # Oppgave_1()

    Oppgave_2()

    # Oppgave_3()