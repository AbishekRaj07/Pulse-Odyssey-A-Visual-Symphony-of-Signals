import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import upfirdn
from matplotlib.animation import FuncAnimation

sps = 8
t = np.arange(-3, 3, 1/sps)
t_extended = np.arange(0, 20, 1/sps)

def rectangular_pulse(t):
    return np.where(np.abs(t) < 0.5, 1.0, 0.0)

def gaussian_pulse(t, sigma=0.5):
    return np.exp(-t**2 / (2 * sigma**2))

def raised_cosine(t, beta=0.35, T=1):
    h = np.zeros_like(t)
    for i in range(len(t)):
        if t[i] == 0.0:
            h[i] = 1.0
        elif beta != 0 and abs(t[i]) == T / (2 * beta):
            h[i] = np.sin(np.pi * t[i] / T) / (np.pi * t[i] / T) * np.pi / 4
        else:
            numerator = np.sin(np.pi * t[i] / T) * np.cos(np.pi * beta * t[i] / T)
            denominator = (np.pi * t[i] / T) * (1 - (2 * beta * t[i] / T)**2)
            h[i] = numerator / denominator if denominator != 0 else 0
    return h

def sinc_pulse(t):
    return np.sinc(t)

def impulse_pulse(t, sigma=0.05):
    return np.exp(-t**2 / (2 * sigma**2))

def chirp_pulse(t):
    return np.sin(2 * np.pi * (0.5 + 0.5 * t) * t)

def triangular_pulse(t):
    return np.maximum(0, 1 - np.abs(t))

def sawtooth_pulse(t):
    return np.where((t >= -0.5) & (t < 0.5), t + 0.5, 0)

def cardiac_pulse(t):
    t_mod = t % 1
    return 0.5 * np.exp(-((t_mod - 0.5)**2) / (2 * 0.1**2))

def pulse_oximetry(t):
    return cardiac_pulse(t)

def emg_pulse(t):
    noise = np.random.normal(0, 0.1, len(t))
    spikes = np.sum([np.exp(-((t - i)**2) / (2 * 0.05**2)) for i in range(1, 20, 2)], axis=0)
    return 0.2 * spikes + noise

def ecg_pulse(t):
    t_mod = t % 1
    p = 0.1 * np.exp(-((t_mod - 0.1)**2) / (2 * 0.05**2))
    qrs = 1.0 * np.exp(-((t_mod - 0.4)**2) / (2 * 0.03**2))
    t_wave = 0.2 * np.exp(-((t_mod - 0.7)**2) / (2 * 0.07**2))
    return p + qrs + t_wave

def neural_pulse(t):
    return np.sum([np.exp(-((t - i)**2) / (2 * 0.02**2)) for i in range(1, 20, 3)], axis=0)

def ultrasound_pulse(t):
    return np.sin(2 * np.pi * 5 * t) * np.exp(-t**2 / (2 * 0.2**2))

def emp_pulse(t):
    return np.exp(-t**2 / (2 * 0.1**2)) * (t >= 0)

def light_pulse(t):
    return np.exp(-t**2 / (2 * 0.01**2))

def acoustic_pulse(t):
    return np.sin(2 * np.pi * 3 * t) * np.exp(-t**2 / (2 * 0.3**2))

def shock_wave_pulse(t):
    return np.exp(-t / 0.1) * (t >= 0)

def thermal_pulse(t):
    return np.exp(-t / 0.2) * (t >= 0)

def particle_pulse(t):
    return np.exp(-t**2 / (2 * 0.05**2))

def pulsar_pulse(t):
    t_mod = t % 1
    return np.exp(-((t_mod - 0.5)**2) / (2 * 0.01**2))

def frb_pulse(t):
    return np.exp(-((t - 2)**2) / (2 * 0.01**2))

def gw_pulse(t):
    return np.sin(2 * np.pi * 2 * t) * np.exp(-t / 0.5) * (t >= 0)

def grb_pulse(t):
    return np.exp(-t / 0.1) * (t >= 0)

def rhythmic_pulse(t):
    t_mod = t % 0.5
    return np.exp(-((t_mod - 0.25)**2) / (2 * 0.01**2))

def audio_pulse_wave(t):
    return np.sign(np.sin(2 * np.pi * 2 * t))

def transient_pulse(t):
    return np.exp(-((t - 2)**2) / (2 * 0.05**2))

def midi_pulse(t):
    t_mod = t % 0.25
    return np.where(np.abs(t_mod - 0.125) < 0.05, 1.0, 0.0)

def seismic_pulse(t):
    return np.exp(-((t - 2)**2) / (2 * 0.2**2))

def volcanic_pulse(t):
    return np.sin(2 * np.pi * 0.5 * t) * np.exp(-t**2 / (2 * 0.5**2))

def hydroacoustic_pulse(t):
    return np.sin(2 * np.pi * 1 * t) * np.exp(-t**2 / (2 * 0.4**2))

def clock_pulse(t):
    return np.sign(np.sin(2 * np.pi * 3 * t))

def data_pulse(t):
    return np.random.choice([0, 1], len(t))

def interrupt_pulse(t):
    return np.where(np.abs(t - 2) < 0.05, 1.0, 0.0)

def weather_radar_pulse(t):
    return np.exp(-((t - 2)**2) / (2 * 0.05**2))

def market_pulse(t):
    return np.random.normal(0, 0.3, len(t)) + np.exp(-((t - 2)**2) / (2 * 0.1**2))

def stimulus_pulse(t):
    return np.exp(-((t - 2)**2) / (2 * 0.05**2))

def control_pulse(t):
    t_mod = t % 0.5
    return np.where(np.abs(t_mod - 0.25) < 0.1, 1.0, 0.0)

def social_media_pulse(t):
    return np.sum([np.exp(-((t - i)**2) / (2 * 0.1**2)) for i in np.random.choice(t, 5)], axis=0)

rc_filter = raised_cosine(t, beta=0.35)
data = np.random.choice([1, -1], size=20)
upsampled_data = upfirdn([1], data, up=sps)
raised_cosine_signal = np.convolve(upsampled_data, rc_filter, mode='same')
signals = [
    ("Rectangular (Engineering)", rectangular_pulse(t_extended)),
    ("Gaussian (Engineering)", gaussian_pulse(t_extended)),
    ("Raised Cosine (Engineering)", raised_cosine_signal),
    ("Sinc (Engineering)", sinc_pulse(t_extended)),
    ("Impulse (Engineering)", impulse_pulse(t_extended)),
    ("Chirp (Engineering)", chirp_pulse(t_extended)),
    ("Triangular (Engineering)", triangular_pulse(t_extended)),
    ("Sawtooth (Engineering)", sawtooth_pulse(t_extended)),
    ("Cardiac (Medicine)", cardiac_pulse(t_extended)),
    ("Pulse Oximetry (Medicine)", pulse_oximetry(t_extended)),
    ("EMG (Medicine)", emg_pulse(t_extended)),
    ("ECG (Medicine)", ecg_pulse(t_extended)),
    ("Neural (Medicine)", neural_pulse(t_extended)),
    ("Ultrasound (Medicine)", ultrasound_pulse(t_extended)),
    ("EMP (Physics)", emp_pulse(t_extended)),
    ("Light Pulse (Physics)", light_pulse(t_extended)),
    ("Acoustic (Physics)", acoustic_pulse(t_extended)),
    ("Shock Wave (Physics)", shock_wave_pulse(t_extended)),
    ("Thermal (Physics)", thermal_pulse(t_extended)),
    ("Particle (Physics)", particle_pulse(t_extended)),
    ("Pulsar (Astronomy)", pulsar_pulse(t_extended)),
    ("FRB (Astronomy)", frb_pulse(t_extended)),
    ("Gravitational Wave (Astronomy)", gw_pulse(t_extended)),
    ("GRB (Astronomy)", grb_pulse(t_extended)),
    ("Rhythmic (Music)", rhythmic_pulse(t_extended)),
    ("Audio Pulse Wave (Music)", audio_pulse_wave(t_extended)),
    ("Transient (Music)", transient_pulse(t_extended)),
    ("MIDI (Music)", midi_pulse(t_extended)),
    ("Seismic (Seismology)", seismic_pulse(t_extended)),
    ("Volcanic (Seismology)", volcanic_pulse(t_extended)),
    ("Hydroacoustic (Seismology)", hydroacoustic_pulse(t_extended)),
    ("Clock (Computer Science)", clock_pulse(t_extended)),
    ("Data (Computer Science)", data_pulse(t_extended)),
    ("Interrupt (Computer Science)", interrupt_pulse(t_extended)),
    ("Weather Radar (Meteorology)", weather_radar_pulse(t_extended)),
    ("Market (Economics)", market_pulse(t_extended)),
    ("Stimulus (Psychology)", stimulus_pulse(t_extended)),
    ("Control (Robotics)", control_pulse(t_extended)),
    ("Social Media (Sociology)", social_media_pulse(t_extended))
]

window_size = 40
window_size_data = 5
step_size = 1
interval = 30
frames = (len(t_extended) - window_size) // step_size

current_pulse_idx = 0
fig = None
ani = None

def show_next_pulse(event=None):
    global current_pulse_idx, fig, ani
    if fig is not None:
        plt.close(fig)
    
    if current_pulse_idx >= len(signals):
        return
    
    title, signal = signals[current_pulse_idx]
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 4))
    fig.patch.set_facecolor('black')
    
    ax = fig.add_subplot(111, facecolor='black')
    line, = ax.plot([], [], 'b-', lw=2)
    ax.set_xlim(0, len(signal) - 1)
    ax.set_ylim(np.min(signal) - 0.1, np.max(signal) + 0.1)
    ax.set_title(title, color='white', fontsize=12)
    ax.grid(True, color='gray')
    ax.tick_params(colors='white')
    
    def update(frame):
        start_idx = frame * step_size
        if "Raised Cosine" in title:
            end_idx = min(start_idx + window_size, len(raised_cosine_signal))
            line.set_data(np.arange(start_idx, end_idx), signal[start_idx:end_idx])
        else:
            end_idx = min(start_idx + window_size, len(t_extended))
            line.set_data(np.arange(start_idx, end_idx), signal[start_idx:end_idx])
        return line,
    
    ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    
    fig.canvas.mpl_connect('close_event', show_next_pulse)
    
    current_pulse_idx += 1
    
    plt.tight_layout()
    plt.show()

show_next_pulse()
