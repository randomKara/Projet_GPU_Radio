

import numpy as np
import sys

SAMPLE_RATE = 10e6
DURATION    = 1.0
SNR_DB      = 25

def generate_signal(sample_rate=SAMPLE_RATE, duration=DURATION, snr_db=SNR_DB):
    N = int(sample_rate * duration)
    t = np.arange(N) / sample_rate
    signal = np.zeros(N, dtype=np.complex64)

    print("=== Génération du signal radio multi-canaux ===")
    print(f"  Sample rate : {sample_rate/1e6:.1f} MHz")
    print(f"  Durée       : {duration:.1f} s")
    print(f"  Échantillons: {N:,}")
    print()

    amp_a = 1.0
    mod_a = 0.4 * np.sin(2*np.pi*5e3*t)
    signal += amp_a * np.exp(1j * 2*np.pi * (1.2e6*t + mod_a))
    print(f"  [A] FM  1.200 MHz | 0 dB   | modulé à 5 kHz")

    amp_b = 10**(-10/20)
    carrier_b = np.exp(1j * 2*np.pi * 2.5e6 * t)
    mod_b = 1.0 + 0.5 * np.cos(2*np.pi*3e3*t)
    signal += amp_b * mod_b * carrier_b
    print(f"  [B] AM  2.500 MHz | -10 dB | modulé à 3 kHz")

    amp_c = 10**(-25/20)
    signal += amp_c * np.exp(1j * 2*np.pi * 3.5e6 * t)
    print(f"  [C] CW  3.500 MHz | -25 dB | porteuse pure (signal de référence faible)")

    amp_d = 10**(-8/20)
    f_start, f_end = 3.8e6, 4.2e6
    chirp_phase = 2*np.pi * (f_start * t + 0.5 * (f_end - f_start) / duration * t**2)
    signal += amp_d * np.exp(1j * chirp_phase)
    print(f"  [D] Chirp 3.8→4.2 MHz | -8 dB | balayage FSK linéaire")

    amp_e = 10**(-15/20)
    burst_rate = 200.0
    duty_cycle = 0.3
    envelope_e = ((t * burst_rate) % 1.0 < duty_cycle).astype(np.float32)
    signal += amp_e * envelope_e * np.exp(1j * 2*np.pi * 1.8e6 * t)
    print(f"  [E] Pulse 1.800 MHz | -15 dB | rafales 200 Hz, duty 30%")

    noise_power = 10**(-snr_db / 10)
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(N) + 1j * np.random.randn(N))
    signal += noise

    print(f"\n  Bruit       : SNR={snr_db} dB")
    return signal.astype(np.complex64)

def save_iq(signal, path='signal.bin'):
    N = len(signal)
    iq = np.empty(2 * N, dtype=np.float32)
    iq[0::2] = np.real(signal)
    iq[1::2] = np.imag(signal)
    iq.tofile(path)
    return 2 * N * 4

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'signal.bin'
    sig = generate_signal()
    size = save_iq(sig, out)
    print(f"\n  Sauvegardé : {out} ({size/1e6:.1f} MB)")
    print("=================================================\n")

    fft_size = 1024
    freq_res = SAMPLE_RATE / fft_size
    print("Bins FFT attendus (pour vérification CFAR) :")
    for name, freq in [('A-FM',1.2e6),('B-AM',2.5e6),('C-CW',3.5e6),
                        ('D-Chirp_start',3.8e6),('D-Chirp_end',4.2e6),
                        ('E-Pulse',1.8e6)]:
        print(f"  [{name}] {freq/1e6:.2f} MHz → bin {int(freq/freq_res)}")

if __name__ == '__main__':
    main()
