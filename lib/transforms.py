import numpy as np


def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np + 1] *= phases
    f[-1:-1 - Np:-1] = np.conj(f[1:Np + 1])
    return np.fft.ifft(f).real


def lowfreqnoise(ecg, min_freq, max_freq):
    """https://stackoverflow.com/a/35091295"""
    samples = ecg.shape[1]
    samplerate = samples
    ecg_new = np.zeros_like(ecg)
    freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
    f[idx] = 1

    for i_lead, lead in enumerate(ecg):
        noise = fftnoise(f)
        ecg_new[i_lead] = noise + lead
        #print(f"Adding noise of mean {np.mean(noise)} and SD {np.std(noise)} to ECG of SD {np.std(lead)}")
    return ecg_new
