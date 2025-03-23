import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft

# Параметри
n = 500  # Довжина сигналу
Fs = 1000  # Частота дискретизації
F_max = 5  # Максимальна частота
Dt_values = [2, 4, 8, 16]  # Кроки дискретизації
F_filter = 12  # Полоса фільтра
M_values = [4, 16, 64, 256]  # Рівні квантування

# Генерація випадкового сигналу
np.random.seed(42)
raw_signal = np.random.normal(0, 10, n)
w = F_max / (Fs / 2)
lowpass_filter = signal.butter(3, w, 'low', output='sos')
filtered_signal = signal.sosfiltfilt(lowpass_filter, raw_signal)

discrete_signals = []
discrete_spectrums = []
restored_signals = []
errors = []
SNR_values = []
quantized_signals = []
quantization_errors = []
quantization_SNR = []

for Dt in Dt_values:
        discrete_signal = np.zeros(n)
        for i in range(0, n, Dt):
                discrete_signal[i] = filtered_signal[i]
        discrete_signals.append(discrete_signal)

        spectrum = fft.fft(discrete_signal)
        discrete_spectrums.append(np.abs(fft.fftshift(spectrum)))

        w_norm = F_filter / (Fs / 2)
        filter_params = signal.butter(3, w_norm, 'low', output='sos')
        restored_signal = signal.sosfiltfilt(filter_params, discrete_signal)
        restored_signals.append(restored_signal)

        error = restored_signal - filtered_signal
        errors.append(np.var(error))
        SNR_values.append(np.var(filtered_signal) / np.var(error))

# Квантування
for M in M_values:
        delta = (np.max(filtered_signal) - np.min(filtered_signal)) / (M - 1)
        quantized_signal = delta * np.round(filtered_signal / delta)
        quantized_signals.append(quantized_signal)

        error = quantized_signal - filtered_signal
        quantization_errors.append(np.var(error))
        quantization_SNR.append(np.var(filtered_signal) / np.var(error))

# Побудова графіків цифрових сигналів з різними рівнями квантування
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
for i in range(2):
        for j in range(2):
                idx = i * 2 + j
                ax[i, j].plot(np.arange(n), quantized_signals[idx], label=f'M={M_values[idx]}')
                ax[i, j].set_title(f'Квантування з M={M_values[idx]}')
                ax[i, j].legend()
plt.tight_layout()
plt.savefig('C:/Users/Евгений/IdeaProjects/TIC_Vansovich_529/SignalProcessing/figures/quantized_signals_grid.png', dpi=600)

plt.figure(figsize=(10, 5))
plt.plot(M_values, quantization_errors, marker='o', linestyle='-')
plt.xlabel('Кількість рівнів M')
plt.ylabel('Дисперсія помилки')
plt.title('Залежність дисперсії від рівнів квантування')
plt.savefig('C:/Users/Евгений/IdeaProjects/TIC_Vansovich_529/SignalProcessing/figures/quantization_error_variance.png', dpi=600)

plt.figure(figsize=(10, 5))
plt.plot(M_values, quantization_SNR, marker='o', linestyle='-')
plt.xlabel('Кількість рівнів M')
plt.ylabel('Співвідношення сигнал-шум')
plt.title('Залежність SNR від рівнів квантування')
plt.savefig('C:/Users/Евгений/IdeaProjects/TIC_Vansovich_529/SignalProcessing/figures/quantization_snr.png', dpi=600)

