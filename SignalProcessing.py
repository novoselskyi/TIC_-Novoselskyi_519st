import numpy as np
from scipy import signal, fft
import matplotlib.pyplot as plt
import os

def generate_signal(n, max_frequency, Fs):
    random_signal = np.random.normal(0, 10, n)
    time_values = np.arange(n) / Fs
    w = max_frequency / (Fs / 2)
    filter_params = signal.butter(3, w, 'low', output='sos')
    filtered_signal = signal.sosfiltfilt(filter_params, random_signal)
    return time_values, filtered_signal

def generate_signal(n, max_frequency, Fs):
    random_signal = np.random.normal(0, 10, n)
    time_values = np.arange(n) / Fs
    w = max_frequency / (Fs / 2)
    filter_params = signal.butter(3, w, 'low', output='sos')
    filtered_signal = signal.sosfiltfilt(filter_params, random_signal)
    return time_values, filtered_signal

if not os.path.exists("figures"):
    os.makedirs("figures")

variances = []  # Визначення масиву для збереження дисперсій
snr_ratios = []  # Визначення масиву для збереження співвідношень сигнал-шум

if __name__ == "__main__":
    n = 500
    F_max = 21  # Новий параметр
    Fs = 1000

    time_values, filtered_signal = generate_signal(n, F_max, Fs)

    # Визначення M_values
    M_values = [4, 16, 64, 256]

    # Відображення результатів квантування та розрахунку дисперсій
    fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
    variances = []  # Визначення масиву для збереження дисперсій
    snr_ratios = []  # Визначення масиву для збереження співвідношень сигнал-шум

    for s in range(4):
        M = M_values[s]
        delta = (np.max(filtered_signal) - np.min(filtered_signal)) / (M - 1)
        quantize_signal = delta * np.round(filtered_signal / delta)

        # Розрахунок бітових послідовностей
        quantize_levels = np.arange(np.min(quantize_signal), np.max(quantize_signal) + 1, delta)
        quantize_bit = [format(bits, '0' + str(int(np.log2(M))) + 'b') for bits in np.arange(0, M)]
        quantize_table = np.c_[quantize_levels[:M], quantize_bit[:M]]

        # Збереження таблиці квантування
        fig_table, ax_table = plt.subplots(figsize=(14 / 2.54, M / 2.54))
        table = ax_table.table(cellText=quantize_table, colLabels=['Значення сигналу', 'Кодова послідовність'],
                               loc='center')
        table.set_fontsize(14)
        table.scale(1, 2)
        ax_table.axis('off')
        plt.savefig(f"./figures/Таблиця квантування для {M} рівнів.png", dpi=600)
        plt.close(fig_table)

        # Кодування сигналу
        bits = []
        for signal_value in quantize_signal:
            for index, value in enumerate(quantize_levels[:M]):
                if np.round(np.abs(signal_value - value), 0) == 0:
                    bits.append(quantize_bit[index])
                    break

        bits = [int(item) for item in list(''.join(bits))]

        # Побудова графіку бітових послідовностей
        fig_bits, ax_bits = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
        ax_bits.step(np.arange(0, len(bits)), bits, linewidth=0.1)
        ax_bits.set_xlabel('Відліки')
        ax_bits.set_ylabel('Бітова послідовність')
        ax_bits.set_title(f'Бітова послідовність для {M} рівнів квантування')
        plt.savefig(f"./figures/Бітова послідовність для {M} рівнів.png", dpi=600)
        plt.close(fig_bits)

        # Збереження даних для графіків
        variances.append(np.var(quantize_signal))
        snr_ratios.append(np.var(filtered_signal) / np.var(quantize_signal))

        # Відображення цифрового сигналу
        i, j = divmod(s, 2)
        ax[i][j].step(time_values, quantize_signal, linewidth=1, where='post', label=f'M = {M}')

    fig.suptitle("Цифрові сигнали з різними рівнями квантування", fontsize=14)
    fig.supxlabel("Час", fontsize=14)
    fig.supylabel("Амплітуда цифрового сигналу", fontsize=14)

    # Збереження зображення
    plt.savefig("figures/Цифрові сигнали з різними рівнями квантування.png", dpi=600)

    # Показати графіки
    plt.show()

    # Графік дисперсії
    fig_variance, ax_variance = plt.subplots(figsize=(10, 6))
    ax_variance.plot(M_values, variances, marker='o', color='b', label='Дисперсія цифрового сигналу')
    ax_variance.set_xlabel('Кількість рівнів квантування')
    ax_variance.set_ylabel('Дисперсія')
    ax_variance.set_xscale('log', base=2)
    ax_variance.legend()
    plt.title("Залежність дисперсії цифрового сигналу від кількості рівнів квантування")

    # Збереження графіку
    plt.savefig("figures/Дисперсія цифрового сигналу.png", dpi=600)

    # Показати графік
    plt.show()

    # Графік співвідношення сигнал-шум
    fig_snr, ax_snr = plt.subplots(figsize=(10, 6))
    ax_snr.plot(M_values, snr_ratios, marker='o', color='r', label='Співвідношення сигнал-шум')
    ax_snr.set_xlabel('Кількість рівнів квантування')
    ax_snr.set_ylabel('Співвідношення сигнал-шум')
    ax_snr.set_xscale('log', base=2)
    ax_snr.legend()
    plt.title("Залежність співвідношення сигнал-шум від кількості рівнів квантування")

    # Збереження графіку
    plt.savefig("figures/Співвідношення сигнал-шум.png", dpi=600)

    # Показати графік
    plt.show()