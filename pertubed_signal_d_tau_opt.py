import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("TkAgg")

from ripser import ripser
from statsmodels.tsa.stattools import acf


def generate_time_series2(duration=80, sampling_rate=100, num_peaks=3, noise_std=0.1, noise_prob=0.1):
    """
    Generating synthetic time series with noise
    """
    time = np.linspace(0, duration, duration * sampling_rate)
    signal = np.zeros_like(time)
    peak_times = np.sort(np.random.uniform(0, duration, num_peaks))
    peak_indices = (peak_times * sampling_rate).astype(int)
    signal[peak_indices] = 1
    signal = np.convolve(signal, np.hanning(10), mode='same')[:len(time)]

    noise = np.zeros_like(time)
    noisy_positions = np.random.rand(len(time)) < noise_prob
    noise[noisy_positions] = np.random.normal(0, noise_std, noisy_positions.sum())
    noisy_signal = signal + noise
    return time, signal, noisy_signal, peak_times

def generate_time_series(duration=100, sampling_rate=20, frequency=1, noise_std=0.1, noise_prob=0.1):
    time = np.linspace(0, duration, duration * sampling_rate)
    signal = np.sin(2 * np.pi * frequency * time)

    noise = np.zeros_like(time)
    noisy_positions = np.random.rand(len(time)) < noise_prob
    noise[noisy_positions] = np.random.normal(0, noise_std, noisy_positions.sum())

    noisy_signal = signal + noise

    return time, signal, noisy_signal


def sliding_window_embedding(time_series, embedding_dim, tau):
    """
    Converting a time series into a point cloud using time delay embedding.
    """
    N = len(time_series)
    point_cloud = []
    for t in range(N - (embedding_dim - 1) * tau):
        window = [time_series[t + i * tau] for i in range(embedding_dim)]
        point_cloud.append(window)

    return np.array(point_cloud)

def compute_persistence_diagram(point_cloud):
    return ripser(point_cloud, maxdim=1)['dgms']

def compute_tau(time_series, threshold=0.1, max_lags=100):
    correlation = acf(time_series, fft=True, nlags=max_lags)
    tau = np.argmax(correlation < threshold)

    if plt:
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(correlation)), correlation, marker='o', linestyle='-')
        plt.axhline(y=threshold, color='r', linestyle='--', label=f"Threshold = {threshold}")
        plt.axvline(x=tau, color='g', linestyle='--', label=f"Selected τ = {tau}")
        plt.xlabel("Time-delay (τ)")
        plt.ylabel("ACF")
        plt.title("ACF vs Time-Delay")
        plt.legend()
        plt.grid()
        plt.show()
    return tau


def false_nearest_neighbors(time_series, embedding_dim, tau, max_neighbors=5):
    N = len(time_series)
    embedded_data = sliding_window_embedding(time_series, embedding_dim, tau)

    fnn_count = 0
    for i in range(N - (embedding_dim - 1) * tau):
        distances = np.linalg.norm(embedded_data[i] - embedded_data, axis=1)
        sorted_indices = np.argsort(distances)

        for j in range(1, max_neighbors + 1):
            neighbor = sorted_indices[j]
            if np.linalg.norm(time_series[i] - time_series[neighbor]) > 1.5 * distances[j]:
                fnn_count += 1

    return fnn_count / N


time, clean_signal, noisy_signal = generate_time_series()


clean_signal_normalized = (clean_signal - np.mean(clean_signal)) / np.std(clean_signal)
noisy_signal_normalized = (noisy_signal - np.mean(noisy_signal)) / np.std(noisy_signal)


tau = compute_tau(clean_signal)
print("Optimal tau:", tau)


embedding_dims = range(2, 10)
fnn_ratios = []

for embedding_dim in embedding_dims:
    fnn_ratio = false_nearest_neighbors(clean_signal_normalized, embedding_dim, tau)
    fnn_ratios.append(fnn_ratio)

plt.plot(embedding_dims, fnn_ratios, marker='o')
plt.xlabel("Embedding Dimension")
plt.ylabel("False Nearest Neighbors Ratio")
plt.title("FNN vs Embedding Dimension")
plt.grid()
plt.show()


optimal_embedding_dim = embedding_dims[np.argmin(fnn_ratios)]
print("Optimal embedding dimension:", optimal_embedding_dim)


clean_point_cloud = sliding_window_embedding(clean_signal_normalized, optimal_embedding_dim, tau)
noisy_point_cloud = sliding_window_embedding(noisy_signal_normalized, optimal_embedding_dim, tau)

plt.figure(figsize=(8, 6))
plt.scatter(clean_point_cloud[:, 0], clean_point_cloud[:, 1], alpha=0.6, label="Clean Point Cloud")
plt.scatter(noisy_point_cloud[:, 0], noisy_point_cloud[:, 1], alpha=0.6, label="Noisy Point Cloud")
plt.title("2D Projection of Point Clouds")
plt.xlabel("Embedding Dimension 1")
plt.ylabel("Embedding Dimension 2")
plt.legend()
plt.grid()
plt.show()


diagrams = ripser(clean_point_cloud)["dgms"]

import numpy as np
import persim
import matplotlib.pyplot as plt
from ripser import ripser

diagrams_clean = ripser(clean_point_cloud)

dgms_sensor1 = diagrams_clean['dgms']


persim.plot_diagrams(
    dgms_sensor1,
    labels=['$H_0$', '$H_1$'],
    show=True
)
