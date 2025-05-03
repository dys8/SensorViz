import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas
import matplotlib
matplotlib.use("TkAgg")

from ripser import ripser
from statsmodels.tsa.stattools import acf


def sliding_window_embedding(time_series, embedding_dim, tau):
    N = len(time_series)
    point_cloud = []
    print(f"\n--- New embedding call ---")
    print(f"Time series length: {N}, embedding_dim: {embedding_dim}, tau: {tau}")
    print(f"Max starting index t: {N - (embedding_dim - 1) * tau - 1}")

    for t in range(N - (embedding_dim - 1) * tau):
        indices = [t + i * tau for i in range(embedding_dim)]
        print('t:', t, 'Indices used:', indices)
        window = [time_series[idx] for idx in indices]
        point_cloud.append(window)

    return np.array(point_cloud)

def compute_persistence_diagram(point_cloud):
    return ripser(point_cloud, maxdim=1)['dgms']

def compute_tau(time_series, threshold=0.3, max_lags=100):
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

input_filename = 'averaged_signal_Events_7_4.csv'


with open(input_filename, mode='r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    rows = list(reader)

    for i, row in enumerate(rows[1:]):
        row[0] = str(i)
        rows[i + 1] = [cell.strip('"') for cell in row]


sensor1_array = np.array(rows[1:], dtype=object)

sensor1_array[:, 0] = sensor1_array[:, 0].astype(int)

sensor1_array[:, 1:] = sensor1_array[:, 1:].astype(float)

print(sensor1_array[:5])
print(sensor1_array.dtype)

import pandas as pd
df = pd.read_csv('averaged_signal_Event_7_4.csv')
print(df.columns)
print(df.head())



time, clean_signal= sensor1_array[:, 0] ,sensor1_array[:, 1]
clean_signal_normalized = (clean_signal - np.mean(clean_signal)) / np.std(clean_signal)


plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, clean_signal, label='Original Signal', color='b')
plt.title('Sensor 4 Data: Time vs Temperature During Brake')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, clean_signal_normalized, label='Normalized Signal', color='r')
plt.title('Normalized Sensor 4 Data: Time vs Normalized Temperature')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

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
required_length = (optimal_embedding_dim - 1) * tau + 1
if len(clean_signal_normalized) < required_length:
    raise ValueError(f"Signal too short for embedding: need at least {required_length} samples.")

clean_point_cloud = sliding_window_embedding(clean_signal_normalized, optimal_embedding_dim, tau)




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