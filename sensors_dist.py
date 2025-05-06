# Duygu Sap
# Feb 2025

import matplotlib.pyplot as plt

points = [(0.5, 1.813), (2, 0.35), (2.346, 1.75), (1.25, 0.45), (1.25, 2), (1.5, 1.1), (1, 1.240)]

x, y = zip(*points)


plt.scatter(x[:-1], y[:-1], color='blue')
plt.scatter(x[-1], y[-1], color='red')


for i, (xi, yi) in enumerate(zip(x, y)):
    label = str(i + 1) if i < len(x) - 1 else '9'

    offset_x = 0.1
    offset_y = 0.1

    if label == '9':
        offset_x = -0.1
        offset_y = -0.1

    plt.text(xi + offset_x, yi + offset_y, label, fontsize=12, ha='center', va='center')

plt.title('Sensors Distribution')
plt.xlabel('x')
plt.ylabel('y')


plt.grid(True)

plt.xlim(0, 3)
plt.ylim(0, 3)


plt.show()
