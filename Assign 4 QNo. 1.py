import numpy as np
import matplotlib.pyplot as plt

n = 1000  # You can change this value
dice1 = np.random.randint(1, 7, n)  # Simulating die 1
dice2 = np.random.randint(1, 7, n)  # Simulating die 2
sums = dice1 + dice2  # Sum of two dice
h, h2 = np.histogram(sums, bins=range(2, 14))
plt.bar(h2[:-1], h / n)  # Normalized frequency
plt.xlabel('Sum of Dice')
plt.ylabel('Relative Frequency')
plt.title(f'Histogram of Dice Sums (n={n})')
plt.show()
