import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# Get inputs
N = int(input("Dimension: "))
S = int(input("Step size: "))

# Create a black matrix
m = np.zeros(shape=(N, N, 3), dtype=np.uint8)
for i in range(0, 1 * N // 4):
    for j in range(0, N):
        m[i][j] = [255, 255, 255]

# m[i][j] (j goes to the right; i goes down)

i = 0
half_size = S // 2
while True:

    if i + half_size > N:
        break

    # Color with white the a step
    for t in range(half_size):
        for u in range(half_size):
            m[1 * N // 4 + t][i + u] = [255, 255, 255]

    i += half_size
    if i + half_size > N:
        break

    # Color with black the other step
    for t in range(half_size):
        for u in range(half_size):
            m[1 * N // 4 - t][i + u] = [0, 0, 0]

    i += half_size


# Rotate image for saving
m = np.rot90(m, 3)
img = Image.fromarray(m, 'RGB')

img.save('pre-fractal.png')