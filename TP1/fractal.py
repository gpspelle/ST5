import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# Get inputs
N = int(input("Dimension: "))
S = int(input("Step size: "))
G = int(input("How many gen: "))

# Create a black matrix
m = np.zeros(shape=(N, N, 3), dtype=np.uint8)
for i in range(0, 1 * N // 4):
    for j in range(0, N):
        m[i][j] = [255, 255, 255]

points = [0, N]

for g in range(G):
    save_points = []
    for i in range(0, len(points[:-1]), 2):

        # Calculate the fractal points
        x1 = points[i]
        x2 = points[i + 1]
        half_size = S // 2
        half_delta = (x2 + x1) // 2
        p1 = half_delta - half_size
        p2 = half_delta + half_size

        # Color with white the a step
        for t in range(half_size):
            for u in range(half_size):
                m[N // 2 - t - 1][p1 + u] = [0, 0, 0]

        # Color with black the other step
        for t in range(half_size):
            for u in range(half_size):
                m[N // 2 + t][p2 - u] = [255, 255, 255]

        # Add the calculated points for next generations of fractal
        save_points.append(p1)
        save_points.append(p2)

    # Make the step smaller
    S = S // 2
    points = points + save_points
    points.sort()

# Rotate image for saving
m = np.rot90(m, 3)
img = Image.fromarray(m, 'RGB')

img.save('fractal' + str(G) + '.png')