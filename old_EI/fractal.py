from matplotlib import pyplot as plt
from math import sqrt
import numpy
from PIL import Image
from matplotlib import colors


def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGB", (w, h), buf.tostring())

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = numpy.fromstring(fig.canvas.tostring_rgb(), dtype=numpy.uint8)
    buf.shape = (w, h, 3)

    return buf

def seg_modify(A,B):
    C,E,X,AB,D = [0,0], [0,0], [0,0], [0,0], [0,0]
    AB[0], AB[1] = B[0]-A[0], B[1]-A[1]
    L = sqrt(AB[0]**2 + AB[1]**2)
    C[0], C[1] = A[0]+AB[0]/3, A[1]+AB[1]/3
    E[0], E[1] = A[0]+AB[0]*2/3, A[1]+AB[1]*2/3
    X[0], X[1] = -AB[1], AB[0] # AB ortogonal
    D[0], D[1] = (A[0]+B[0])/2+sqrt(3)*-X[0]/6, (A[1]+B[1])/2+sqrt(3)*-X[1]/6
    return C,D,E

# initial eq. triangle
big_data = [[]]
big_data[0] = [[0,0],[2,0],[1,sqrt(3)],[0,0]]
number_iterations = 9
for it in range(1,number_iterations+1):
    print('iteraction',it)
    big_data.append([big_data[it-1][0]]) #insert the new iteration in the list
    for i in range(len(big_data[it-1])-1):
        C,D,E = seg_modify(big_data[it-1][i], big_data[it-1][i+1])
        big_data[it].append(C)
        big_data[it].append(D)
        big_data[it].append(E)
        big_data[it].append(big_data[it-1][i+1])
    dataT = list(map(list, zip(*big_data[it]))) #transpose list of lists

figure = plt.figure()
plt.plot(dataT[0],dataT[1])
rgb_image = fig2data(figure)
plt.show()

x, y, z = rgb_image.shape
print(x, y, z)
metade = int(y/2)
print(metade)

new = numpy.copy(rgb_image)
img = Image.fromarray(new, 'RGB')
img.save('fractal_x__.png')
img = Image.fromarray(rgb_image, 'RGB')
img.save('fractal_x._.png')
for i in range(x):
    for j in numpy.arange(0, metade):
        new[i][j] = [0, 0, 0]

img = Image.fromarray(new, 'RGB')
img.save('fractal_x_.png')