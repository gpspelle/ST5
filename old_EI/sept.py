# Find an “e-optimal” form on the chosen range of frequencies obtained by minimizing
# ∑(j=1 to n) of J(Ω, ωj) for ωj the local maxima of the reference energy.

# Une forme Γ∗ ∈ Uad est dite ε-optimale pour la plage des fréquences [ω0,ω1], si pour tout ω ∈[ω0,ω1]
# on a

import numpy as np
from scipy.optimize import minimize
from PIL import Image
import os
import imageio

# TODO: import teach_fem package

# TODO: everything below this
# define initial and final frequence values
w0 = 1
w1 = 2000 # TODO: check this value for cars and trucks

# create a range of frequences
w = [i for i in range(w0, w1)]
it = 0
max_it = 10
id = None

initial_edge = 'pre-fractal.png'

'''
    input:
    - edge - current edge in study
    output:
    - sum - total sum of acoustic energy for this edge in a frequency range
'''


def sum_edge_energy(edge):
    global it
    global id
    global w

    im = Image.fromarray(edge)
    im.save(id + '/edge_' + str(it) + '.png')
    it += 1
    sum_ = 0

    for wj in w:
        # TODO: in fact, J is the teach_fem result using this specific bord and this frequence w
        # TODO: sum += (J(edge_opt), J(edge_star, wj))
        sum_ += J(edge, wj)

    return sum_

# use scipy.optimize.mimize to mimize edge_etoile
# edge is a matrix

'''
    input:
    - edge0 - initial edge from which we start our minimization process
    output:
    - edge* - final e-optimal edge
'''


def minimization(edge0):

    # Calculate the edge that minimizes the sum of energy in a frequency range
    for i in range(max_it):
        edge_star = (edge0)
    edge_star = minimize(lambda e: sum_edge_energy(e), x0=edge0)
    return edge_star


def main():
    global id
    print("**********************************************")
    print("Calculate question 7 from EI - CentraleSupélec")
    print("**********************************************")
    print("         Ensemble on va plus loin")
    print("**********************************************")
    id = input("Please, enter with an identifier: ")

    if len(id) == 0:
        id = "X"

    path = os.getcwd()
    if not os.path.exists(os.path.join(path, id)):
        os.makedirs(os.path.join(path, id))

    edge0 = imageio.imread(initial_edge)
    edge_star = minimization(edge0)
    im = Image.fromarray(edge_star)
    im.save(id + '/edge_star.png')


if __name__ == '__main__':
    main()