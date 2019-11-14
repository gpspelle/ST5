import sys
import numpy
from matplotlib import pyplot as plt
from compute_alpha_w import compute_alpha

if __name__ == '__main__':
    print('Computing alpha for a range of frequencies')
    omegas = numpy.logspace(numpy.log10(600), numpy.log10(30000), num=40)
    alphas = [ compute_alpha(omega) for omega in omegas ]

    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(omegas, numpy.real(alphas))
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\operatorname{Re}(\bar{\alpha})$')

    plt.subplot(1, 2, 2)
    plt.plot(omegas, numpy.imag(alphas))
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\operatorname{Im}(\bar{\alpha})$')

    plt.savefig('alpha_ISOREL.png')
