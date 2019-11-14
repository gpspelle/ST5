from scipy import integrate
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from mpmath import *
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec

# Coded by Gabriel Pellegrino. During the TP of the ST5: Pollution Acoustique
# et Contrôle des Ondes at CentraleSupélec (2019).
# This code calculate alpha, a parameter that depends on the wave frequency and is
# related to a material's phorosity, resistivity and tortuosity.

# Callback class used to stop the scipy.optimize.minize process after the error is under the
# threshold tolerance.
class StopOptimizingException(Exception):
    pass

class CallbackCollector:

    def __init__(self, f, thresh):
        self._f = f
        self._thresh = thresh

    def __call__(self, xk):
        if self._f(xk) < self._thresh:
            self.x_opt = xk
            self.v = self._f(xk)
            raise StopOptimizingException()


# Constants used during the numerical analysis
mp.dps = 25

gamaP = 1.4                     # air constant
ro0 = 1.2                       # air constant
mi0 = 1                         # air constant
a0 = 0                          # air constant
ksi0 = 0.000008650519031141869  # air constant

l = 0.5 # grid size
L = 2*l
deltaX = 0.1 # step size in the grid
A = B = 1 # imposed
c0 = 340  # velocity of sound in air
w0 = 340 # first value of frequency
wf = 20010 # last value of frequency
wst = 340  # step size for frequency


def integrand_real(y, k):
    return (1 / ((2 * np.pi) ** 0.5)) * exp(-((y - 1 / 2) / 2) * (y - 1 / 2) / 2) * exp(-1j * k * np.pi / l).real


def integrand_imag(y, k):
    return (1 / ((2 * np.pi) ** 0.5)) * exp(-((y - 1 / 2) / 2) * (y - 1 / 2) / 2) * exp(-1j * k * np.pi / l).imag


def g(k):
    integral_r = integrate.quad(integrand_real, -l, l, args=k) # integrating in the real part
    integral_i = integrate.quad(integrand_imag, -l, l, args=k) # integrating in the imag part
    return (1/L) * (integral_r[0] + integral_i[0]) # return the sum of both


def lambda0(k, omega):
    x = complex(k * k - ksi0 * omega * omega / mi0) ** 0.5

    # removing math imprecision from pure real or pure complex numbers
    if x.imag > x.real:
        return (0 + x.imag*j)
    else:
        return (x.real + 0j)


def f(l0, alpha):
    return (l0 * mi0 - alpha) * exp(-l0 * L) + (l0 * mi0 + alpha) * exp(l0 * L)


def lambda1(k, omega, ksi, mi, a):

    # below a lot of math steps to calculate lambda1
    kuww = (ksi * omega * omega) / mi
    awu_awu = (a * omega / mi) * (a * omega / mi)
    inside_term = (complex(((k*k - kuww)**2) + awu_awu)) ** 0.5
    first_term = (complex((k*k - kuww + inside_term) / 2)) ** 0.5
    second_term = (complex((k*k - kuww - inside_term) / 2)) ** 0.5

    return first_term - second_term


def X(gk, falpha, fl1mi, l0, l1, mi, alpha):
    return gk * (((l0 * mi0 - l1 * mi) / fl1mi) - ((l0 * mi0 - alpha) / falpha))


def eta(gk, falpha, fl1mi, l0, l1, mi, alpha):
    return gk * (((l0 * mi0 + l1 * mi) / fl1mi) - ((l0 * mi0 + alpha) / falpha))


def calc_error(ksi, mi, a, omega, alpha):
    error = 0
    for n in np.arange(-L/deltaX, L/deltaX): # loop over all wave numbers

        # below a lot of math steps to calculate the error
        k = (n * np.pi) / L
        l0 = lambda0(k, omega)
        l1 = lambda1(k, omega, ksi, mi, a)
        
        #gk = g(k)
        gk = 1
        falpha = complex(f(l0, alpha))
        fl1mi = complex(f(l0, l1*mi))
        X_ = complex(X(gk, falpha, fl1mi, l0, l1, mi, alpha))
        eta_ = complex(eta(gk, falpha, fl1mi, l0, l1, mi, alpha))
        absk2 = k * k
        absX2 = abs(X_) * abs(X_)
        absN2 = abs(eta_) * abs(eta_)

        if k*k >= (ksi0/mi0) * omega*omega:
            error += ( (A + B * absk2) * ( (1/(2*l0)) * (absX2 * (1 - exp(-2 * l0 * L)) + absN2 * (-1 + exp(2 * l0 * L))) + 2 * L * complex((X_* eta_.conjugate())).real))
            error += (B * (l0 / 2)* (absX2 * (1 - exp(-2 * l0 * L)) + absN2 * (exp(2 * l0 * L) - 1)))
            error -= (2* B * l0 * l0 * L * complex((X_ * eta_.conjugate())).real)
        else:
            error += ((A + B * absk2) * ((L * (absX2 + absN2)) + (complex(1j) / l0) * complex((X_ * eta_.conjugate() * (1 - exp(-2 * l0 * L)))).imag))
            error += (B * L * l0 * l0 * (absX2 + absN2))
            error += (B * l0 * (complex(1j)) * complex((X_ * eta_.conjugate() * (1 - exp(-2 * l0 * L)))).imag)


    return np.float64((error.imag * error.imag + error.real*error.real) ** 0.5)


def minimize_error(ksi, mi, a, omega):
    """
        Create a function with only one parameter and all the other constants.
        Note that the last parameter of calc_error is the only variable now (alpha).
    """

    optfunc = partial(calc_error, ksi, mi, a, omega)
    #try:
        # Use a callback class to control the minimize process and stop after the given threshold
        # of e-3.
    #   cb = CallbackCollector(lambda alpha: optfunc(convert_to_complex(alpha)), thresh=0.001)
        # The initial point is really important, if you don't set it correctly acording to your problem
        # it will lead to a bad value. The best thing you can do is to cancel the callback function and get some points
        # over your interest range. After this, you create a callback function with an appropriate
        # error value and then you calculate the minization closer to the interest point.
        # It will be really faster like this.
    sol = minimize(lambda alpha: optfunc(convert_to_complex(alpha)), x0=convert_to_real(np.ones(1) * 20 - 20j))
    return convert_to_complex(sol.x), sol.fun

    #except StopOptimizingException:
    #return convert_to_complex(cb.x_opt, cb.v
    #sol = minimize(lambda alpha: optfunc(convert_to_complex(alpha)), x0=convert_to_real(np.ones(1)*20 - 20j), options={'disp': False, 'maxiter':10})


def convert_to_complex(z): # real vector of length 2n -> complex of length n
    return z[:len(z)//2] + 1j * z[len(z)//2:]


def convert_to_real(z):    # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))


def calc_alpha(phorosity, resistivity, tortuosity):
    re = []
    im = []
    errors = []
    ksi = phorosity * gamaP / (c0 * c0)
    mi = phorosity / tortuosity
    a = resistivity * phorosity * phorosity * gamaP / (c0 * c0 * ro0 * tortuosity)
    interval = [i for i in range(w0, wf, wst)] # creating the range
    for omega in interval:
        alpha, error = minimize_error(ksi, mi, a, omega) # get alpha and error
        re.append(alpha.real)
        im.append(alpha.imag)
        errors.append(error)
        print(omega, alpha, error) # stdout found values

    # Create 2x2 sub plots
    gs = gridspec.GridSpec(2, 2)

    pl.figure()
    ax = pl.subplot(gs[0, 0])  # row 0, col 0
    pl.plot(interval, re)

    ax = pl.subplot(gs[0, 1])  # row 0, col 1
    pl.plot(interval, im)

    ax = pl.subplot(gs[1,:])  # row 1, span all columns
    pl.plot(interval, errors)

    plt.show()


def main():
    material = input("Which material? ISOREL or B5?")

    if material == "ISOREL":
        calc_alpha(0.7, 142300, 1.15)  # Isorel: (phorosity, resistivity, tortuosity)
    else:
        calc_alpha(0.2, 2124000, 1.22) # B5: (porosity, resistivity, tortuosity)


if __name__ == "__main__":
    main()

