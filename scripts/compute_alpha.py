import scipy
from scipy.optimize import minimize
from scipy.integrate import quad
import numpy
import matplotlib.pyplot as plt

def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return scipy.real(func(x))
    def imag_func(x):
        return scipy.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return real_integral[0] + 1j*imag_integral[0]

def real_to_complex(z):      # real vector of length 2n -> complex of length n
    return z[0] + 1j * z[1]

def complex_to_real(z):      # complex vector of length n -> real of length 2n
    return numpy.array([numpy.real(z), numpy.imag(z)])

class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}
    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        #Warning: You may wish to do a deepcopy here if returning objects
        return self.memo[args]

def compute_alpha(omega):
    resolution = 6

    # Defining parameters value
    c_0 = 340.0
    gamma_p = 7.0/5.0
    rho_0 = 1.2
# ISOREL
#    phi = 0.7
#    alpha_h = 1.15
#    sigma = 142300.0
# Melamine
    phi = 0.99
    alpha_h = 1.02
    TC = 20
    TR = 9/5*(273.15+TC)
# Sutherland's law
    a = 0.555
    b = 120
    T0 = 524.07
    mu0 = 1.827*1e-05
    mu = mu0 * (a*T0+b)/(a*TR+b) * (TR/T0)**(3/2)
    k0 = 1.6853*1e-09
    sigma = mu/k0
# Size of the cavity
    L = 0.01
    l = 0.01
# Objective function
    A = 1.0
    B = 1.0

    # Defining other parameters
    mu_0 = 1
    ksi_0 = 1 / ( c_0 ** 2 )
    mu_1 = phi / alpha_h
    ksi_1 = phi * gamma_p / ( c_0 ** 2 )
    a = sigma * ( phi ** 2 ) * gamma_p / ( ( c_0 ** 2 ) * rho_0 * alpha_h )

    # Defining k, omega and alpha dependant parameters as functions
    @Memoize
    def lambda_0(k, omega):
        if k ** 2 >= ( omega ** 2 ) * ksi_0 / mu_0 :
            return numpy.sqrt( k**2 - ( omega ** 2 ) * ksi_0 / mu_0 )
        else:
            return numpy.sqrt( ( omega ** 2 ) * ksi_0 / mu_0 - k ** 2 ) * 1j
    
    @Memoize
    def lambda_1(k, omega):
        truc1 = ( omega ** 2 ) * ksi_1 / mu_1
        truc2 = numpy.sqrt( ( k ** 2 - truc1 ) ** 2 + ( a * omega / mu_1 ) ** 2 )
        real = 1 / numpy.sqrt(2) * numpy.sqrt( k ** 2 - truc1 + truc2 )
        im = -1 / numpy.sqrt(2) * numpy.sqrt( truc1 - k ** 2 + truc2 )
        return complex( real, im )

    # Defining functions
    @Memoize
    def g(y):
        return 1.0

    @Memoize
    def f(x, k):
        return ( ( lambda_0(k, omega) * mu_0 - x ) * numpy.exp( -lambda_0(k, omega) * L ) \
             + ( lambda_0(k, omega) * mu_0 + x ) * numpy.exp( lambda_0(k, omega) * L ) )
    
    @Memoize
    def g_k(k): 
        if k == 0:
            return 1.
        else:
            return 0.
#        return ( ( 1 / ( 2 * L ) ) * complex_quadrature( lambda y: ( g(y) \
#            * numpy.exp( complex( 0, -k * numpy.pi / l ) ) ), -l, l ) )
    
    @Memoize
    def chi(k, alpha, omega):
        return ( g_k(k) * ( ( lambda_0(k, omega) * mu_0 - lambda_1(k, omega) * mu_1 ) \
            / f( lambda_1(k, omega) * mu_1, k ) - ( lambda_0(k, omega) * mu_0 - alpha ) / f( alpha, k ) ) )

    @Memoize
    def eta(k, alpha, omega):
        return ( g_k(k) * ( ( lambda_0(k, omega) * mu_0 + lambda_1(k, omega) * mu_1 ) \
            / f( lambda_1(k, omega) * mu_1, k ) - ( lambda_0(k, omega) * mu_0 + alpha ) / f( alpha, k ) ) )

    @Memoize
    def e_k(k, alpha, omega):
        exp = numpy.exp( -2 * lambda_0(k, omega) * L )
        
        if k ** 2 >= ( omega ** 2 ) * ksi_0 / mu_0 :
            return ( ( A + B * ( numpy.abs(k) ** 2 ) ) * ( ( 1 / ( 2 * lambda_0(k, omega) ) ) \
                * ( ( numpy.abs(chi(k, alpha, omega)) ** 2 ) * ( 1 - exp ) \
                + ( numpy.abs(eta(k, alpha, omega)) ** 2 ) * ( exp - 1 ) ) \
                + 2 * L * numpy.real( chi(k, alpha, omega) * numpy.conj( eta(k, alpha, omega) ) ) ) \
                + B * L * numpy.abs(lambda_0(k, omega)) / 2 * ( ( chi(k, alpha, omega) ** 2 ) * ( 1 - exp ) \
                + ( numpy.abs(eta(k, alpha, omega)) ** 2 ) * ( exp - 1 ) ) \
                - 2 * B * ( lambda_0(k, omega) ** 2 ) * L * numpy.real( chi(k, alpha, omega) * numpy.conj( eta(k, alpha, omega ) ) ) )
        else:
            return ( ( A + B * ( numpy.abs(k) ** 2 ) ) * ( L \
                * ( ( numpy.abs( chi(k, alpha, omega) ) ** 2 ) + ( numpy.abs( eta(k, alpha, omega) ) ** 2 ) ) \
                + complex(0, 1 / lambda_0(k, omega) * numpy.imag( chi(k, alpha, omega) * numpy.conj( eta(k, alpha, omega) \
                * ( 1 - exp ) ) ) ) ) + B * L * ( numpy.abs( lambda_0(k, omega) ) ** 2 ) \
                * ( ( numpy.abs( chi(k, alpha, omega) ) ** 2 ) + ( numpy.abs( eta(k, alpha, omega) ) ** 2 ) ) \
                + complex(0, B * lambda_0(k, omega) * numpy.imag( chi(k, alpha, omega) * numpy.conj( eta(k, alpha, omega) \
                * ( 1 - exp ) ) ) ) )
    
    @Memoize
    def sum_e_k(omega):
        def sum_func(alpha):
            s = 0
            for n in range(-resolution, resolution+1):
                s += e_k(n*numpy.pi/L, alpha, omega)
            return s
        return sum_func
    
    @Memoize
    def alpha(omega):
        alpha_0 = numpy.array(complex(40, -40))
        return real_to_complex(minimize(lambda z: numpy.real(sum_e_k(omega)(real_to_complex(z))), complex_to_real(alpha_0), tol=1e-4).x)

    return alpha(omega)

if __name__ == '__main__':
    print('Computing alpha...')
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

    plt.savefig('alpha_Melamine.png')
