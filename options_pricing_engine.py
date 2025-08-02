# bsm model for european options
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from scipy.stats import norm
from autograd import grad
import autograd.numpy as anp
from autograd.scipy.stats import norm as anorm
N_prime = anorm.pdf
N = anorm.cdf

#call calculation
def call_bsm(s, k, r, t, sigma):
    d1 = (anp.log(s / k) + (r + (anp.square(sigma) / 2)) * t) / (sigma * (anp.sqrt(t)))
    d2 = d1 - (sigma * anp.sqrt(t))
    return s*N(d1)-N(d2)*k*anp.exp(-r*t)

def put_bsm(s, k, r, t, sigma):
    d1 = (anp.log(s / k) + (r + (anp.square(sigma) / 2)) * t) / (sigma * (anp.sqrt(t)))
    d2 = d1 - (sigma * anp.sqrt(t))
    return N(-d2)*k*anp.exp(-r*t) - N(-d1)*s

#greeks
# using autograd
def get_greeks(call_bsm, put_bsm, s, k, r, t, sigma):
    delta_c = grad(call_bsm, 0)
    delta_p = grad(put_bsm, 0)
    gamma_c = grad(delta_c)
    gamma_p = grad(delta_p)
    vega_c = grad(call_bsm,4)
    vega_p = grad(put_bsm,4)
    theta_c = grad(call_bsm,3)
    theta_p = grad(put_bsm,3)
    rho_c = grad(call_bsm,2)
    rho_p = grad(put_bsm,2)
    greeks = {
        'delta_c' : delta_c,
        'delta_p' : delta_p,
        'gamma_c' : gamma_c,
        'gamma_p' : gamma_p,
        'vega_c' : vega_c,
        'vega_p' : vega_p,
        'theta_c' : theta_c,
        'theta_p' : theta_p,
        'rho_c' : rho_c,
        'rho_p' : rho_p,
    }
    return greeks

# option volatility heat map (volatility against spot prices) - maintaining constant strike price of 100
#fixed values
r1 = r
t1 = t
k1 = k
#varying values

def vol_heat_map(call_bsm, put_bsm, s, k, r, t, sigma):
    s_list = np.linspace(70, 130, 10)
    vol_list = np.linspace(0.1, 0.9, 10)

    # heatmap variable = option price ()
    # the logic is to now create a matrix calculating the row spot values and column sigma to get resultant price
    #using nested loop and create matrix
    heat_array_c = np.zeros((len(s_list), len(vol_list)))

    for i, s in enumerate(s_list):
        for j, sigma in enumerate(vol_list):
            price = call_bsm(s, k1, r1, t1, sigma)
            heat_array_c[i][j] = price

    heat_array_p = np.zeros((len(s_list), len(vol_list)))

    for i, s in enumerate(s_list):
        for j, sigma in enumerate(vol_list):
            price = put_bsm(s, k1, r1, t1, sigma)
            heat_array_p[i][j] = price

    return heat_array_c, heat_array_p

plt.imshow(heat_array_c, cmap="viridis", extent=[vol_list[0], vol_list[-1], s_list[0], s_list[-1]], aspect='auto', origin='lower')
plt.colorbar(label='Option Price')
plt.xlabel("Volatility sigma")
plt.ylabel("Spot Price s")
plt.title("call option price heatmap")


plt.imshow(heat_array_p, cmap="viridis", extent=[vol_list[0], vol_list[-1], s_list[0], s_list[-1]], aspect='auto', origin='lower')
plt.colorbar(label='Option Price')
plt.xlabel("Volatility sigma")
plt.ylabel("Spot Price s")
plt.title("Put option price heatmap")

