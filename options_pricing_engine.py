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

#inputs
s = float(input("Enter current asset price"))
k = float(input("enter strike price of the option"))
r = float(input("enter riskfree rate of the option"))
t = float(input("enter time to maturity of the option"))
sigma = float(input("enter annualised volatility of the assets return"))


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
#delta - symbolic finite differences
# using autograd
delta_c = grad(call_bsm, 0)
delta_p = grad(put_bsm, 0)
#gamma - second order diff
gamma_c = grad(delta_c)
gamma_p = grad(delta_p)
#vega - derivative wrt sigma (volatility)
vega_c = grad(call_bsm,4)
vega_p = grad(put_bsm,4)
#theta - derivative wrt t (time)
theta_c = grad(call_bsm,3)
theta_p = grad(put_bsm,3)
#rho - derivative wrt r (rfr)
rho_c = grad(call_bsm,2)
rho_p = grad(put_bsm,2)

print("Call data")
print("this is delta" , delta_c(s, k, r, t, sigma))
print("this is gamma" , gamma_c(s, k, r, t, sigma))
print("this is vega" , vega_c(s, k, r, t, sigma))
print("this is theta" , theta_c(s, k, r, t, sigma))
print("this is rho" , rho_c(s, k, r, t, sigma))
print("")
print("Put data")
print("this is delta" , delta_p(s, k, r, t, sigma))
print("this is gamma" , gamma_p(s, k, r, t, sigma))
print("this is vega" , vega_p(s, k, r, t, sigma))
print("this is theta" , theta_p(s, k, r, t, sigma))
print("this is rho" , rho_p(s, k, r, t, sigma))

# option volatility heat map (volatility against spot prices) - maintaining constant strike price of 100

#fixed values
r1 = r
t1 = t
k1 = k
#varying values
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


plt.imshow(heat_array_c, cmap="viridis", extent=[vol_list[0], vol_list[-1], s_list[0], s_list[-1]], aspect='auto', origin='lower')
plt.colorbar(label='Option Price')
plt.xlabel("Volatility sigma")
plt.ylabel("Spot Price s")
plt.title("call option price heatmap")
plt.show()

plt.imshow(heat_array_p, cmap="viridis", extent=[vol_list[0], vol_list[-1], s_list[0], s_list[-1]], aspect='auto', origin='lower')
plt.colorbar(label='Option Price')
plt.xlabel("Volatility sigma")
plt.ylabel("Spot Price s")
plt.title("Put option price heatmap")
plt.show()
