import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def black76_swaption_price(phi, p_market, T, A_fixed, K, sigma):
    d1 = (math.log(p_market / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    price = phi * A_fixed * (p_market * norm.cdf(phi * d1) - K * norm.cdf(phi * d2))
    return price


# Parameter ranges
sigma_range = np.linspace(0.01, 0.5, 100)
A_fixed_range = np.linspace(500000, 2000000, 100)
p_market_range = np.linspace(1.5, 3.5, 100)
K_range = np.linspace(2.0, 3.0, 100)
sigma = 10
A_fixed = 700000
p_market = 5
K = 5
phi = 1  # Payer swaption
T = 1  # Time to expiry

# Swaption prices for each parameter range
prices_sigma = [black76_swaption_price(phi, p_market, T, A_fixed, K, sigma) for sigma, p_market in zip(sigma_range, p_market_range)]
prices_A_fixed = [black76_swaption_price(phi, p_market, T, A_fixed, K, sigma) for A_fixed in A_fixed_range]
prices_p_market = [black76_swaption_price(phi, p_market, T, A_fixed, K, sigma) for p_market in p_market_range]
prices_K = [black76_swaption_price(phi, p_market, T, A_fixed, K, sigma) for K in K_range]

# Subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Black76 Swaption Prices vs. Parameters', fontsize=16)

# Prices vs sigma
axs[0, 0].plot(sigma_range, prices_sigma)
axs[0, 0].tick_params(axis='both', which='both', labelleft=False, labelbottom=False)
axs[0, 0].set_xlabel('$\sigma$')
axs[0, 0].set_ylabel('Price')
axs[0, 0].set_title('Price vs. $\sigma$')

# Prices vs A_fixed
axs[0, 1].plot(A_fixed_range, prices_A_fixed)
axs[0, 1].tick_params(axis='both', which='both', labelleft=False, labelbottom=False)
axs[0, 1].set_xlabel('$A^{Fixed}$')
axs[0, 1].set_ylabel('Price')
axs[0, 1].set_title('Price vs. $A^{Fixed}$')

# Prices vs p_market
axs[1, 0].plot(p_market_range, prices_p_market)
axs[1, 0].tick_params(axis='both', which='both', labelleft=False, labelbottom=False)
axs[1, 0].set_xlabel('$p^{Market}$')
axs[1, 0].set_ylabel('Price')
axs[1, 0].set_title('Price vs. $p^{Market}$')

# Prices vs K
axs[1, 1].plot(K_range, prices_K)
axs[1, 1].tick_params(axis='both', which='both', labelleft=False, labelbottom=False)
axs[1, 1].set_xlabel('K')
axs[1, 1].set_ylabel('Price')
axs[1, 1].set_title('Price vs. K')

# Plot
plt.tight_layout()
plt.show()
