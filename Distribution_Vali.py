import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

gold_medals = np.array([24, 6, 18, 19, 39, 52, 74, 56, 46])

shape, loc, scale = stats.lognorm.fit(gold_medals, floc=0)
estimated_mu = np.log(scale)
estimated_sigma = shape


x = np.linspace(min(gold_medals), max(gold_medals), 100)
pdf_fitted = stats.lognorm.pdf(x, shape, loc=0, scale=scale)


plt.figure(figsize=(12, 6))
plt.hist(gold_medals, bins=range(0, 80, 5), density=True, alpha=0.7, color='blue', edgecolor='black')
plt.plot(x, pdf_fitted, 'r-', label=f'Log-Normal Fit: shape={shape:.2f}, scale={scale:.2f}')
plt.title('Fit of Log-Normal Distribution to Gold Medals Data')
plt.xlabel('Number of Gold Medals')
plt.ylabel('Density')
plt.legend()
plt.show()


ks_stat, p_value = stats.kstest(gold_medals, 'lognorm', args=(shape, 0, scale))
ks_stat, p_value
