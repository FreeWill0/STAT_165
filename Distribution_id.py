
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


years = np.array([1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020])
gold_medals = np.array([24, 6, 18, 19, 39, 52, 74, 56, 46])


plt.figure(figsize=(12, 6))
plt.hist(gold_medals, bins=range(0, 55, 5), alpha=0.7, color='blue', density=True, edgecolor='black')
plt.title('Distribution of Gold Medals Won by China')
plt.xlabel('Number of Gold Medals')
plt.ylabel('Probability')


mu, std = stats.norm.fit(gold_medals)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label=f"Norm: $\mu$={mu:.2f}, $\sigma$={std:.2f}")


shape, loc, scale = stats.lognorm.fit(gold_medals, floc=0)
lognorm_pdf = stats.lognorm.pdf(x, shape, loc, scale)
plt.plot(x, lognorm_pdf, 'r', linewidth=2, label=f"Log-Norm: s={shape:.2f}, scale={scale:.2f}")


rate = np.mean(gold_medals)
poisson_pmf = stats.poisson.pmf(np.round(x), rate)
plt.plot(x, poisson_pmf, 'g', linewidth=2, label=f"Poisson: $\lambda$={rate:.2f}")

plt.legend()
plt.grid(True)
plt.show()


