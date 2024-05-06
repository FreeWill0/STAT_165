import numpy as np
import scipy.stats as stats

years = np.array([1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016])
gold_medals = np.array([24, 6, 18, 19, 39, 52, 74, 56, 46])

log_gold_medals = np.log(gold_medals)

mu = np.mean(log_gold_medals)
sigma = np.std(log_gold_medals, ddof=1) 

z = stats.norm.ppf(0.9) 
margin_error = z * sigma / np.sqrt(len(gold_medals))

ci_lower = np.exp(mu - margin_error)
ci_upper = np.exp(mu + margin_error)

ci_lower, ci_upper
