import numpy as np

iirf_horizon = 100

partition_fraction = np.array([1, 0, 0, 0])
unperturbed_lifetime = np.array([1]*4)


g1 = np.sum(partition_fraction*unperturbed_lifetime*(1 - (1 + iirf_horizon/unperturbed_lifetime) * np.exp(-iirf_horizon / unperturbed_lifetime)))

g0 = np.exp(-1 * np.sum(partition_fraction * unperturbed_lifetime * (1 - np.exp(-iirf_horizon / unperturbed_lifetime))) / g1)


print(f"g0 = {round(g0, 9)}, and g1 = {round(g1, 9)}")