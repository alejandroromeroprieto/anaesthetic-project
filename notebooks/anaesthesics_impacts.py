# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr

# %%
from fair import FAIR
from fair.io import read_properties
from fair.interface import fill
from fair.io import read_properties
from fair.interface import initialise

# %% [markdown]
#
# 1. Initialise FaIR

# %%
f = FAIR()

# %% [markdown]
# 2. Define time horizon

# %%
f.define_time(1750, 2500, 1)

# %% [markdown]
# 3. Define scenarios

# %%
# scenarios = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp534-over', 'ssp585']
scenarios = ["ssp119", "ssp245", "ssp585"]
f.define_scenarios(scenarios)

# %% [markdown]
# 4. Define configs

# %%
# Old simple configs
# configs = ["high", "central", "low"]
# f.define_configs(configs)

fair_params_1_4_0 = '../data/fair-calibration/calibrated_constrained_parameters_1.4.0_plus_anaesthesics.csv'
df_configs = pd.read_csv(fair_params_1_4_0, index_col=0)[400:]
configs = df_configs.index  # label for the "config" axis
f.define_configs(configs)

# %% [markdown]
# 5. Define species and properties

# %%
species, properties = read_properties("/home/eearp/code/helping_tom/FAIR/src/fair/anesthesics/species_configs_properties_1.4.0_anesthesics.csv")
f.define_species(species, properties)

# %% [markdown]
# 6. Modify run options (optional)

# %% [markdown]
# 7. Initialise arrays

# %%
f.allocate()

# %% [markdown]
# 8. Fill in data

# %%
f.fill_from_csv(
    forcing_file='../data/forcing/volcanic_solar.csv',
)

# %%
# I was lazy and didn't convert emissions to CSV, so use the old clunky method of importing from netCDF
# this is from calibration-1.4.0
da_emissions = xr.load_dataarray("../data/emissions/ssp_emissions_1750-2500_anesthesics.nc")
output_ensemble_size = len(configs)
da = da_emissions.loc[dict(config="unspecified", scenario=scenarios)]
fe = da.expand_dims(dim=["config"], axis=(2))
f.emissions = fe.drop_vars(("config")) * np.ones((1, 1, output_ensemble_size, 1))

# %%
# Fill anesthesics
anesthesics_df = pd.read_csv("../data/emissions/anesthesics_gas_emissions.csv")

# %%
# Prepare the new emissions
species1_df = anesthesics_df[
    (anesthesics_df["Variable"] == "HFE-236ea2") &
    (anesthesics_df["Scenario"].isin(scenarios))
]
emissions_1 = species1_df.loc[:, '1751':'2500'].astype(float).values.T
emissions_1 = emissions_1[:, :, np.newaxis]
emissions_1 = np.repeat(emissions_1, len(configs), axis=2)

# %%
species2_df = anesthesics_df[
    (anesthesics_df["Variable"] == "HFE-347mmz1") &
    (anesthesics_df["Scenario"].isin(scenarios))
]
emissions_2 = species2_df.loc[:, '1751':'2500'].astype(float).values.T
emissions_2 = emissions_2[:, :, np.newaxis]
emissions_2 = np.repeat(emissions_2, len(configs), axis=2)

# %%
species3_df = anesthesics_df[
    (anesthesics_df["Variable"] == "HCFE-235da2") &
    (anesthesics_df["Scenario"].isin(scenarios))
]
emissions_3 = species3_df.loc[:, '1751':'2500'].astype(float).values.T
emissions_3 = emissions_3[:, :, np.newaxis]
emissions_3 = np.repeat(emissions_3, len(configs), axis=2)


# %%
fill(f.emissions, emissions_1, specie="HFE-236ea2")
fill(f.emissions, emissions_2, specie="HFE-347mmz1")
fill(f.emissions, emissions_3, specie="HCFE-235da2")
fill(f.emissions, 0, specie="Halon-2311")

# %%

# %% [markdown]
# 8a. Fill in data - species configs

# %%
f.fill_species_configs("../data/fair-calibration/species_configs_properties_1.4.0_anesthesics.csv")

# %% [markdown]
# 8b. Fill in data - emissions

# %%
# initialising 
initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

# %% [markdown]
# 8c. Fill in data - climate configs

# %%

f.override_defaults('../data/fair-calibration/calibrated_constrained_parameters_1.4.0_plus_anaesthesics.csv')

# %% [markdown]
# 9 Run FaIR

# %%
f.run()

# %% [markdown]
# 10 Pretty plots!

# %%
# Run twin experiment, but without anesthesics
f_no_anesthesics = FAIR()
f_no_anesthesics.define_time(1750, 2500, 1)
f_no_anesthesics.define_scenarios(scenarios)
f_no_anesthesics.define_configs(configs)
species, properties = read_properties("../data/fair-calibration/species_configs_properties_1.4.0_original.csv")
f_no_anesthesics.define_species(species, properties)
f_no_anesthesics.allocate()
f_no_anesthesics.fill_from_csv(
    forcing_file='../data/forcing/volcanic_solar.csv',
)
da_emissions = xr.load_dataarray("../data/emissions/ssp_emissions_1750-2500.nc")
output_ensemble_size = len(configs)
da = da_emissions.loc[dict(config="unspecified", scenario=scenarios)]
fe = da.expand_dims(dim=["config"], axis=(2))
f_no_anesthesics.emissions = fe.drop_vars(("config")) * np.ones((1, 1, output_ensemble_size, 1))
f_no_anesthesics.fill_species_configs("../data/fair-calibration/species_configs_properties_1.4.0_anesthesics.csv")
initialise(f_no_anesthesics.concentration, f_no_anesthesics.species_configs["baseline_concentration"])
initialise(f_no_anesthesics.forcing, 0)
initialise(f_no_anesthesics.temperature, 0)
initialise(f_no_anesthesics.cumulative_emissions, 0)
initialise(f_no_anesthesics.airborne_emissions, 0)
f_no_anesthesics.override_defaults('../data/fair-calibration/calibrated_constrained_parameters_1.4.0.csv')

# %%
f_no_anesthesics.run()

# %%
# Plot comparison between anesthesics and no anesthesics
scenario_to_compare = "ssp245"
temperature_anomalies = f.temperature.loc[dict(scenario=scenario_to_compare, layer=0)] - f_no_anesthesics.temperature.loc[dict(scenario=scenario_to_compare, layer=0)]

# %%
plt.plot(f.timebounds[150:], temperature_anomalies[150:])
plt.title('Central scenario: temperature')
plt.xlabel('year')
plt.ylabel('Temperature anomaly (K)')
plt.show()

# %% [markdown]
# Same plot with standard deviation, rather than simply all the projections printed to screen

# %%
# Slice the relevant part of the data
time = f.timebounds[150:]
temp = temperature_anomalies[150:]

# Compute mean and standard deviation over 'config'
mean_temp = temp.mean(dim='config')
std_temp = temp.std(dim='config')

# Convert to numpy for plotting
time_np = time
mean_np = mean_temp.values
std_np = std_temp.values

# Plot the mean with a shaded area for ±1 std dev
plt.plot(time_np, mean_np, label='Mean')
plt.fill_between(time_np, mean_np - std_np, mean_np + std_np, alpha=0.3, label='±1 std dev')

plt.title('Central scenario: temperature')
plt.xlabel('Year')
plt.ylabel('Temperature anomaly (K)')
plt.legend()
plt.show()

# %% [markdown]
# 5-95th percentiles

# %%
p05 = temp.quantile(0.05, dim='config')
p95 = temp.quantile(0.95, dim='config')

# Plot the mean with a shaded area for 5-95 percentile
plt.plot(time_np, mean_np, label='Mean')
plt.fill_between(time_np, p05, p95, alpha=0.3, label='5-95% percentile')

plt.title('Central scenario: temperature')
plt.xlabel('Year')
plt.ylabel('Temperature anomaly (K)')
plt.legend()
plt.show()
