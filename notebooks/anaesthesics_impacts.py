# %%
import matplotlib.pyplot as pl
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
# Look for len(configs)
configs = ["high", "central", "low"]
f.define_configs(configs)

# %% [markdown]
# 5. Define species and properties

# %%
species, properties = read_properties("../data/fair-calibration/species_configs_properties_1.4.0_anesthesics.csv")
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

f.override_defaults('../data/fair-calibration/configs_ensemble_simple.csv')

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
f_no_anesthesics.override_defaults('../data/fair-calibration/configs_ensemble_simple.csv')
f_no_anesthesics.run()

# %%
# Plot comparison between anesthesics and no anesthesics
scenario_to_compare = "ssp245"
temperature_anomalies = f.temperature.loc[dict(scenario=scenario_to_compare, layer=0)] - f_no_anesthesics.temperature.loc[dict(scenario=scenario_to_compare, layer=0)]

# %%
pl.plot(f.timebounds[150:], temperature_anomalies[150:], label=f.configs)
pl.title('Central scenario: temperature')
pl.xlabel('year')
pl.ylabel('Temperature anomaly (K)')
pl.legend()
pl.show()

# %%
