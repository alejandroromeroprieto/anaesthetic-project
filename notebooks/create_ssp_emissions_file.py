import xarray as xr


# # Creating ssp emissions file with anesthesics
da_emissions = xr.load_dataarray("../data/emissions/ssp_emissions_1750-2500.nc")

sf6_data = da_emissions.sel(specie="SF6")
new_gas_1 = sf6_data.expand_dims(dim="specie").assign_coords(specie=["HFE-236ea2"])
new_gas_2 = sf6_data.expand_dims(dim="specie").assign_coords(specie=["HFE-347mmz1"])
new_gas_3 = sf6_data.expand_dims(dim="specie").assign_coords(specie=["HCFE-235da2"])
new_gas_4 = sf6_data.expand_dims(dim="specie").assign_coords(specie=["Halon-2311"])

da_emissions = xr.concat([da_emissions, new_gas_1, new_gas_2, new_gas_3, new_gas_4], dim="specie")

da_emissions.to_netcdf("../data/emissions/ssp_emissions_1750-2500_anesthesics_new.nc")