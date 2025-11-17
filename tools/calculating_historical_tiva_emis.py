# number of interventions
TIVA_INTERVENTIONS = 40000000               # From Laurentiu
KG_PLASTIC_PER_TIVA_INTERVENTION = 0.5      # From Laurentiu?
CO2_EMISSIONS_PER_KG_PLASTIC_BURN = 3       # kg of CO2 FROM https://acp.copernicus.org/articles/23/14561/2023/acp-23-14561-2023.pdf

print(TIVA_INTERVENTIONS*KG_PLASTIC_PER_TIVA_INTERVENTION*CO2_EMISSIONS_PER_KG_PLASTIC_BURN / 1e12)
print(6e-5)