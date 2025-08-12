# Calculate the fractional release of a species given its ozone depleting power (ODP)

# Data for target species
odp = 1.56
total_lifetime = 6.6 # 1
molecule_mass = 197.4 # g/mol
number_br_atoms = 1
number_cl_atoms = 1
number_io_atoms = 0

# CFC-11 numbers
fractional_release_CFC_11 = 0.47
total_lifetime_CFC_11 = 52 # years
molecule_mass_CFC_11 = 137.36

rel_factor_br = 60
rel_factor_io = 240

new_fractional_release = 3 * (fractional_release_CFC_11 / (number_cl_atoms + number_br_atoms*rel_factor_br + number_io_atoms*rel_factor_io)) * odp * (total_lifetime_CFC_11/total_lifetime) * (molecule_mass/molecule_mass_CFC_11)


print(f"The fractional release for the supplied species is: {new_fractional_release}")