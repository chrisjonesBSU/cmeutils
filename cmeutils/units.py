def base_units():
    units = {}
    units["avogadro"] = 6.022140857e23
    units["boltzmann"] = 1.38064852e-23
    units["kj_to_j"] = 1e3 
    units["kcal_to_j"] = 4184
    units["amu_to_kg"] = 1.6605e-27
    units["amu_to_g"] = 1.6605e-24
    units["cm_to_nm"] = 1e7 
    units["nm_to_m"] = 1e-9
    units["ang_to_m"] = 1e-10
    units["m_to_nm"] = 1e9 
    units["m_to_ang"] = 1e10
    units["mass_units"] = "amu"
    units["energy_units"] = "kcal/mol"
    units["distance_units"] = "angstroms"
    return units

def reduce_from_kelvin(T_SI, ref_energy, precision=2):
    units = base_units.base_units()
    T = (units["avogadro"] * units["boltzmann"] * T_SI) / ( 
        units["kcal_to_j"] * ref_energy
    )   
    T = round(T, precision)
    return T


def kelvin_from_reduced(T_reduced, ref_energy, precision=0):
    units = base_units.base_units()
    T_SI = (T_reduced * ref_energy * units["kcal_to_j"]) / ( 
        units["avogadro"] * units["boltzmann"]
    )   
    T_SI = round(T_SI, precision)
    return T_SI


def convert_to_real_time(dt, ref_energy, ref_distance, ref_mass, precision=3):
    units = base_units.base_units()
    time_tau = ref_mass * units["amu_to_kg"]
    time_tau *= (ref_distance * units["ang_to_m"]) ** 2
    time_tau /= ref_energy * units["kcal_to_j"] / units["avogadro"]
    real_time = dt * (time_tau ** 0.5) * 1e15
    real_time = round(real_time, precision)
    return real_time
