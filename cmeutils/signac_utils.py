import numpy as np
from cmeutils.sampling import equil_sample, is_equilibrated


def sample_job(
        job,
        filename,
        variable,
        threshold_fraction,
        threshold_neff,
        conservative=True
):
    log_file = np.genfromtxt(job.fn(filename), names=True)
    col_name = _parse_hoomd_log_names(variable, log_file.dtype.names)
    data = log_file[col_name]

    uncorr_sample, uncorr_indices, prod_start, ineff, Neff = equil_sample(
            data=data,
            threshold_fraction=threshold_fraction,
            threshold_neff=threshold_neff,
            conservative=conservative
    )

    try:
        job.doc.samples
    except AttributeError:
        job.doc.samples = {}

    variable_dict = {
            "start": uncorr_indices.start + prod_start,
            "stop": uncorr_indices.stop + prod_start,
            "step": uncorr_indices.step,
            "N_samples": np.round(Neff, 0)
    }
    job.doc.samples[variable] = variable_dict


def get_sample(job, filename, variable):
    log_file = np.genfromtxt(job.fn(filename), names=True)
    col_name = _parse_hoomd_log_names(variable, log_file.dtype.names)
    data = log_file[col_name]

    indices = range(
            job.doc.samples[variable]["start"],
            job.doc.samples[variable]["stop"], 
            job.doc.samples[variable]["step"], 
    )
    return data[indices]


def check_equilibration(
        job,
        filename,
        variable,
        threshold_fraction,
        threshold_neff,
):
    log_file = np.genfromtxt(job.fn(filename), names=True)
    col_name = _parse_hoomd_log_names(variable, log_file.dtype.names)
    data = log_file[col_name]

    equilibrated, t0, g, Neff = is_equilibrated(
            data=data,
            threshold_fraction=threshold_fraction,
            threshold_neff=threshold_neff,
    )
    return equilibrated


def _parse_hoomd_log_names(variable, names):
    name_match = [i for i in names if variable in i]
    if len(name_match) == 0:
        raise ValueError(
               f"{variable} did not return any matches "
               "in the log file column names." 
               f"The possible column names are {names}."
        )
    if len(name_match) > 1:
        raise ValueError(
                f"The variable {variable} returned multiple column headers:"
                f"{name_match}."
        )
    return name_match[0]
