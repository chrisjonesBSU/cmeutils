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
    try:
        data = log_file[varialbe]
    except ValueError:
        print(f"{variable} is not a valid header in {filename}.")

    uncorr_sample, uncorr_indices, prod_start, ineff, Neff = equil_sample(
            data=data,
            threshold_fraction=threshold_fraction,
            threshold_neff=threshold_neff,
            conservative=conservative
    )


def is_equilibrated(
        job,
        filename,
        variable,
        threshold_fraction,
        threshold_neff,
        strict
):
    log_file = np.genfromtxt(job.fn(filename), names=True)
    try:
        data = log_file[varialbe]
    except ValueError:
        print(f"{variable} is not a valid header in {filename}.")

    equilibrated, t0, g, Neff = is_equilibrated(
            data=data,
            threshold_fraction=threshold_fraction,
            threshold_neff=threshold_neff,
            strict=strict
    )
    return equilibrated
