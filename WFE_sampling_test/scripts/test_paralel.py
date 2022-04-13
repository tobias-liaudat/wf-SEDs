import numpy as np
import matplotlib.pyplot as plt
import wf_psf
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count, parallel_backend


# Total number of datasets
n_procs = 150
n_cpus = 8


# Print some info
cpu_info = ' - Number of available CPUs: {}'.format(cpu_count())
proc_info = ' - Total number of processes: {}'.format(n_procs)


print(cpu_info)
print(proc_info)


# Generate catalog list
star_id_list = [0 + i for i in range(n_procs)]


def generate_star(star_id):
    print('\nProcessing star: ', star_id)


with parallel_backend("loky", inner_max_num_threads=1):
    results = Parallel(n_jobs=n_cpus)(delayed(generate_star)(_star_id)
                                        for _star_id in star_id_list)