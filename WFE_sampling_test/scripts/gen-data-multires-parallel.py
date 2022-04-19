import numpy as np
import matplotlib.pyplot as plt
import wf_psf as wf_psf
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count, parallel_backend

import sys
import time

# Paths

# SED folder path
#SED_path = '/home/ecentofanti/wf-psf/data/SEDs/save_SEDs/'                 # Candide
#SED_path = './../../../wf-psf/data/SEDs/save_SEDs/'                        # Local
SED_path = '/feynman/work/dap/lcs/ec270266/wf-psf/data/SEDs/save_SEDs/'     # Feynman

# Output saving path (in node05 of candide)
#output_folder = '/n05data/ecentofanti/WFE_sampling_test/multires_dataset/' # Candide
#output_folder = './../../../output/'                                       # Local
output_folder = '/feynman/work/dap/lcs/ec270266/output/'                    # Feynman

# Temporary local storage for large data processing
tmp_folder = '/tmp/tmp.WISiX0KNNG'

# Save output prints to logfile
old_stdout = sys.stdout
log_file = open(output_folder + 'output.log','w')
sys.stdout = log_file
print('Starting the log file.')

# Dataset ID
dataset_id = 2
dataset_id_str = '%03d'%(dataset_id)

# This list must be in order from bigger to smaller
n_star_list = [2000, 1000, 500, 200]
n_test_stars = 400  # 20% of the max test stars
# Total stars
n_stars = n_star_list[0] + n_test_stars
# Max train stars
tot_train_stars = n_star_list[0]

# Parameters
d_max = 2
max_order = 45
x_lims = [0, 1e3]
y_lims = [0, 1e3]
grid_points = [4, 4]
n_bins = 20
auto_init = False
verbose = True

oversampling_rate = 3.
output_Q = 3.

max_wfe_rms = 0.1
output_dim = 32
LP_filter_length = 2
euclid_obsc = True

# Desired WFE resolutions
#WFE_resolutions = [256, 1024, 4096]
WFE_resolutions = [4096, 256]

print('\nInit dataset generation')

zernikes_multires = []
sim_PSF_toolkit_multires = []
gen_poly_fieldPSF_multires = []

for i, pupil_diameter_ in tqdm(enumerate(WFE_resolutions)):

    # Generate Zernike maps in max resolution
    zernikes_multires.append( wf_psf.utils.zernike_generator(n_zernikes=max_order, wfe_dim=pupil_diameter_) )
    
    # Initialize PSF simulator for each star (no euclid obscurations and wfr_rms init)
    packed_PSFToolkit = [ wf_psf.SimPSFToolkit(
        zernikes_multires[i], max_order=max_order, max_wfe_rms=max_wfe_rms, oversampling_rate=oversampling_rate,
        output_Q=output_Q, output_dim=output_dim, pupil_diameter=pupil_diameter_, euclid_obsc=False,
        LP_filter_length=LP_filter_length)
        for j in range(n_stars)
    ]
    sim_PSF_toolkit_multires.append( packed_PSFToolkit )

    # Initialize one PSF field for each resolution
    packed_polyField_PSF = [ wf_psf.GenPolyFieldPSF(sim_PSF_toolkit_multires[i][j], d_max=d_max,
        grid_points=grid_points, max_order=max_order,
        x_lims=x_lims, y_lims=y_lims, n_bins=n_bins,
        lim_max_wfe_rms=max_wfe_rms, auto_init=auto_init, verbose=verbose)
        for j in range(n_stars)
    ]
    gen_poly_fieldPSF_multires.append( packed_polyField_PSF)
    

# Dummy SimPSFToolkit to init obscurations
init_toolkit = []
init_polyField = []
for i, pupil_diameter_ in enumerate(WFE_resolutions):
    init_toolkit.append( wf_psf.SimPSFToolkit(
        zernikes_multires[i], max_order=max_order, max_wfe_rms=max_wfe_rms, oversampling_rate=oversampling_rate,
        output_Q=output_Q, output_dim=output_dim, pupil_diameter=pupil_diameter_, euclid_obsc=euclid_obsc,
        LP_filter_length=LP_filter_length) )
    init_polyField.append( wf_psf.GenPolyFieldPSF(init_toolkit[i], d_max=d_max,
        grid_points=grid_points, max_order=max_order,
        x_lims=x_lims, y_lims=y_lims, n_bins=n_bins,
        lim_max_wfe_rms=max_wfe_rms, auto_init=True, verbose=verbose))

# # Dummy PolyFieldPSF to init C_poly coefficients
# init_polyField = wf_psf.GenPolyFieldPSF(sim_PSF_toolkit_multires[0][0], d_max=d_max,
#         grid_points=grid_points, max_order=max_order,
#         x_lims=x_lims, y_lims=y_lims, n_bins=n_bins,
#         lim_max_wfe_rms=max_wfe_rms, auto_init=True, verbose=verbose)

# Share C_poly coefficients throughout all the different resolution models
for i in range(len(gen_poly_fieldPSF_multires)):
    for j in range(n_stars):
        gen_poly_fieldPSF_multires[i][j].set_C_poly(init_polyField[i].C_poly)
        gen_poly_fieldPSF_multires[i][j].set_WFE_RMS(init_polyField[i].WFE_RMS)
        gen_poly_fieldPSF_multires[i][j].sim_psf_toolkit.obscurations = init_toolkit[i].obscurations

# Load the SEDs
stellar_SEDs = np.load(SED_path + 'SEDs.npy', allow_pickle=True)
stellar_lambdas = np.load(SED_path + 'lambdas.npy', allow_pickle=True)

# Generate all the stars and then go saving different subsets
# Select random SEDs
SED_list = []
for it in range(n_stars):
    selected_id_SED = np.random.randint(low=0, high=13)
    concat_SED_wv = np.concatenate((stellar_lambdas.reshape(-1,1),
                                    stellar_SEDs[selected_id_SED,:].reshape(-1,1)), axis=1)
    SED_list.append(concat_SED_wv)

# First we choose the locations (randomly)
pos_np = np.random.rand(n_stars, 2)

pos_np[:,0] = pos_np[:,0]*(x_lims[1] - x_lims[0]) + x_lims[0]
pos_np[:,1] = pos_np[:,1]*(y_lims[1] - y_lims[0]) + y_lims[0]    
    
# # Plot star positions
# plt.scatter(pos_np[:,0],pos_np[:,1])
# plt.title('Samples - Star positions')
print('\nStar positions selected')

#######################################
#            PARALELLIZED             #
#######################################

# Total number of stars
n_procs = n_stars*len(WFE_resolutions)
n_cpus = 8

# Print some info
cpu_info = ' - Number of available CPUs: {}'.format(cpu_count())
proc_info = ' - Total number of processes: {}'.format(n_procs)
print(cpu_info)
print(proc_info)

# Generate star list
star_id_list = [id_ for id_ in range(n_procs)]

# ndArray for PSFs and zernikes
poly_psf_multires = np.zeros((len(WFE_resolutions),n_stars),dtype=object)
zernike_coef_multires = np.zeros((len(WFE_resolutions),n_stars),dtype=object)

# Function to get (i,j) from id
def unwrap_id(id, n_stars):
    i = int(id/n_stars)
    j = int(id - i * n_stars)
    return i, j

def print_status(star_id, i, j):
    print('\nStar ' +str(star_id)+ ' done!' + '   index=('+str(i)+','+str(j)+')')

# Function to get one PSF
def simulate_star(star_id, gen_poly_fieldPSF_multires):
    i,j = unwrap_id(star_id, n_stars)
    _psf, _zernike, _ = gen_poly_fieldPSF_multires[i][j].get_poly_PSF(xv_flat=pos_np[j, 0],
                                                           yv_flat=pos_np[j, 1],
                                                           SED=SED_list[j])
    print_status(star_id, i, j)
    return (star_id, _psf, _zernike)

# Measure time
start_time = time.time()
with parallel_backend("loky", inner_max_num_threads=1,):
    results = Parallel(n_jobs=n_cpus)(delayed(simulate_star)(_star_id, gen_poly_fieldPSF_multires)
                                        for _star_id in star_id_list)
end_time = time.time()
print('\nAll stars generated in '+ str(end_time-start_time) +' seconds')

#######################################
#            END PARALLEL             #
#######################################

poly_psf_list = np.array( [star[1] for star in results] )
zernike_coef_list = np.array( [star[2] for star in results] )

zernike_coef_multires = []
poly_psf_multires = []

for i in range(len(WFE_resolutions)):
    poly_psf_multires.append(poly_psf_list[i*n_stars:(i+1)*n_stars])
    zernike_coef_multires.append(zernike_coef_list[i*n_stars:(i+1)*n_stars])

# star_to_show = 0

# plt.figure(figsize=(30,10))
# plt.suptitle('Pixel PSFs with different WFE resolution', fontsize=35)
# plt.subplot(131)
# plt.imshow(poly_psf_multires[0][star_to_show,:,:], cmap='gist_stern');plt.colorbar()
# plt.title('WFE dim: 256', fontsize=24)
# plt.subplot(132)
# plt.imshow(poly_psf_multires[1][star_to_show,:,:], cmap='gist_stern');plt.colorbar()
# plt.title('WFE dim: 1024', fontsize=24)
# plt.subplot(133)
# plt.imshow(poly_psf_multires[2][star_to_show,:,:], cmap='gist_stern');plt.colorbar()
# plt.title('WFE dim: 4096', fontsize=24)
# plt.show()

# plt.figure(figsize=(30,10))
# plt.suptitle('Pixelwise differences between PSFs with different WFE resolution', fontsize=35)
# plt.subplot(131)
# plt.imshow(np.abs( poly_psf_multires[0][star_to_show,:,:]-poly_psf_multires[1][star_to_show,:,:] ), cmap='gist_stern');plt.colorbar()
# plt.title('256 vs 1024', fontsize=24)
# plt.subplot(132)
# plt.imshow(np.abs( poly_psf_multires[0][star_to_show,:,:]-poly_psf_multires[2][star_to_show,:,:] ), cmap='gist_stern');plt.colorbar()
# plt.title('256 vs 4096', fontsize=24)
# plt.subplot(133)
# plt.imshow(np.abs( poly_psf_multires[2][star_to_show,:,:]-poly_psf_multires[1][star_to_show,:,:] ), cmap='gist_stern');plt.colorbar()
# plt.title('1024 vs 4096', fontsize=24)
# plt.show()


WFE_res_id = 0

# SNR varying randomly from 10 to 120 - shared over all WFE resolutions
rand_SNR = (np.random.rand(tot_train_stars) * 100) + 10
# Copy the training stars
train_stars = np.copy(np.array(poly_psf_multires[0])[:tot_train_stars, :, :])
# Add Gaussian noise to the observations
noisy_train_stars = np.stack([wf_psf.utils.add_noise(_im, desired_SNR=_SNR) 
                              for _im, _SNR in zip(train_stars, rand_SNR)], axis=0)
# Generate Gaussian noise patterns to be shared over all datasets (but not every star)
noisy_train_patterns = noisy_train_stars - train_stars


# Generate datasets for every WFE resolution
for poly_psf_np, zernike_coef_np in zip(poly_psf_multires, zernike_coef_multires):
    
    # Generate numpy array from the SED list
    SED_np = np.array(SED_list)

    # Add same noise dataset to each WFE-resolution dataset
    noisy_train_stars = np.copy(poly_psf_np[:tot_train_stars, :, :]) + noisy_train_patterns

    # Save only one test dataset
    # Build param dicitionary
    dataset_params = {'d_max':d_max, 'max_order':max_order, 'x_lims':x_lims, 'y_lims':y_lims,
                     'grid_points':grid_points, 'n_bins':n_bins, 'max_wfe_rms':max_wfe_rms,
                     'oversampling_rate':oversampling_rate, 'output_Q':output_Q,
                     'output_dim':output_dim, 'LP_filter_length':LP_filter_length,
                     'pupil_diameter':WFE_resolutions[WFE_res_id], 'euclid_obsc':euclid_obsc,
                     'n_stars':n_test_stars}

    # Save dataset C coefficient matrix (reproductible dataset)
    C_poly = init_polyField[0].C_poly

    test_psf_dataset = {'stars' : poly_psf_np[tot_train_stars:, :, :],
                         'positions' : pos_np[tot_train_stars:, :],
                         'SEDs' : SED_np[tot_train_stars:, :, :],
                         'zernike_coef' : zernike_coef_np[tot_train_stars:, :, :],
                         'C_poly' : C_poly,
                         'parameters': dataset_params}

    np.save(output_folder + 'test_Euclid_res_id_' + dataset_id_str + '_wfeRes_' + str(WFE_resolutions[WFE_res_id]) + '.npy',
            test_psf_dataset, allow_pickle=True)



    # Save the different train datasets
    for it_glob in range(len(n_star_list)):

        n_train_stars = n_star_list[it_glob]

        # Build param dicitionary
        dataset_params = {'d_max':d_max, 'max_order':max_order, 'x_lims':x_lims, 'y_lims':y_lims,
                         'grid_points':grid_points, 'n_bins':n_bins, 'max_wfe_rms':max_wfe_rms,
                         'oversampling_rate':oversampling_rate, 'output_Q':output_Q,
                         'output_dim':output_dim, 'LP_filter_length':LP_filter_length,
                         'pupil_diameter':WFE_resolutions[WFE_res_id], 'euclid_obsc':euclid_obsc,
                         'n_stars':n_train_stars}

        train_psf_dataset = {'stars' : poly_psf_np[:n_train_stars, :, :],
                         'noisy_stars': noisy_train_stars[:n_train_stars, :, :],
                         'positions' : pos_np[:n_train_stars, :],
                         'SEDs' : SED_np[:n_train_stars, :, :],
                         'zernike_coef' : zernike_coef_np[:n_train_stars, :, :],
                         'C_poly' : C_poly,
                         'parameters': dataset_params}


        np.save(output_folder + 'train_Euclid_res_' + str(n_train_stars) + '_TrainStars_id_' + dataset_id_str + '_wfeRes_' + str(WFE_resolutions[WFE_res_id]) +'.npy',
                train_psf_dataset, allow_pickle=True)
        
    WFE_res_id += 1

# Load and test generated dataset
path = output_folder

dataset_4096 = np.load(path + 'train_Euclid_res_2000_TrainStars_id_002_wfeRes_'+str(WFE_resolutions[0])+'.npy', allow_pickle=True)[()]
dataset_256 = np.load(path + 'train_Euclid_res_2000_TrainStars_id_002_wfeRes_'+str(WFE_resolutions[1])+'.npy', allow_pickle=True)[()]

star_to_show = 0

plt.figure(figsize=(15,9))
plt.suptitle('Noisy star PSF', fontsize=30)
plt.subplot(131)
plt.imshow(dataset_4096['noisy_stars'][star_to_show,:,:], cmap='gist_stern');plt.colorbar()
plt.title('WFE dim: 4096', fontsize=20)
plt.subplot(132)
plt.imshow(dataset_256['noisy_stars'][star_to_show,:,:], cmap='gist_stern');plt.colorbar()
plt.title('WFE dim: 256', fontsize=20)
plt.subplot(133)
plt.imshow(np.abs(dataset_256['noisy_stars'][star_to_show,:,:] - dataset_4096['noisy_stars'][star_to_show,:,:] ), cmap='gist_stern');plt.colorbar()
plt.title('Absolute difference', fontsize=20)

plt.savefig(output_folder + 'multiple_WFE_resolution_dataset_psf_comparison.pdf')

print('\nFigure saved at: ' + output_folder)
print('\nDone!')

# Close log file
sys.stdout = old_stdout
log_file.close()
