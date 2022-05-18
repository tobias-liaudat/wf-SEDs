import numpy as np
import matplotlib.pyplot as plt
import wf_psf as wf_psf
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count, parallel_backend

import sys
import time

# Paths

# SED folder path
SED_path = '/home/ecentofanti/wf-psf/data/SEDs/save_SEDs/'                 # Candide
#SED_path = './../../../wf-psf/data/SEDs/save_SEDs/'                        # Local
#SED_path = '/feynman/work/dap/lcs/ec270266/wf-psf/data/SEDs/save_SEDs/'     # Feynman

# Output saving path (in node05 of candide or $WORK space on feynman)
output_folder = '/n05data/ecentofanti/WFE_sampling_test/super_res/'         # Candide
#output_folder = './../../../output/'                                       # Local
#output_folder = '/feynman/work/dap/lcs/ec270266/output/super_res/'         # Feynman

# Number of cpu on the machine (may differ from n_cpus available !!!)
n_cpus = cpu_count()
# Number of cpus to use for parallelization
n_cpus = 24

# Save output prints to logfile
old_stdout = sys.stdout
log_file = open(output_folder + 'output.log','w')
sys.stdout = log_file
print('Starting the log file.')

# Dataset ID
dataset_id = 5
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

# Values for getting 3xEuclid_resolution PSFs outputs.
original_out_Q = output_Q
original_out_dim = output_dim
super_out_Q = 1
super_out_res = 64

# Desired WFE resolutions
WFE_resolutions = [4096, 256]
#WFE_resolutions = [1024, 256]

print('\nInit dataset generation')

zernikes_multires = []
sim_PSF_toolkit_multires = []
gen_poly_fieldPSF_multires = []

for i, pupil_diameter_ in tqdm(enumerate(WFE_resolutions)):

    # Generate Zernike maps in max resolution
    zernikes_multires.append( wf_psf.utils.zernike_generator(n_zernikes=max_order, wfe_dim=pupil_diameter_) )
    
    # Initialize PSF simulator for each cpu available (no euclid obscurations and wfr_rms init)
    packed_PSFToolkit = [ wf_psf.SimPSFToolkit(
        zernikes_multires[i], max_order=max_order, max_wfe_rms=max_wfe_rms, oversampling_rate=oversampling_rate,
        output_Q=output_Q, output_dim=output_dim, pupil_diameter=pupil_diameter_, euclid_obsc=False,
        LP_filter_length=LP_filter_length)
        for j in range(n_cpus)
    ]
    sim_PSF_toolkit_multires.append( packed_PSFToolkit )

    # Initialize one PSF field for each resolution
    packed_polyField_PSF = [ wf_psf.GenPolyFieldPSF(sim_PSF_toolkit_multires[i][j], d_max=d_max,
        grid_points=grid_points, max_order=max_order,
        x_lims=x_lims, y_lims=y_lims, n_bins=n_bins,
        lim_max_wfe_rms=max_wfe_rms, auto_init=auto_init, verbose=verbose)
        for j in range(n_cpus)
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

# Share C_poly coefficients throughout all the different resolution models
for i in range(len(gen_poly_fieldPSF_multires)):
    for j in range(n_cpus):
        gen_poly_fieldPSF_multires[i][j].set_C_poly(init_polyField[0].C_poly)
        gen_poly_fieldPSF_multires[i][j].set_WFE_RMS(init_polyField[0].WFE_RMS)
        gen_poly_fieldPSF_multires[i][j].sim_psf_toolkit.obscurations = init_toolkit[i].obscurations

# Load the SEDs
stellar_SEDs = np.load(SED_path + 'SEDs.npy', allow_pickle=True)
stellar_lambdas = np.load(SED_path + 'lambdas.npy', allow_pickle=True)

# Generate all the stars positions and assign SEDs
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

# Print some info
cpu_info = ' - Number of available CPUs: {}'.format(n_cpus)
proc_info = ' - Total number of processes: {}'.format(n_procs)
print(cpu_info)
print(proc_info)

# Generate star list
star_id_list = [id_ for id_ in range(n_stars)]

# Function to get (i,j) from id
def unwrap_id(id, n_stars):
    i = int(id/n_stars)
    j = int(id - i * n_stars)
    return i, j

def print_status(star_id, i, j):
    print('\nStar ' +str(star_id)+ ' done!' + '   index=('+str(i)+','+str(j)+')')

# Get batches from a list
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

# Function to get one PSF
def simulate_star(star_id, gen_poly_fieldPSF_multires,i):
    i_,j_ = unwrap_id(star_id, n_cpus)
    _psf, _zernike, _ = gen_poly_fieldPSF_multires[i][j_].get_poly_PSF(xv_flat=pos_np[star_id, 0],
                                                           yv_flat=pos_np[star_id, 1],
                                                           SED=SED_list[star_id])
    # Change output parameters to get the super resolved PSF
    gen_poly_fieldPSF_multires[i][j_].sim_psf_toolkit.output_Q = super_out_Q
    gen_poly_fieldPSF_multires[i][j_].sim_psf_toolkit.output_dim = super_out_res
    super_psf, _, _ = gen_poly_fieldPSF_multires[i][j_].get_poly_PSF(xv_flat=pos_np[star_id, 0],
                                                           yv_flat=pos_np[star_id, 1],
                                                           SED=SED_list[star_id])
    # Put back original parameters
    gen_poly_fieldPSF_multires[i][j_].sim_psf_toolkit.output_Q = original_out_Q
    gen_poly_fieldPSF_multires[i][j_].sim_psf_toolkit.output_dim = original_out_dim
    #print_status(star_id, i, star_id)
    return (star_id, _psf, _zernike, super_psf)

# Measure time
start_time = time.time()

zernike_coef_multires = []
poly_psf_multires = []
index_multires = []
super_psf_multires = []

for i in range(len(WFE_resolutions)):
    index_i_list = []
    psf_i_list = []
    z_coef_i_list = []
    super_psf_i_list = []
    for batch in chunker(star_id_list, n_cpus):
        with parallel_backend("loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=n_cpus, verbose=100)(delayed(simulate_star)(_star_id, gen_poly_fieldPSF_multires,i)
                                                for _star_id in batch)
        index_batch,psf_batch,z_coef_batch,super_psf_batch = zip(*results)
        index_i_list.extend(index_batch)
        psf_i_list.extend(psf_batch)
        z_coef_i_list.extend(z_coef_batch)
        super_psf_i_list.extend(super_psf_batch)

    index_multires.append(np.array(index_i_list) )
    poly_psf_multires.append(np.array( psf_i_list)) 
    zernike_coef_multires.append(np.array(z_coef_i_list))
    super_psf_multires.append(np.array(super_psf_i_list))

end_time = time.time()
print('\nAll stars generated in '+ str(end_time-start_time) +' seconds')

#######################################
#            END PARALLEL             #
#######################################

# Add noise to generated PSFs and save datasets

# SNR varying randomly from 10 to 120 - shared over all WFE resolutions
rand_SNR = (np.random.rand(tot_train_stars) * 100) + 10
# Copy the training stars
train_stars = np.copy(np.array(poly_psf_multires[0])[:tot_train_stars, :, :])
# Add Gaussian noise to the observations
noisy_train_stars = np.stack([wf_psf.utils.add_noise(_im, desired_SNR=_SNR) 
                              for _im, _SNR in zip(train_stars, rand_SNR)], axis=0)
# Generate Gaussian noise patterns to be shared over all datasets (but not every star)
noisy_train_patterns = noisy_train_stars - train_stars

WFE_res_id = 0

# Generate datasets for every WFE resolution
for poly_psf_np, zernike_coef_np, super_psf_np in zip(poly_psf_multires, zernike_coef_multires, super_psf_multires):
    
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
    C_poly = gen_poly_fieldPSF_multires[WFE_res_id][n_cpus-1].C_poly

    test_psf_dataset = {'stars' : poly_psf_np[tot_train_stars:, :, :],
                        'super_res_stars' : super_psf_np[tot_train_stars:, :, :],
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
                         'super_res_stars' : super_psf_np[:n_train_stars, :, :],
                         'positions' : pos_np[:n_train_stars, :],
                         'SEDs' : SED_np[:n_train_stars, :, :],
                         'zernike_coef' : zernike_coef_np[:n_train_stars, :, :],
                         'C_poly' : C_poly,
                         'parameters': dataset_params}


        np.save(output_folder + 'train_Euclid_res_' + str(n_train_stars) + '_TrainStars_id_' + dataset_id_str + '_wfeRes_' + str(WFE_resolutions[WFE_res_id]) +'.npy',
                train_psf_dataset, allow_pickle=True)

    # Next desired resolution   
    WFE_res_id += 1

# Load and test generated dataset
path = output_folder

dataset_1 = np.load(path + 'train_Euclid_res_'+str(n_star_list[0])+'_TrainStars_id_00'+str(dataset_id)+'_wfeRes_'+str(WFE_resolutions[0])+'.npy', allow_pickle=True)[()]
dataset_2 = np.load(path + 'train_Euclid_res_'+str(n_star_list[0])+'_TrainStars_id_00'+str(dataset_id)+'_wfeRes_'+str(WFE_resolutions[1])+'.npy', allow_pickle=True)[()]

star_to_show = 0

plt.figure(figsize=(15,9))
plt.suptitle('Noisy star PSF', fontsize=30)
plt.subplot(231)
plt.imshow(dataset_1['noisy_stars'][star_to_show,:,:], cmap='gist_stern');plt.colorbar()
plt.title('WFE dim: '+str(WFE_resolutions[0]), fontsize=20)
plt.subplot(232)
plt.imshow(dataset_2['noisy_stars'][star_to_show,:,:], cmap='gist_stern');plt.colorbar()
plt.title('WFE dim: '+str(WFE_resolutions[1]), fontsize=20)
plt.subplot(233)
plt.imshow(np.abs(dataset_2['noisy_stars'][star_to_show,:,:] - dataset_1['noisy_stars'][star_to_show,:,:] ), cmap='gist_stern');plt.colorbar()
plt.title('Absolute difference', fontsize=20)

plt.subplot(234)
plt.imshow(dataset_1['super_res_stars'][star_to_show,:,:], cmap='gist_stern');plt.colorbar()
plt.title('WFE dim: '+str(WFE_resolutions[0])+' - 3xRes', fontsize=20)
plt.subplot(235)
plt.imshow(dataset_2['super_res_stars'][star_to_show,:,:], cmap='gist_stern');plt.colorbar()
plt.title('WFE dim: '+str(WFE_resolutions[1])+' - 3xRes', fontsize=20)
plt.subplot(236)
plt.imshow(np.abs(dataset_2['super_res_stars'][star_to_show,:,:] - dataset_1['super_res_stars'][star_to_show,:,:] ), cmap='gist_stern');plt.colorbar()
plt.title('Absolute difference - 3xRes', fontsize=20)

plt.savefig(output_folder + 'multiple_WFE_resolution_dataset_psf_comparison_'+str(WFE_resolutions[0])+'vs'+str(WFE_resolutions[1])+'_.pdf')

print('\nFigure saved at: ' + output_folder)
print('\nDone!')

# Close log file
sys.stdout = old_stdout
log_file.close()
