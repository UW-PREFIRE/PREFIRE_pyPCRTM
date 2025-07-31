#
# test script to investigate how cloud specification works in PCRTM.
#
# I believe what the script output demonstrates (and confirmed by 
# reviewing PCRTM_CALC_CLOUD.f90) is that the cloud pressure value
# is used to compute a sub-layer cloud temperature by interpolating
# from the temperature profile at the bounding layers.
# the bounding layers are determined by the where the cloud index
# flag is nonzero.
# this does mean that the cloud temperature is actually extrapolated,
# if the cloud pressure is outside the layer defined by
# P[k], P[k+1] for a value of k determined by the cloud flag location
# in the profile.
# That extrapolation process is illustrated by this example, where
# the cloud pressure varies from 500 to 1000 hPa.
# in one version, the index value is kept fixed (which is not correct),
# and you get a different result compared to moving the index value
# along with the cloud pressure. 
#
import numpy as np
import pyPCRTM
import netCDF4

F = pyPCRTM.PCRTM()

#             output_jacob_flag=False,
#             output_tr_flag=False,
#             output_ch_flag=True,
#             output_jacob_ch_flag=False,
#             output_bt_flag=True,
#             output_jacob_bt_flag=False,

F.init(2, output_ch_flag=True, output_bt_flag=True,
       output_jacob_flag=True)

F.psfc = 1013.0
F.pobs = 0.005
F.sensor_zen = 0.0
F.emis = np.ones(F.num_monofreq, np.float32)

std_atm_names = ['tropic', 'midlat_summer', 'midlat_winter',
                 'subarc_summer', 'subarc_winter', 'us_standard']


# the std atm that will be used
selected_atm = 'tropic'
cloud_P_range = [500,1000]
cloud_P_step = 1.0
cloud_P_vals = np.arange(cloud_P_range[0], cloud_P_range[1], cloud_P_step)
OD_val = 5.0
De_val = 30.0

a = std_atm_names.index(selected_atm)
ncfile = '../../PREFIRE_sim_tools/data/Modtran_standard_profiles_PCRTM_levels.nc'
P = np.loadtxt('../../PREFIRE_sim_tools/data/plevs101.txt')
k_fixed = P.searchsorted(0.5*(cloud_P_range[0]+cloud_P_range[1]))-1

with netCDF4.Dataset(ncfile, 'r') as nc:
    
    F.tskin = nc['temp'][-1,a]

    F.tlev = nc['temp'][:,a].astype(np.float32)
    # very approx conversion from ppm to q [g/kg]
    q = nc['h2o'][:,a] * 0.622 * 1e-6 * 1e3
    F.h2o = q.astype(np.float32)
    F.co2 = nc['co2'][:,a].astype(np.float32)
    F.o3 = nc['o3'][:,a].astype(np.float32)
    F.n2o = nc['n2o'][:,a].astype(np.float32)
    F.co = nc['co'][:,a].astype(np.float32)
    F.ch4 = nc['ch4'][:,a].astype(np.float32)

rad_fix_clayer = np.zeros((5421,cloud_P_vals.shape[0]))
rad_adj_clayer = np.zeros((5421,cloud_P_vals.shape[0]))
bt_fix_clayer = np.zeros((5421,cloud_P_vals.shape[0]))
bt_adj_clayer = np.zeros((5421,cloud_P_vals.shape[0]))

def set_cld_profiles(F, k, P_val, De_val, OD_val):
    cld = np.zeros(100, np.int32)
    cldP = np.zeros(100, np.float32)
    cldDe = np.zeros(100, np.float32)
    cldOD = np.zeros(100, np.float32)
    cld[k] = 1
    cldP[k] = P_val
    cldDe[k] = De_val
    cldOD[k] = OD_val
    F.cld = cld
    F.cldP = cldP
    F.cldDe = cldDe
    F.cldOD = cldOD


for n in range(cloud_P_vals.shape[0]):

    P_val = cloud_P_vals[n]
    print('test {0:d} of {1:d} pressure = {2:8.3f}'.format(
        n,cloud_P_vals.shape[0],P_val))

    set_cld_profiles(F, k_fixed, P_val, De_val, OD_val)

    dat = F.forward_rt()
    wn = dat['wn']
    rad_fix_clayer[:,n] = dat['rad']
    bt_fix_clayer[:,n] = dat['bt']

    k = P.searchsorted(P_val) - 1
    set_cld_profiles(F, k, P_val, De_val, OD_val)

    dat = F.forward_rt()
    wn = dat['wn']
    rad_adj_clayer[:,n] = dat['rad']
    bt_adj_clayer[:,n] = dat['bt']


np.savez('test_cloud_pressure_radiance_output.npz',
         wn=wn, De_val=De_val, OD_val=OD_val, P_vals = cloud_P_vals,
         rad_fix_clayer=rad_fix_clayer, rad_adj_clayer=rad_adj_clayer,
         bt_fix_clayer=bt_fix_clayer, bt_adj_clayer=bt_adj_clayer)
