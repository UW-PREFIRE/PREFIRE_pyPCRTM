import numpy as np
import pyPCRTM
import netCDF4
import matplotlib.pyplot as plt  
import matplotlib as mpl
import sys, os
import copy
import PREFIRE_sim_tools
print(sys.path)

# This script will create plots of convolved Jacobians with the choosen SRF file
# This can be applied to the radiance Jacobians only
# set these values before running(see below):
#   filesave (0 or 1)
#   kt_choice (0-8)
#   srfuse (0-1)

#set filesave to 1 if you are ready to output to various files
filesave=0

#choose which Jacobian: 0=temperature, 1=h2o, 2=co2, 3=o3, 4=n2o, 5=co, 6=ch4
# 7 = surface temp, 8 = surface emissivity
kt_choice = 0

#choose which SRF version to use 0 = v0.09.1, 1 = v0.09.2:
srfuse=1

kt_labels = ['temp','h2o','co2','o3','n2o','co','ch4','surftemp','surfemis']
kt_units = ['[(W/(m$^2$ sr $\\mu$m))/K]','[(W/(m$^2$ sr $\\mu$m))/concentration]','[(W/(m$^2$ sr $\\mu$m))/concentration]','[(W/(m$^2$ sr $\\mu$m))/concentration]','[(W/(m$^2$ sr $\\mu$m))/concentration]','[(W/(m$^2$ sr $\\mu$m))/concentration]','[(W/(m$^2$ sr $\\mu$m))/concentration]','[K/K]','[K/unit emissivity]']
kt_range= [0.007,0.075,0.000065,0.06,0.007,0.0013,0.0011,0.13,7.0]


srffiles = ['/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.09.1_2020-02-21.nc','/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.09.2_2020-02-21.nc']
srf_labels = ['v0.09.1','v0.09.2']
srf_labels_fn = ['v0p09p1','v0p09p2']

#template for the filename to save
sfile ='Kt_rad_pcrtm_srf{0:s}_{1:s}_{2:s}.png'

F = pyPCRTM.PCRTM()

sensor_id = 2

F.init(sensor_id,
           output_jacob_flag=True,
           output_tr_flag=False,
           output_ch_flag=True,
           output_jacob_ch_flag=True,
           output_bt_flag=True,
           output_jacob_bt_flag=True)

F.psfc = 1013.0
F.pobs = 0.005
F.sensor_zen = 0.0
F.emis = 0.98 + np.zeros(F.num_monofreq, np.float32)

std_atm_names = ['tropic', 'midlat_summer', 'midlat_winter',
                 'subarc_summer', 'subarc_winter', 'us_standard']

nc = netCDF4.Dataset('../../PREFIRE_sim_tools/data/'+
                     'Modtran_standard_profiles_PCRTM_levels.nc','r')


for a, aname in enumerate(std_atm_names):

    F.z = nc['z'][:,a].astype(np.float32)
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

    r = F.forward_rt()

    #determines the Jacobian as set above
    if kt_choice == 0:
        Kt_pcrm = copy.deepcopy(r['krad_t'])
    else:
        if kt_choice <= 6:
            Kt_pcrm_all6 = copy.deepcopy(r['krad_mol'])
            Kt_pcrm = Kt_pcrm_all6[:,:,(kt_choice-1)]
        else:
            if kt_choice == 7:
                Kt_pcrm = copy.deepcopy(r['krad_tsurf'])
            else:
                Kt_pcrm = copy.deepcopy(r['krad_esurf'])
                

    w,wr,yc,_ = PREFIRE_sim_tools.TIRS.srf.apply_SRF_wngrid(Kt_pcrm, SRFfile=srffiles[srfuse],spec_grid='wl')
    ym = np.ma.masked_invalid(yc)
    winnum = (a+1)

    plt.figure(winnum,figsize=(12,5))
    ym=ym.transpose()

    if kt_choice <= 6:
        #plt.pcolormesh(wr,F.z,ym,cmap='bwr')
        plt.pcolormesh(wr,F.z,ym,vmin=(-1.0)*kt_range[kt_choice],vmax=kt_range[kt_choice],cmap='bwr')
        plt.xlabel('Wavelength ($\\mu$m)')
        plt.ylabel('profile level (km)')
        cbar = plt.colorbar()
        plt.title('PCRTM, SRF'+srf_labels[srfuse]+' Jacobian_rad, '+kt_labels[kt_choice]+', '+std_atm_names[a]+' atm profile '+kt_units[kt_choice])
    else:
         plt.ylim(top=kt_range[kt_choice])
         plt.plot(w, ym[0,:], label = std_atm_names[a] )
         plt.xlabel('Wavelength ($\\mu$m)')
         plt.ylabel(kt_units[kt_choice])
         plt.title('PCRTM, SRF'+srf_labels[srfuse]+' Jacobian_rad, '+kt_labels[kt_choice]+', '+std_atm_names[a]+' atm profile ')


    sfile_out = sfile.format(srf_labels_fn[srfuse],kt_labels[kt_choice],std_atm_names[a])
 
    if filesave == 0:
        plt.show()
    else:
        print(sfile_out)
        plt.savefig(os.path.join('/home/nnn/plots/PREFIRE/jacobians/SRFs',sfile_out))
        plt.close()

    








