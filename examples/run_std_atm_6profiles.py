import numpy as np
import pyPCRTM
import netCDF4
import matplotlib.pyplot as plt  
import copy

F = pyPCRTM.PCRTM()

#             output_jacob_flag=False,
#             output_tr_flag=False,
#             output_ch_flag=True,
#             output_jacob_ch_flag=False,
#             output_bt_flag=True,
#             output_jacob_bt_flag=False,

F.init(2, False, False, True, False, True, False)

F.psfc = 1013.0
F.pobs = 0.005
F.sensor_zen = 0.0
F.emis = 0.98 + np.zeros(F.num_monofreq, np.float32)

std_atm_names = ['tropic', 'midlat_summer', 'midlat_winter',
                 'subarc_summer', 'subarc_winter', 'us_standard']
color_names = ['b-', 'k-', 'g-','r-', 'c-', 'm-']

nc = netCDF4.Dataset('/home/nnn/projects/PREFIRE_sim_tools/data/'+
                     'Modtran_standard_profiles_PCRTM_levels.nc','r')

for a, aname in enumerate(std_atm_names):
    
    F.tskin = nc['temp'][-1,a]
    F.z = nc['z'][:,a].astype(np.float32)
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

    trans = copy.deepcopy(r['layer_trans'])
#    plt.plot(F.tlev,F.plevels, color_names[a], label = std_atm_names[a] )
#   plt.plot(q,F.plevels, color_names[a], label = std_atm_names[a] )
    plt.plot(r['wn'], r['bt'], label = std_atm_names[a] )


#plt.ylim(1100,0)

#plt.xlabel('Temp (K)')
#plt.ylabel('pressure (hPa)')

#plt.xlabel('q (g/kg)')
#plt.ylabel('altitude (km)')

plt.xlabel('Wavenumber (1/cm)')   
plt.ylabel('bt (K)') 


plt.legend(loc='lower left')
plt.show()
