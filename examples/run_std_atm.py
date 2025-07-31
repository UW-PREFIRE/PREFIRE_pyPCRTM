import datetime

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

F.init(2, True, True, True, True, True, True)

F.psfc = 1013.0
F.pobs = 0.005
F.sensor_zen = 0.0
F.emis = 0.98 + np.zeros(F.num_monofreq, np.float32)

std_atm_names = ['tropic', 'midlat_summer', 'midlat_winter',
                 'subarc_summer', 'subarc_winter', 'us_standard']

nc = netCDF4.Dataset('../../PREFIRE_sim_tools/data/'+
                     'Modtran_standard_profiles_PCRTM_levels.nc','r')

for a, aname in enumerate(std_atm_names):
    
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

    with netCDF4.Dataset('PCRTM_forward_stdatm{0:1d}_{1:s}.nc'.format(
            a+1, aname), 'w') as ncr:

        ncr.createDimension('spectral', 5421)
        for v in 'wavenum', 'rad', 'btemp', 'tstar':
            ncr.createVariable(v, np.float, ('spectral',))

        ncr['wavenum'][:] = r['wn']
        ncr['rad'][:] = r['rad']
        ncr['btemp'][:] = r['bt']
        ncr['tstar'][:] = np.prod(r['layer_trans'][:,:98], axis=1)

        ncr['wavenum'].setncattr('long_name', 'channel wavenumber')
        ncr['wavenum'].setncattr('units', '1/cm')

        ncr['rad'].setncattr('long_name', 'spectral radiance')
        ncr['rad'].setncattr('units', 'mW / [m^2 sr cm^-1]')

        ncr['btemp'].setncattr('long_name', 'brightness temperature')
        ncr['btemp'].setncattr('units', 'K')

        ncr['tstar'].setncattr('long_name', 'total atmosphere transmission')
        ncr['tstar'].setncattr('units', 'none')

        ncr.setncattr('comment1', 'PCRTM V3.4 forward calculation')
        ncr.setncattr(
            'comment2',
            'Produced by A., using pyPCRTMwrapper/examples/run_std_atm.py')
        ncr.setncattr(
            'date_created', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    F.co2 = np.zeros(101) + 400.0
    r = F.forward_rt()

    with netCDF4.Dataset('PCRTM_forward_stdatm{0:1d}_{1:s}_400ppmCO2.nc'.format(
            a+1, aname), 'w') as ncr:

        ncr.createDimension('spectral', 5421)
        for v in 'wavenum', 'rad', 'btemp', 'tstar':
            ncr.createVariable(v, np.float, ('spectral',))

        ncr['wavenum'][:] = r['wn']
        ncr['rad'][:] = r['rad']
        ncr['btemp'][:] = r['bt']
        ncr['tstar'][:] = np.prod(r['layer_trans'][:,:98], axis=1)

        ncr['wavenum'].setncattr('long_name', 'channel wavenumber')
        ncr['wavenum'].setncattr('units', '1/cm')

        ncr['rad'].setncattr('long_name', 'spectral radiance')
        ncr['rad'].setncattr('units', 'mW / [m^2 sr cm^-1]')

        ncr['btemp'].setncattr('long_name', 'brightness temperature')
        ncr['btemp'].setncattr('units', 'K')

        ncr['tstar'].setncattr('long_name', 'total atmosphere transmission')
        ncr['tstar'].setncattr('units', 'none')

        ncr.setncattr('comment1', 'PCRTM V3.4 forward calculation')
        ncr.setncattr(
            'comment2', 
            'Produced by A., using pyPCRTMwrapper/examples/run_std_atm.py')
        ncr.setncattr(
            'comment3',
            'Standard atmosphere, except CO2 mole fraction = 400.0 ppm')
        ncr.setncattr(
            'date_created', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
