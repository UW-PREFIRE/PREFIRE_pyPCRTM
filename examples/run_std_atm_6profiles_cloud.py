import numpy as np
import pyPCRTM
import netCDF4
import matplotlib.pyplot as plt  
import copy

#set cloud type: 1 = ice cloud, 2 = liquid, 0 = clear
#1=this test program will set an ice cloud at 506 hPa with reff = 10 microns
#2=this test program will set an ice cloud at 892 hPa with reff = 20 microns
cloudtype = 1

# set cloud optical depth
cod = 1.0


#begin script
cld_prof =  np.zeros(100)
cld_od =  np.zeros(100)
cld_de =  np.zeros(100)
cld_dp =  np.zeros(100)

#I use the next line as I would use 'stop' in idl
#raise ValueError()

F = pyPCRTM.PCRTM()

F.init(2, output_jacob_flag=True, output_ch_flag=True, output_jacob_ch_flag=True)
#F.init(2, output_jacob_flag=True, output_ch_flag=True)
#F.init(7, output_jacob_flag=True, output_ch_flag=True, output_jacob_ch_flag=True)

plev = F.plevels

#set cloud pressure profile
for i in range(100):
    cld_dp[i] = (plev[i] + plev[i+1])/2.0


#add clouds properties if cloudtype 1 or 2
if cloudtype == 1:
    pind = 75 #75 is 506 hPa pressure level
    cld_prof[pind] = cloudtype
    cld_od[pind] = cod
    cld_de[pind] = 2.0*20.0 #reff set to 20 microns for ice

if cloudtype == 2:
    pind = 92 #92 is 892 hPa pressure level
    cld_prof[pind] = cloudtype
    cld_od[pind] = cod
    cld_de[pind] = 2.0*10.0 #reff set to 10 microns for liquid

F.psfc = 1013.0
F.pobs = 0.005
F.sensor_zen = 0.0
#set surface emissivity to 0.98
F.emis = 0.98 + np.zeros(F.num_monofreq, np.float32)

std_atm_names = ['tropic', 'midlat_summer', 'midlat_winter',
                 'subarc_summer', 'subarc_winter', 'us_standard']
color_names = ['b-', 'k-', 'g-','r-', 'c-', 'm-']

nc = netCDF4.Dataset('/home/nnn/projects/PREFIRE_sim_tools/data/'+
                     'Modtran_standard_profiles_PCRTM_levels.nc','r')

for a, aname in enumerate(std_atm_names):
    
    F.tskin = nc['temp'][-1,a]
    #F.z = nc['z'][:,a].astype(np.float32)
    F.tlev = nc['temp'][:,a].astype(np.float32)
    # very approx conversion from ppm to q [g/kg]
    q = nc['h2o'][:,a] * 0.622 * 1e-6 * 1e3
    
    F.h2o = q.astype(np.float32)
    F.co2 = nc['co2'][:,a].astype(np.float32)
    F.o3 = nc['o3'][:,a].astype(np.float32)
    F.n2o = nc['n2o'][:,a].astype(np.float32)
    F.co = nc['co'][:,a].astype(np.float32)
    F.ch4 = nc['ch4'][:,a].astype(np.float32)

    #check if there is any cloud present as set above
    if sum(cld_prof[:]) >= 1:
        F.cld = cld_prof[:]
        F.cldDe = cld_de[:]
        F.cldOD = cld_od[:]
        F.cldP = cld_dp[:]

    r = F.forward_rt()
    
    #example of how you would create a variableof the layer transmission output
    layertrans_current = copy.deepcopy(r['layer_trans'])

    #plt.plot(F.tlev,F.plevels, color_names[a], label = std_atm_names[a] )
    #plt.plot(q,F.plevels, color_names[a], label = std_atm_names[a] )
    #plt.plot(10000.0/r['wn'], r['bt'], label = std_atm_names[a] )
    #plt.plot(10000.0/r['wn'], r['rad'], label = std_atm_names[a] )
    #if a == 1:
    plt.plot(r['wn'], r['rad'], label = std_atm_names[a] )


#plt.ylim(1100,0)

#plt.xlabel('Temp (K)')
#plt.ylabel('pressure (hPa)')

#plt.xlabel('q (g/kg)')
#plt.ylabel('altitude (km)')

#plt.xlabel('Wavelength ($\mu$m)')   
#plt.ylabel('BT (K)') 

plt.xlabel('Wavelength ($\mu$m)')   
plt.ylabel('Radiance (mW/(m$^2$ sr cm$^{-1}$))')

#plt.xlim(3.0,60.0)

plt.legend(loc='upper right')
plt.show()


raise ValueError()
