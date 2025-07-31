from ctypes import CDLL, POINTER, c_bool, c_int, c_long, c_float, c_double
import numpy as np
import os.path


# implementation details:
# I was worried what might happen if you somehow create more than one
# of these wrapper objects - there is "global" data in the module at the
# fortran level, and I am unsure how that interacts with multiple instances.
# So, for now implement this as a Singleton pattern so there can be only
# one instance created.
# this is following ClassVariableSingleton.py from:
# https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html
#
# note, now redesigned to allow a singleton for shared objects that are
# different PCRTM versions. This is done by hiding a dictionary as a class
# attribute.
#


#
# base class implements the following:
# gets the directory paths to the code, data, project dirs (based
#     on the value of __file__)
# loads the 101-pressure level data file (common to all PCRTM variants)
# manages the singleton, based on what object filename is input during
# creation. Base class should not be run by itself.
#
# subclasses must do several operations:
#   * call this class creation method, with the name of the object file
#   * and (optionally) a string describing the model version.
#   * define their own init method to define the number of molecules
#     (in the class attribute  _nmol).
#  
class PCRTMbase:
    __instance = {}
    _code_dir = os.path.dirname(os.path.realpath(__file__))
    _proj_dir = os.path.split(_code_dir)[0]
    _data_dir = os.path.join(_proj_dir, 'data')
    _plevels = np.loadtxt(os.path.join(_data_dir, 'plevs101.txt'))
 
    def __new__(cls, object_file, model_version=''):

        # we hard code the model_version here - this means
        # we cannot make more than one of this object.
        #model_version = 'PREFIRE_V3.4'

        if object_file not in PCRTMbase.__instance:

            PCRTMbase.__instance[object_file] = object.__new__(cls)

            PCRTMbase.__instance[object_file]._object_file = object_file
            PCRTMbase.__instance[object_file]._model_version = model_version

            cdll_path = os.path.join(cls._code_dir, object_file)
            PCRTMbase.__instance[object_file]._m = CDLL(cdll_path)

            # set ctypes for only those functions that return scalar types.
            PCRTMbase.__instance[object_file]._m.pcrtmw_get_init_status.restype = c_bool
            PCRTMbase.__instance[object_file]._m.pcrtmw_get_run_status.restype = c_bool
            PCRTMbase.__instance[object_file]._m.pcrtmw_get_tskin.restype = c_float
            PCRTMbase.__instance[object_file]._m.pcrtmw_get_nlevels.restype = c_int
            PCRTMbase.__instance[object_file]._m.pcrtmw_get_nchannels.restype = c_int
            PCRTMbase.__instance[object_file]._m.pcrtmw_get_neof.restype = c_int
            PCRTMbase.__instance[object_file]._m.pcrtmw_get_nmonofreq.restype = c_int
            PCRTMbase.__instance[object_file]._m.pcrtmw_get_nbands.restype = c_int

            PCRTMbase.__instance[object_file]._m.pcrtmw_start()

        return PCRTMbase.__instance[object_file]


class PCRTM(PCRTMbase):
    """
    singleton class wrapper for the PCRTM fortran90 library

    Example usage::

        >>> import pyPCRTM
        >>> F = pyPCRTM.PCRTM()

        >>> F.init(2, output_ch_flag=True)
        >>> F.psfc = 1003.25
        >>> F.pobs = 0.005
        >>> F.sensor_zen = 60.0
        >>> F.emis = 0.98 + np.zeros(F.num_monofreq, np.float32)
        >>> F.tskin = 280.0

        # set remaining profiles, assuming you have np.float32 arrays
        # containing the profile information.
        >>> F.tlev = tlev
        # etc...

        # run forward model
        >>> dat = F.forward_rt()

        # now, the spectrum could be plotted with e.g. matplotlib:
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(dat['wn'], dat['rad'])

    """
 
    def __new__(cls):
        # we hard code the model_version here - this means
        # we cannot make more than one of this object.
        model_version = 'PREFIRE_V3.4'
        object_file = 'PCRTM_wrapper_module.so'

        instance = super(PCRTM, cls).__new__(cls, object_file, model_version)

        return instance


    def __init__(self):
        self._nmol = 6

    def _get_init_status(self):
        return self._m.pcrtmw_get_init_status()
    def _get_run_status(self):
        return self._m.pcrtmw_get_run_status()

    def _get_nlevels(self):
        return self._m.pcrtmw_get_nlevels()
    def _get_nlayers(self):
        return self._m.pcrtmw_get_nlayers()
    def _get_nchannels(self):
        return self._m.pcrtmw_get_nchannels()
    def _get_neof(self):
        return self._m.pcrtmw_get_neof()
    def _get_nmonofreq(self):
        return self._m.pcrtmw_get_nmonofreq()
    def _get_nbands(self):
        return self._m.pcrtmw_get_nbands()

    def _get_nchannels_per_band(self):
        nchannels = np.zeros(self.num_bands, np.int32)
        self._m.pcrtmw_get_nchannels_per_band(
            nchannels.ctypes.data_as(POINTER(c_int)),
            c_int(nchannels.shape[0]))
        return nchannels
    def _get_neof_per_band(self):
        neof = np.zeros(self.num_bands, np.int32)
        self._m.pcrtmw_get_neof_per_band(
            neof.ctypes.data_as(POINTER(c_int)),
            c_int(neof.shape[0]))
        return neof
    def _get_nmonofreq_per_band(self):
        nmonofreq = np.zeros(self.num_bands, np.int32)
        self._m.pcrtmw_get_nmonofreq_per_band(
            nmonofreq.ctypes.data_as(POINTER(c_int)),
            c_int(nmonofreq.shape[0]))
        return nmonofreq

    def _get_monofreq(self):
        monofreq = np.zeros(self.num_monofreq, np.float64)
        self._m.pcrtmw_get_monofreq(
            monofreq.ctypes.data_as(POINTER(c_double)),
            c_int(monofreq.shape[0]))
        return monofreq

    def _get_geom(self):
        geom = np.zeros(3, dtype=np.float32)
        self._m.pcrtmw_get_geom(geom.ctypes.data_as(POINTER(c_float)))
        return geom
    def _set_geom(self, psfc, pobs, sensor_zen):
        geom = np.array([psfc, pobs, sensor_zen]).astype(np.float32)
        self._m.pcrtmw_set_geom(geom.ctypes.data_as(POINTER(c_float)))

    def _get_psfc(self):
        geom = self._get_geom()
        return geom[0]
    def _set_psfc(self,psfc):
        geom = self._get_geom()
        self._set_geom(psfc, geom[1], geom[2])

    def _get_pobs(self):
        geom = self._get_geom()
        return geom[1]
    def _set_pobs(self,pobs):
        geom = self._get_geom()
        self._set_geom(geom[0], pobs, geom[2])

    def _get_sensor_zen(self):
        geom = self._get_geom()
        return geom[2]
    def _set_sensor_zen(self,sensor_zen):
        geom = self._get_geom()
        self._set_geom(geom[0], geom[1], sensor_zen)

    def _get_tskin(self):
        return self._m.pcrtmw_get_tskin()
    def _set_tskin(self, tskin):
        self._m.pcrtmw_set_tskin(c_float(tskin))

    def _get_emis(self):
        emis = np.zeros(self.num_monofreq, np.float32)
        self._m.pcrtmw_get_emis(emis.ctypes.data_as(POINTER(c_float)),
                                c_int(emis.shape[0]))
        return emis

    def _set_emis(self, emis):
        # the C library assumes pointer to float32, so convert it while
        # processing through np.array (so a list would be acceptable).
        # same pattern is used below in tlev, mol profile set helpers.
        emis_f32 = np.array(emis, np.float32)
        if emis_f32.shape != (self.num_monofreq,):
            raise ValueError(
                'Incorrect shape for emis array, expecting: '+
                str((self.num_monofreq,)))
        self._m.pcrtmw_set_emis(emis_f32.ctypes.data_as(POINTER(c_float)),
                                c_int(emis.shape[0]))

    def _get_tlev(self):
        tlev = np.zeros(self.num_levels, np.float32)
        self._m.pcrtmw_get_tlev(tlev.ctypes.data_as(POINTER(c_float)))
        return tlev

    def _set_tlev(self, tlev):
        tlev_f32 = np.array(tlev, np.float32)
        if tlev_f32.shape != (self.num_levels,):
            raise ValueError(
                'Incorrect shape for T level array, expecting: '+
                str((self.num_levels,)))
        self._m.pcrtmw_set_tlev(tlev_f32.ctypes.data_as(POINTER(c_float)))

    _mol_map = ['', 'h2o', 'co2', 'o3', 'n2o', 'co', 'ch4']

    def _get_mol(self, mol):
        try:
            mol_id = self._mol_map.index(mol)
        except ValueError:
            raise ValueError(
                'Unknown molecule name: ' + mol)
        vmr = np.zeros(self.num_levels, np.float32)
        self._m.pcrtmw_get_vmr(
            vmr.ctypes.data_as(POINTER(c_float)), c_int(mol_id))
        return vmr

    def _set_mol(self, vmr, mol):
        vmr_f32 = np.array(vmr, np.float32)
        mol_id = self._mol_map.index(mol)
        if vmr_f32.shape != (self.num_levels,):
            raise ValueError(
                'Incorrect shape for VMR level array, expecting: '+
                str((self.num_levels,)))
        self._m.pcrtmw_set_vmr(
            vmr_f32.ctypes.data_as(POINTER(c_float)), c_int(mol_id))

    def _get_h2o(self):
        return self._get_mol('h2o')
    def _get_co2(self):
        return self._get_mol('co2')
    def _get_o3(self):
        return self._get_mol('o3')
    def _get_n2o(self):
        return self._get_mol('n2o')
    def _get_co(self):
        return self._get_mol('co')
    def _get_ch4(self):
        return self._get_mol('ch4')

    def _set_h2o(self,vmr):
        return self._set_mol(vmr,'h2o')
    def _set_co2(self,vmr):
        return self._set_mol(vmr,'co2')
    def _set_o3(self,vmr):
        return self._set_mol(vmr,'o3')
    def _set_n2o(self,vmr):
        return self._set_mol(vmr,'n2o')
    def _set_co(self,vmr):
        return self._set_mol(vmr,'co')
    def _set_ch4(self,vmr):
        return self._set_mol(vmr,'ch4')

    # Note - here the array is set as a pointing to c_int, with numpy data type
    # float32. This might be an ifort quirk, that c_int is a 32 bit signed integer?
    # If the numpy type is np.int16 (as one would expect for c_int), this results
    # in data corruption in the Fortran compound user type.
    def _get_cld(self):
        cld = np.zeros(self.num_levels-1, np.int32)
        self._m.pcrtmw_get_cld(cld.ctypes.data_as(POINTER(c_int)))
        return cld

    def _set_cld(self, cld):
        cld_i32 = np.array(cld, np.int32)
        if cld_i32.shape != (self.num_levels-1,):
            raise ValueError(
                'Incorrect shape for cld array, expecting: '+
                str((self.num_levels-1,)))
        self._m.pcrtmw_set_cld(cld_i32.ctypes.data_as(POINTER(c_int)))

    def _get_cldP(self):
        cldP = np.zeros(self.num_levels-1, np.float32)
        self._m.pcrtmw_get_cldP(cldP.ctypes.data_as(POINTER(c_float)))
        return cldP

    def _set_cldP(self, cldP):
        cldP_f32 = np.array(cldP, np.float32)
        if cldP_f32.shape != (self.num_levels-1,):
            raise ValueError(
                'Incorrect shape for cldP array, expecting: '+
                str((self.num_levels-1,)))
        self._m.pcrtmw_set_cldP(cldP_f32.ctypes.data_as(POINTER(c_float)))

    def _get_cldDe(self):
        cldDe = np.zeros(self.num_levels-1, np.float32)
        self._m.pcrtmw_get_cldDe(cldDe.ctypes.data_as(POINTER(c_float)))
        return cldDe

    def _set_cldDe(self, cldDe):
        cldDe_f32 = np.array(cldDe, np.float32)
        if cldDe_f32.shape != (self.num_levels-1,):
            raise ValueError(
                'Incorrect shape for cldDe array, expecting: '+
                str((self.num_levels-1,)))
        self._m.pcrtmw_set_cldDe(cldDe_f32.ctypes.data_as(POINTER(c_float)))

    def _get_cldOD(self):
        cldOD = np.zeros(self.num_levels-1, np.float32)
        self._m.pcrtmw_get_cldOD(cldOD.ctypes.data_as(POINTER(c_float)))
        return cldOD

    def _set_cldOD(self, cldOD):
        cldOD_f32 = np.array(cldOD, np.float32)
        if cldOD_f32.shape != (self.num_levels-1,):
            raise ValueError(
                'Incorrect shape for cldOD array, expecting: '+
                str((self.num_levels-1,)))
        self._m.pcrtmw_set_cldOD(cldOD_f32.ctypes.data_as(POINTER(c_float)))

    def _get_plevels(self):
        return self._plevels.copy()

    def init(self,
             sensor_id,
             output_jacob_flag=False,
             output_tr_flag=False,
             output_ch_flag=True,
             output_jacob_ch_flag=False,
             output_bt_flag=True,
             output_jacob_bt_flag=False,
             molindx = None,
             scalfac = None):
        """
        initialize PCRTM module with a specific sensor_id.
        The various output* keywords control whether those data are computed
        and stored in the output data arrays after a call to forward.

        Parameters
        __________

        sensor_id : int
            the id number for the desired sensor. From the PCRTM documentation:

            1: CLARREO 0.1 cm-1 spectral resolution

            2: CLARREO 0.5 cm-1 spectral resolution

            3: CLARREO 1.0 cm-1 spectral resolution

            see PCRTM documentation for further selections.
        
        output_jacob_flag : bool
            controls output of the jacobian

        output_tr_flag : bool
            controls output of the transmittance (current unimplemented)

        output_ch_flag : bool
            controls whether output arrays are generated for the sensor channels
            By default, output would only be generated for the EOF
            coefficients and the monochromatic radiances.

        output_jacob_ch_flag : bool
            controls whether jacobian outputs are generated for the sensor
            channels.

        output_bt_flag : bool
            controls whether the brightness temperature is computed in addition
            to the radiance.

        output_jacob_bt_flag : bool
            controls whether jacobian outputs are computed in brightness temperature
            in addition to the radiance.

        molindx : list or ndarray
            6 element integer array (see PCRTM documentation), controlling the
            treatment of the molecular species (in order: H2O, CO2, O3, N2O, CO, CH4).

            molindx = 0 profile is fixed to the US Std profile value, and is
            treated as "fixed" (I think this means, no jacobian is computed).

            molindx = 1, profile is fixed to US Std, treated as "variable".

            molindx = 2, profile is variable and must be input (the default).

            This is untested for any but the default value.

        scalfac : list or ndarray
            6 element floating point array with scale factors to apply to each
            gas profile. Default is 1.0 for each (equivalent to no scaling.)

        """

        if molindx is None:
            molindx = np.zeros(self._nmol, dtype=np.int32)
            molindx[:] = 2
        if scalfac is None:
            scalfac = np.zeros(self._nmol, dtype=np.float32)
            scalfac[:] = 1.0

        # make sure these are the correct np array, if user
        # input a list, for example.
        molindx = np.array(molindx, dtype=np.int32)
        scalfac = np.array(scalfac, dtype=np.float32)
        if molindx.shape != (self._nmol,):
            raise ValueError('Wrong shape for molindx, '+
                             'expecting ({:d},)'.format(self._nmol))
        if scalfac.shape != (self._nmol,):
            raise ValueError('Wrong shape for scalfac, '+
                             'expecting ({:d},)'.format(self._nmol))
        self._m.pcrtmw_init(
            c_int(sensor_id),
            c_bool(output_jacob_flag),
            c_bool(output_tr_flag),
            c_bool(output_ch_flag),
            c_bool(output_jacob_ch_flag),
            c_bool(output_bt_flag),
            c_bool(output_jacob_bt_flag),
            molindx.ctypes.data_as(POINTER(c_int)),
            scalfac.ctypes.data_as(POINTER(c_float)))

    init_status = property(
        _get_init_status, None,
        doc="Get the initialization status of the PCRTM object (boolean)")
    run_status = property(
        _get_run_status, None,
        doc="Get the run status of the PCRTM object (boolean), meaning, "+
        "True if the fortran library has executed the forward model run")

    num_levels = property(
        _get_nlevels, None,
        doc="get the number of vertical levels (always 101)")
    num_layers = property(
        _get_nlayers, None,
        doc="get the number of vertical layers (always 100)")
    num_channels = property(
        _get_nchannels, None,
        doc="get the number of sensor channels. Dependent on the sensor_id "+
        "used during the model initialization")
    num_eof = property(
        _get_neof, None,
        doc="get the number of EOFs. Dependent on the sensor_id "+
        "used during the model initialization")
    num_monofreq = property(
        _get_nmonofreq, None,
        doc="get the number of monochromatic frequencies. Dependent on the "
        "sensor_id used during the model initialization")
    num_bands = property(
        _get_nbands, None,
        doc="get the number of bands. Dependent on the "
        "sensor_id used during the model initialization")

    num_channels_per_band = property(
        _get_nchannels_per_band, None,
        doc="get the number of channels per band. Dependent on the "
        "sensor_id used during the model initialization")
    num_eof_per_band = property(
        _get_neof_per_band, None,
        doc="get the number of eof per band. Dependent on the "
        "sensor_id used during the model initialization")
    num_monofreq_per_band = property(
        _get_nmonofreq_per_band, None,
        doc="get the number of monochromatic frequencies per band. "
        "Dependent on the sensor_id used during the model initialization")

    monofreq = property(
        _get_monofreq, None,
        doc="get the array of monochromatic frequencies in the PC Solution")

    plevels = property(
        _get_plevels, None,
        doc="get the fixed PCRTM pressure levels [hPa]")
    psfc = property(
        _get_psfc, _set_psfc,
        doc="get or set surface pressure [hPa]")
    pobs = property(
        _get_pobs, _set_pobs,
        doc="get or set pressure level of observer [hPa]\n"+
        "0.005 is the TOA and equivalent to a satellite observer")
    sensor_zen = property(
        _get_sensor_zen, _set_sensor_zen,
        doc="get or set sensor zenith angle [deg]")

    tskin = property(
        _get_tskin, _set_tskin,
        doc="get or set the surface temperature [K]")
    emis = property(
        _get_emis, _set_emis,
        doc="get or set the surface emissivity, a (Vm,) shaped array")
    tlev = property(
        _get_tlev, _set_tlev,
        doc="get or set the temperature profile on the 101 pressure levels [K]")

    h2o = property(
        _get_h2o, _set_h2o, 
        doc="get or set water vapor profile on the 101 pressure levels [g/kg]")
    co2 = property(
        _get_co2, _set_co2,
        doc="get or set co2 profile on the 101 pressure levels [ppm]")
    o3  = property(
        _get_o3,  _set_o3,
        doc="get or set o3 profile on the 101 pressure levels [ppm]")
    n2o = property(
        _get_n2o, _set_n2o,
        doc="get or set n2o profile on the 101 pressure levels [ppm]")
    co  = property(
        _get_co,  _set_co,
        doc="get or set co profile on the 101 pressure levels [ppm]")
    ch4 = property(
        _get_ch4, _set_ch4,
        doc="get or set ch4 profile on the 101 pressure levels [ppm]")

    cld = property(
        _get_cld, _set_cld,
        doc="get or set cloud flag profile on the 100 pressure layers. "+
        "flag options are 0=clear, 1=ice, 2=water")

    cldP = property(
        _get_cldP, _set_cldP,
        doc="get or set cloud pressure profile on the 100 pressure layers")

    cldDe = property(
        _get_cldDe, _set_cldDe,
        doc="get or set cloud effective Diameter profile on the 100 pressure layers [um]")

    cldOD = property(
        _get_cldOD, _set_cldOD,
        doc="get or set cloud optical depth profile on the 100 pressure layers")

    def forward_rt(self, num_profile_jacob = None):
        """
        Runs the PCRTM forward model given the current data specifying the
        atmospheric conditions, sensor geometry, etc.

        returns a python dictionary with the output data.

        optional inputs:
        num_profile_jacob: an integer, passed into the PCRTM forward, that limits
        the profile jacobian calculations. Right now this will just cause the channel
        expansion of the jacobians to be skipped. By default this will have a value of 7,
        meaning retain the temperature profile and all 6 molecular concentration jacobians.
        Setting this to zero will mean that no profile jacobians are computed.

        All dictionary values will contain numpy ndarray with the specified
        shapes, all with dtype numpy.float32.  All values are output, but note
        that some of them will contain uninitialized data depending on the values
        of the various control flags set when initializing the model.
        (see the init() function.)


        Radiance units are [mW/(m^2 sr cm^-1)], R.U. below, for short.

        The shapes are:

        E = number of EOFs

        K = number of atmospheric levels

        L = number of atmospheric layers (= K-1)

        M = number of molecular species (always 6)

        Vm = number of monochromatic frequencies

        Vc = number of channel frequencies


        Returns
        -------

        eof : the EOF coefficients (E,)

        mono_wn : the monochromatic wavenumbers (Vm,) [1/cm]

        mono_rad : the monochromatic radiance (Vm,), [RU]

        wn : the channel center wavenumbers (Vc,) [1/cm]

        rad : the channel radiance (Vc,) [RU]

        bt : the channel brightness temperatures [K]

        layer_trans : the per-layer channel transmission (Vc,L) [unitless]

        krad_tsurf : surface temperature jacobian, (Vc,) [RU / K]

        krad_esurf : surface emissivity jacobian, (Vc,) [RU / unit emissivity]

        krad_t : the temperature profile jacobian, (Vc,K) [RU / K]

        krad_mol : the molecular profile jacobian, (Vc,K,M) [RU / concentration]

        kbt_tsurf : surface temperature jacobian, (Vc,) [K / K]

        kbt_esurf : surface emissivity jacobian, (Vc,) [K / unit emissivity]

        kbt_t : the temperature profile jacobian, (Vc,K) [K / K]

        kbt_mol : the molecular profile jacobian, (Vc,K,M) [K / concentration]

        """
        neof = self.num_eof
        nmono = self.num_monofreq
        nchannel = self.num_channels
        nlevel = self.num_levels
        nlayer = self.num_layers

        nmol = self._nmol

        if num_profile_jacob is None:
            num_profile_jacob = self._nmol + 1

        eof = np.empty(neof, np.float32)
        mono_wn = np.empty(nmono, np.float64)
        mono_rad = np.empty(nmono, np.float32)
        wn = np.empty(nchannel, np.float64)
        rad = np.empty(nchannel, np.float32)
        bt = np.empty(nchannel, np.float32)

        layer_trans = np.empty((nchannel, nlayer), np.float32)

        krad_tsurf = np.empty(nchannel, np.float32)
        krad_esurf = np.empty(nchannel, np.float32)
        krad_t = np.empty((nchannel, nlevel), np.float32)
        krad_mol = np.empty((nchannel, nlevel, nmol), np.float32)

        kbt_tsurf = np.empty(nchannel, np.float32)
        kbt_esurf = np.empty(nchannel, np.float32)
        kbt_t = np.empty((nchannel, nlevel), np.float32)
        kbt_mol = np.empty((nchannel, nlevel, nmol), np.float32)


        self._m.pcrtmw_forward_rt(
            c_int(num_profile_jacob),
            eof.ctypes.data_as(POINTER(c_float)),
            mono_wn.ctypes.data_as(POINTER(c_double)),
            mono_rad.ctypes.data_as(POINTER(c_float)),
            wn.ctypes.data_as(POINTER(c_double)),
            rad.ctypes.data_as(POINTER(c_float)),
            bt.ctypes.data_as(POINTER(c_float)),
            layer_trans.ctypes.data_as(POINTER(c_float)),
            krad_tsurf.ctypes.data_as(POINTER(c_float)),
            krad_esurf.ctypes.data_as(POINTER(c_float)),
            krad_t.ctypes.data_as(POINTER(c_float)),
            krad_mol.ctypes.data_as(POINTER(c_float)),
            kbt_tsurf.ctypes.data_as(POINTER(c_float)),
            kbt_esurf.ctypes.data_as(POINTER(c_float)),
            kbt_t.ctypes.data_as(POINTER(c_float)),
            kbt_mol.ctypes.data_as(POINTER(c_float)),
            c_int(neof), c_int(nmono), c_int(nchannel),
            c_int(nlevel), c_int(nlayer), c_int(nmol))

        dat = {}

        dat['eof'] = eof
        dat['mono_wn'] = mono_wn
        dat['mono_rad'] = mono_rad

        dat['wn'] = wn
        dat['rad'] = rad
        dat['bt'] = bt

        dat['layer_trans'] = np.reshape(layer_trans, (nlayer, nchannel)).T

        dat['krad_tsurf'] = krad_tsurf
        dat['krad_esurf'] = krad_esurf
        dat['krad_t'] = np.reshape(krad_t, (nlevel, nchannel)).T
        dat['krad_mol'] = np.transpose(
            krad_mol.reshape((nmol,nlevel,nchannel)), (2,1,0))

        dat['kbt_tsurf'] = kbt_tsurf
        dat['kbt_esurf'] = kbt_esurf
        dat['kbt_t'] = np.reshape(kbt_t, (nlevel, nchannel)).T
        dat['kbt_mol'] = np.transpose(
            kbt_mol.reshape((nmol,nlevel,nchannel)), (2,1,0))

        return dat


class PCRTM_SOLAR(PCRTM):
    """
    singleton class wrapper for the PCRTM fortran90 library

    same as PCRTM, but this works with the PCRTM_SOLAR code

    certain methods/properties, that have aspects particular to PCRTM_SOLAR,
    are overwritten with new methods.
    Otherwise, most of the methods are the same as PCRTM, and we get them
    via subclassing PCRTM so it is (mostly) the same code.

    """
    def __new__(cls):

        model_version = 'SOLAR_V4.2'
        object_file = 'PCRTM_SOLAR_wrapper_module.so'

        # note that we call the original base class __new__
        # here, not PCRTM (we are calling the superclass of the superclass)
        instance = super(PCRTM, cls).__new__(cls, object_file, model_version)

        return instance

    def __init__(self):
        self._nmol = 15
 
    def init(self,
             sensor_id,
             output_jacob_flag=False,
             output_tr_flag=False,
             output_ch_flag=True,
             output_jacob_ch_flag=False,
             output_bt_flag=True,
             output_jacob_bt_flag=False,
             molindx = None,
             scalfac = None):
        """
        initialize PCRTM_SOLAR module with a specific sensor_id.
        The various output* keywords control whether those data are computed
        and stored in the output data arrays after a call to forward.

        Parameters
        __________

        sensor_id : int
            the id number for the desired sensor. From the PCRTM documentation:
            currently only 26 is possible (IASI full spectra)
        
        output_jacob_flag : bool
            controls output of the jacobian

        output_tr_flag : bool
            controls output of the transmittance (current unimplemented)

        output_ch_flag : bool
            controls whether output arrays are generated for the sensor channels
            By default, output would only be generated for the EOF
            coefficients and the monochromatic radiances.

        output_jacob_ch_flag : bool
            controls whether jacobian outputs are generated for the sensor
            channels.

        output_bt_flag : bool
            controls whether the brightness temperature is computed in addition
            to the radiance.

        output_jacob_bt_flag : bool
            controls whether jacobian outputs are computed in brightness temperature
            in addition to the radiance.

        molindx : list or ndarray
            15 element integer array (see PCRTM documentation), controlling the
            treatment of the molecular species (in order: H2O, CO2, O3, N2O, CO, CH4).

            molindx = 0 profile is fixed to the US Std profile value, and is
            treated as "fixed" (I think this means, no jacobian is computed).

            molindx = 1, profile is fixed to US Std, treated as "variable".

            molindx = 2, profile is variable and must be input (the default).

            This is untested for any but the default value.

        scalfac : list or ndarray
            15 element floating point array with scale factors to apply to each
            gas profile. Default is 1.0 for each (equivalent to no scaling.)

        """

        # we need to intercept the init method in order to handle
        # the special case for the fixed values of the molindx

        if molindx is None:
            molindx = np.zeros(15, dtype=np.int32)
            molindx[:6] = 2
        if scalfac is None:
            scalfac = np.zeros(15, dtype=np.float32)
            scalfac[:] = 1.0

        PCRTM.init(self, sensor_id,
                   output_jacob_flag=output_jacob_flag,
                   output_tr_flag=output_tr_flag,
                   output_ch_flag=output_ch_flag,
                   output_jacob_ch_flag=output_jacob_ch_flag,
                   output_bt_flag=output_bt_flag,
                   output_jacob_bt_flag=output_jacob_bt_flag,
                   molindx = molindx,
                   scalfac = scalfac)

    # remaining new functions implement the modified geometry inputs,
    # which now include sensor azimuth and both solar angles.
    def _get_geom(self):
        geom = np.zeros(6, dtype=np.float32)
        self._m.pcrtmw_get_geom(geom.ctypes.data_as(POINTER(c_float)))
        return geom
    def _set_geom(self, psfc, pobs, sensor_zen, sensor_azi, solar_zen, solar_azi):
        var_list = [psfc, pobs, sensor_zen, sensor_azi, solar_zen, solar_azi]
        geom = np.array(var_list).astype(np.float32)
        self._m.pcrtmw_set_geom(geom.ctypes.data_as(POINTER(c_float)))

    def _get_psfc(self):
        geom = self._get_geom()
        return geom[0]
    def _set_psfc(self,psfc):
        geom = self._get_geom()
        self._set_geom(psfc,    geom[1], geom[2],
                       geom[3], geom[4], geom[5])

    def _get_pobs(self):
        geom = self._get_geom()
        return geom[1]
    def _set_pobs(self,pobs):
        geom = self._get_geom()
        self._set_geom(geom[0], pobs,    geom[2],
                       geom[3], geom[4], geom[5])

    def _get_sensor_zen(self):
        geom = self._get_geom()
        return geom[2]
    def _set_sensor_zen(self,sensor_zen):
        geom = self._get_geom()
        self._set_geom(geom[0], geom[1], sensor_zen,
                       geom[3], geom[4], geom[5])

    def _get_sensor_azim(self):
        geom = self._get_geom()
        return geom[3]
    def _set_sensor_azim(self,sensor_azim):
        geom = self._get_geom()
        self._set_geom(geom[0],     geom[1], geom[2],
                       sensor_azim, geom[4], geom[5])

    def _get_solar_zen(self):
        geom = self._get_geom()
        return geom[4]
    def _set_solar_zen(self,solar_zen):
        geom = self._get_geom()
        self._set_geom(geom[0], geom[1],   geom[2],
                       geom[3], solar_zen, geom[5])

    def _get_solar_azim(self):
        geom = self._get_geom()
        return geom[5]
    def _set_solar_azim(self,solar_azim):
        geom = self._get_geom()
        self._set_geom(geom[0], geom[1], geom[2],
                       geom[3], geom[4], solar_azim)

    psfc = property(
        _get_psfc, _set_psfc,
        doc="get or set surface pressure [hPa]")
    pobs = property(
        _get_pobs, _set_pobs,
        doc="get or set pressure level of observer [hPa]\n"+
        "0.005 is the TOA and equivalent to a satellite observer")

    sensor_zen = property(
        _get_sensor_zen, _set_sensor_zen,
        doc="get or set sensor zenith angle [deg]")
    sensor_azim = property(
        _get_sensor_azim, _set_sensor_azim,
        doc="get or set sensor azimuth angle [deg]")
    solar_zen = property(
        _get_solar_zen, _set_solar_zen,
        doc="get or set solar zenith angle [deg]")
    solar_azim = property(
        _get_solar_azim, _set_solar_azim,
        doc="get or set solar azimuth angle [deg]")
