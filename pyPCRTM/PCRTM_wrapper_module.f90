!
! PCRTM_wrapper_module:
!
! a wrapper module that applies various C bindings to the functions defined
! in the PCRTM static library.
! this also implements instances of the user types in PCRTM that define
! the atmosphere profile, etc, and stores these internally.
! This allows the python module to get/set these data, in a fashion that is
! similar to a "normal" python object.
!

module pcrtm_wrapper_module

  use iso_c_binding, only: c_float, c_double, c_int, c_long, c_bool

  use INIT_PCRTM
  use CLEAR_PCRTM
  use PCRTM_FORWARD_MODEL

  implicit none

  public :: pcrtmw_init
  public :: pcrtmw_get_init_status
  public :: pcrtmw_get_run_status

  public :: pcrtmw_set_geom,  pcrtmw_get_geom
  public :: pcrtmw_set_tlev,  pcrtmw_get_tlev
  public :: pcrtmw_set_tskin, pcrtmw_get_tskin
  public :: pcrtmw_set_vmr,   pcrtmw_get_vmr
  public :: pcrtmw_set_emis,  pcrtmw_get_emis
  public :: pcrtmw_set_emis_scalar

  public :: pcrtmw_set_cld,   pcrtmw_get_cld
  public :: pcrtmw_set_cldP,  pcrtmw_get_cldP
  public :: pcrtmw_set_cldDe, pcrtmw_get_cldDe
  public :: pcrtmw_set_cldOD, pcrtmw_get_cldOD

  public :: pcrtmw_get_nlevels
  public :: pcrtmw_get_nchannels
  public :: pcrtmw_get_neof
  public :: pcrtmw_get_nmonofreq
  public :: pcrtmw_get_nbands

  public :: pcrtmw_get_nchannels_per_band
  public :: pcrtmw_get_neof_per_band
  public :: pcrtmw_get_nmonofreq_per_band

  public :: pcrtmw_get_monofreq

  public :: pcrtmw_forward_rt

  ! unknown at this point whether we need to include the
  ! various CLEAR_PCRTM functions.
  ! perhaps, it would be wise to clear the JACOBIAN, RT_solution
  ! types, after a forward run, but leave the ATM, GEOM always
  ! allocated?

  ! various user_types from PCRTM
  private ATM, GEOMETRY
  private PCRTM_STND, ATM_ABS_COEF, ICE_GRID, WAT_GRID
  private TR_SOLUTION, EOF_SOLUTION, RT_SOLUTION
  private K_M, K_CH, K_PC
  
  ! input control variables
  private SENSOR_ID
  private MOLINDX, SCALFAC
  private OUTPUT_JACOB_FLAG, OUTPUT_TR_FLAG
  private OUTPUT_CH_FLAG, OUTPUT_JACOB_CH_FLAG
  private OUTPUT_BT_FLAG, OUTPUT_JACOB_BT_FLAG

  ! state variables
  private INIT_STATUS
  private RUN_STATUS

  integer, parameter :: n_absmolsp = 6  ! Number of absorber molecular species

  TYPE(PCRTM_ATMOSPHERE_TYPE) :: ATM
  TYPE(PCRTM_GEOMETRY_TYPE)   :: GEOMETRY

  TYPE(PCRTM_ATM_ABS_STRUCT_TYPE)   :: PCRTM_STND
  TYPE(PCRTM_ATM_ABSORPTION_TYPE)   :: ATM_ABS_COEF
  TYPE(PCRTM_CLD_TABLE_DEF)         :: ICE_GRID
  TYPE(PCRTM_CLD_TABLE_DEF)         :: WAT_GRID

  TYPE(PCRTM_TR_SOLUTION_TYPE),  ALLOCATABLE :: TR_SOLUTION(:) 
  TYPE(PCRTM_EOF_SOLUTION_TYPE), ALLOCATABLE :: EOF_SOLUTION(:) 
  TYPE(PCRTM_RT_SOLUTION_TYPE)               :: RT_SOLUTION

  TYPE(PCRTM_NM_JACOBIAN_TYPE)              :: K_M
  TYPE(PCRTM_CH_JACOBIAN_TYPE), ALLOCATABLE :: K_CH(:)
  TYPE(PCRTM_PC_JACOBIAN_TYPE), ALLOCATABLE :: K_PC(:)

  INTEGER                     :: SENSOR_ID
  INTEGER                     :: MOLINDX(n_absmolsp)
  integer, dimension(n_absmolsp) :: SCALFAC
  LOGICAL                     :: OUTPUT_JACOB_FLAG
  LOGICAL                     :: OUTPUT_TR_FLAG
  LOGICAL                     :: OUTPUT_CH_FLAG
  LOGICAL                     :: OUTPUT_JACOB_CH_FLAG
  LOGICAL                     :: OUTPUT_BT_FLAG
  LOGICAL                     :: OUTPUT_JACOB_BT_FLAG

  LOGICAL :: INIT_STATUS, RUN_STATUS

contains

  subroutine pcrtmw_start() bind(c, name='pcrtmw_start')
    INIT_STATUS = .FALSE.
    RUN_STATUS = .FALSE.
  end subroutine pcrtmw_start

  function pcrtmw_get_init_status()  result(answer)  &
       bind(c, name='pcrtmw_get_init_status')
    logical(c_bool) :: answer
    answer = INIT_STATUS
  end function pcrtmw_get_init_status

  function pcrtmw_get_run_status()  result(answer)  &
       bind(c, name='pcrtmw_get_run_status')
    logical(c_bool) :: answer
    answer = RUN_STATUS
  end function pcrtmw_get_run_status

  ! implementation note: I have made the input variable names (dummy
  ! arguments, basically) have trailing "_in" to distinguish between the
  ! c_* bound inputs and the "plain" fortran types that are stored in the
  ! module. The thought here was to be less susceptable to type mismatch:
  ! for example the line SENSOR_ID = sensor_id_in, should allow the integer
  ! type to be converted at runtime.

  ! TBD: Fix seg fault from ipython tab complete - i think, prevent things
  ! from happening if not yet INIT.
  subroutine pcrtmw_init( sensor_id_in, &
       output_jacob_flag_in, output_tr_flag_in, &
       output_ch_flag_in, output_jacob_ch_flag_in, &
       output_bt_flag_in, output_jacob_bt_flag_in, &
       molindx_in, scalfac_in) &
       bind(c, name='pcrtmw_init')

    integer(c_int), intent(in), value :: sensor_id_in
    integer(c_int), intent(in), dimension(n_absmolsp) :: molindx_in
    integer(c_int), intent(in), dimension(n_absmolsp) :: scalfac_in
    logical(c_bool), intent(in), value :: output_jacob_flag_in
    logical(c_bool), intent(in), value :: output_tr_flag_in
    logical(c_bool), intent(in), value :: output_ch_flag_in
    logical(c_bool), intent(in), value :: output_jacob_ch_flag_in
    logical(c_bool), intent(in), value :: output_bt_flag_in
    logical(c_bool), intent(in), value :: output_jacob_bt_flag_in

    character(len=120) :: INPUTDIR

    ! copy input vars to internal fortran vars.
    SENSOR_ID = sensor_id_in
    OUTPUT_JACOB_FLAG = output_jacob_flag_in
    OUTPUT_TR_FLAG = output_tr_flag_in
    OUTPUT_CH_FLAG = output_ch_flag_in
    OUTPUT_JACOB_CH_FLAG = output_jacob_ch_flag_in
    OUTPUT_BT_FLAG = output_bt_flag_in
    OUTPUT_JACOB_BT_FLAG = output_jacob_bt_flag_in

    MOLINDX = molindx_in
    SCALFAC = scalfac_in

    ! get from env var
    call get_environment_variable('PCRTM_INPUT_DIR', INPUTDIR)

    if (len(trim(INPUTDIR)) .EQ. 0) then
       INPUTDIR = 'INPUTDIR/'
    end if

    call PCRTM_INIT(           &
         INPUTDIR,             &
         SENSOR_ID,            &
         OUTPUT_JACOB_FLAG,    &
         OUTPUT_CH_FLAG,       &
         OUTPUT_BT_FLAG,       &
         OUTPUT_JACOB_CH_FLAG, &
         OUTPUT_JACOB_BT_FLAG, &
         OUTPUT_TR_FLAG,       &
         PCRTM_STND,           &
         ATM_ABS_COEF,         &
         ICE_GRID,             &
         WAT_GRID,             &
         EOF_SOLUTION,         &
         ATM,                  &
         RT_SOLUTION,          &
         K_M,                  &
         K_PC,                 &
         K_CH,                 &
         TR_SOLUTION)

    PCRTM_STND%MOLINDX(1:n_absmolsp) = MOLINDX(1:n_absmolsp)
    PCRTM_STND%SCALFAC(1:n_absmolsp) = SCALFAC(1:n_absmolsp)

    ! init these to zero (enforces a clear sky setup.)
    ATM%CLD_FLAG = 0
    ATM%PCLD     = 0
    ATM%TAUCLD   = 0
    ATM%DECLD    = 0

    INIT_STATUS = .TRUE.

  end subroutine pcrtmw_init

  subroutine pcrtmw_get_geom(geom) bind(c, name = 'pcrtmw_get_geom')
    real(c_float), intent(inout), dimension(3) :: geom
    ! line 15, PCRTM_ATMOSPHERE_LAYER.f90; note only the *'ed variables
    ! are user-settable, to my knowledge: the rest are internally computed
    ! at some point.
    !
    !  TYPE PCRTM_GEOMETRY_TYPE
    !     REAL(SINGLE) :: PSFC * surface P in hPa
    !     REAL(SINGLE) :: POBS * observer P in hPa (0.005 if TOA.)
    !     INTEGER      :: LSFC
    !     INTEGER      :: LOBS
    !     INTEGER      :: NTOP
    !     INTEGER      :: NBOT
    !     REAL(SINGLE) :: SCALSFC
    !     REAL(SINGLE) :: SCALOBS
    !     REAL(SINGLE) :: SECZANG  
    !     REAL(SINGLE) :: SATANG * view zenith in deg   
    !  END TYPE PCRTM_GEOMETRY_TYPE

    geom(1) = GEOMETRY%PSFC
    geom(2) = GEOMETRY%POBS
    geom(3) = GEOMETRY%SATANG

  end subroutine pcrtmw_get_geom

  subroutine pcrtmw_set_geom(geom) bind(c, name = 'pcrtmw_set_geom')
    real(c_float), intent(in), dimension(3) :: geom
    GEOMETRY%PSFC = geom(1)
    GEOMETRY%POBS = geom(2)
    GEOMETRY%SATANG = geom(3)
  end subroutine pcrtmw_set_geom


  ! ATM structure: line 14 PCRTM_ATMOSPHERE_DEFINE.f90
  ! I think, user-set data should be the levels; layer values are
  ! computed internally.
  !
  !  TYPE PCRTM_ATMOSPHERE_TYPE
  !     REAL(SINGLE), ALLOCATABLE  :: TLEV(:)          ! TLEVELS(NLEV)
  !     REAL(SINGLE), ALLOCATABLE  :: VMR(:,:)         ! VMR(NLEV,NMOL)     
  !     REAL(SINGLE), ALLOCATABLE  :: TLAY(:)          ! TLAY(NLAY)
  !     REAL(SINGLE), ALLOCATABLE  :: GASPROF(:,:)     ! GASPROF(NLAY,NMOL)
  !     REAL(SINGLE), ALLOCATABLE  :: AIRAMT(:)        ! AIRAMT(NLAY)
  !     REAL(SINGLE), ALLOCATABLE  :: DLAYDLEV(:)      ! DLAYDLEV(NLEV)
  !     INTEGER,      ALLOCATABLE  :: CLD_FLAG(:)      ! CLOUD FLAG(NLEV)
  !     REAL(SINGLE), ALLOCATABLE  :: PCLD(:)          ! CLOUD PARAM(NLEV)
  !     REAL(SINGLE), ALLOCATABLE  :: TAUCLD(:)        ! CLOUD tau(NLEV)
  !     REAL(SINGLE), ALLOCATABLE  :: DECLD(:)         ! CLOUD d_eff(NLEV)
  !     REAL(SINGLE)               :: TSKIN            ! skin temp
  !  END TYPE PCRTM_ATMOSPHERE_TYPE
  subroutine pcrtmw_set_tlev(tlev) bind(c, name = 'pcrtmw_set_tlev')
    real(c_float), intent(in), dimension(101) :: tlev
    ATM%TLEV(1:101) = tlev
  end subroutine pcrtmw_set_tlev

  subroutine pcrtmw_set_tskin(tskin) bind(c, name = 'pcrtmw_set_tskin')
    real(c_float), intent(in), value :: tskin
    ATM%TSKIN = tskin
  end subroutine pcrtmw_set_tskin

  subroutine pcrtmw_set_vmr(vmr, mol_id) bind(c, name = 'pcrtmw_set_vmr')
    real(c_float), intent(in), dimension(101) :: vmr
    integer(c_int), intent(in), value :: mol_id
    ATM%VMR(1:101, mol_id) = vmr
  end subroutine pcrtmw_set_vmr

  subroutine pcrtmw_set_emis(emis, nfreq) bind(c, name='pcrtmw_set_emis')
    real(c_float), intent(in), dimension(nfreq) :: emis
    integer(c_int), intent(in), value :: nfreq
    integer :: w
    do w = 1, nfreq
       RT_SOLUTION%EMIS(w) = emis(w)
    end do
  end subroutine pcrtmw_set_emis

  subroutine pcrtmw_set_emis_scalar(emis) bind(c, name='pcrtmw_set_emis_scalar')
    real(c_float), intent(in), value :: emis
    RT_SOLUTION%EMIS = emis
  end subroutine pcrtmw_set_emis_scalar

  subroutine pcrtmw_set_cld(cld_flag) bind(c, name = 'pcrtmw_set_cld')
    integer(c_int), intent(in), dimension(100) :: cld_flag
    ATM%CLD_FLAG(1:100) = cld_flag
  end subroutine pcrtmw_set_cld

  subroutine pcrtmw_set_cldP(cldP) bind(c, name = 'pcrtmw_set_cldP')
    real(c_float), intent(in), dimension(100) :: cldP
    ATM%PCLD(1:100) = cldP
  end subroutine pcrtmw_set_cldP

  subroutine pcrtmw_set_cldDe(cldDe) bind(c, name = 'pcrtmw_set_cldDe')
    real(c_float), intent(in), dimension(100) :: cldDe
    ATM%DECLD(1:100) = cldDe
  end subroutine pcrtmw_set_cldDe

  subroutine pcrtmw_set_cldOD(cldOD) bind(c, name = 'pcrtmw_set_cldOD')
    real(c_float), intent(in), dimension(100) :: cldOD
    ATM%TAUCLD(1:100) = cldOD
  end subroutine pcrtmw_set_cldOD

  subroutine pcrtmw_get_tlev(tlev) bind(c, name = 'pcrtmw_get_tlev')
    real(c_float), intent(inout), dimension(101) :: tlev
    tlev = ATM%TLEV(1:101)
  end subroutine pcrtmw_get_tlev

  function pcrtmw_get_tskin()  result(answer)  &
       bind(c, name = 'pcrtmw_get_tskin')
    real(c_float) :: answer
    answer = ATM%TSKIN
  end function pcrtmw_get_tskin

  subroutine pcrtmw_get_vmr(vmr, mol_id) bind(c, name = 'pcrtmw_get_vmr')
    real(c_float), intent(inout), dimension(101) :: vmr
    integer(c_int), intent(in), value :: mol_id
    vmr = ATM%VMR(1:101, mol_id)
  end subroutine pcrtmw_get_vmr

  subroutine pcrtmw_get_emis(emis, nfreq) bind(c, name='pcrtmw_get_emis')
    real(c_float), intent(inout), dimension(nfreq) :: emis
    integer(c_int), intent(in), value :: nfreq
    integer :: w
    do w = 1, nfreq
       emis(w) = RT_SOLUTION%EMIS(w)
    end do
  end subroutine pcrtmw_get_emis

  subroutine pcrtmw_get_cld(cld_flag) bind(c, name = 'pcrtmw_get_cld')
    integer(c_int), intent(inout), dimension(100) :: cld_flag
    cld_flag = ATM%CLD_FLAG(1:100)
  end subroutine pcrtmw_get_cld

  subroutine pcrtmw_get_cldP(cldP) bind(c, name = 'pcrtmw_get_cldP')
    real(c_float), intent(inout), dimension(100) :: cldP
    cldP = ATM%PCLD(1:100)
  end subroutine pcrtmw_get_cldP

  subroutine pcrtmw_get_cldDe(cldDe) bind(c, name = 'pcrtmw_get_cldDe')
    real(c_float), intent(inout), dimension(100) :: cldDe
    cldDe = ATM%DECLD(1:100)
  end subroutine pcrtmw_get_cldDe

  subroutine pcrtmw_get_cldOD(cldOD) bind(c, name = 'pcrtmw_get_cldOD')
    real(c_float), intent(inout), dimension(100) :: cldOD
    cldOD = ATM%TAUCLD(1:100)
  end subroutine pcrtmw_get_cldOD

  ! utility functions to get array shapes: note these can only run
  ! after the PCRTM INIT was run, since the shapes are basically set
  ! by the array sizes of the coefficient files.
  function pcrtmw_get_nlevels()  result(answer)  &
       bind(c, name = 'pcrtmw_get_nlevels')
    integer(c_int) :: answer
    if (INIT_STATUS) then
       answer = PCRTM_STND%NLEV
    else
       answer = 0
    end if
  end function pcrtmw_get_nlevels

  function pcrtmw_get_nlayers()  result(answer)  &
       bind(c, name = 'pcrtmw_get_nlayers')
    integer(c_int) :: answer
    if (INIT_STATUS) then
       answer = PCRTM_STND%NLAY
    else
       answer = 0
    end if
  end function pcrtmw_get_nlayers

  function pcrtmw_get_nchannels()  result(answer)  &
       bind(c, name = 'pcrtmw_get_nchannels')
    integer(c_int) :: answer
    integer :: b
    answer = 0
    if (INIT_STATUS .and. OUTPUT_CH_FLAG) then
       do b = 1, SIZE(EOF_SOLUTION)
          answer = answer + EOF_SOLUTION(b)%NCHBND
       end do
    end if
  end function pcrtmw_get_nchannels

  function pcrtmw_get_neof()  result(answer)  &
       bind(c, name = 'pcrtmw_get_neof')
    integer(c_int) :: answer
    integer :: b
    answer = 0
    if (INIT_STATUS) then
       do b = 1, SIZE(EOF_SOLUTION)
          answer = answer + EOF_SOLUTION(b)%NPCBND
       end do
    end if
  end function pcrtmw_get_neof

  function pcrtmw_get_nmonofreq()  result(answer)  &
       bind(c, name = 'pcrtmw_get_nmonofreq')
    integer(c_int) :: answer
    integer :: b
    answer = 0
    if (INIT_STATUS) then
       do b = 1, SIZE(EOF_SOLUTION)
          answer = answer + EOF_SOLUTION(b)%NREG
       end do
    end if
  end function pcrtmw_get_nmonofreq

  function pcrtmw_get_nbands()  result(answer)  &
       bind(c, name = 'pcrtmw_get_nbands')
    integer(c_int) :: answer
    answer = SIZE(EOF_SOLUTION)
  end function pcrtmw_get_nbands

  subroutine pcrtmw_get_nchannels_per_band(nchannels, nband) &
       bind(c, name='pcrtmw_get_nchannels_per_band')
    integer(c_int), intent(out) :: nchannels(nband)
    integer(c_int), intent(in), value :: nband
    integer :: b
    if (INIT_STATUS) then
       do b = 1, SIZE(EOF_SOLUTION)
          nchannels(b) = EOF_SOLUTION(b)%NCHBND
       end do
    end if
  end subroutine pcrtmw_get_nchannels_per_band

  subroutine pcrtmw_get_neof_per_band(neof, nband) &
       bind(c, name='pcrtmw_get_neof_per_band')
    integer(c_int), intent(out) :: neof(nband)
    integer(c_int), intent(in), value :: nband
    integer :: b
    if (INIT_STATUS) then
       do b = 1, SIZE(EOF_SOLUTION)
          neof(b) = EOF_SOLUTION(b)%NPCBND
       end do
    end if
  end subroutine pcrtmw_get_neof_per_band

  subroutine pcrtmw_get_nmonofreq_per_band(nmonofreq, nband) &
       bind(c, name='pcrtmw_get_nmonofreq_per_band')
    integer(c_int), intent(out) :: nmonofreq(nband)
    integer(c_int), intent(in), value :: nband
    integer :: b
    if (INIT_STATUS) then
       do b = 1, SIZE(EOF_SOLUTION)
          nmonofreq(b) = EOF_SOLUTION(b)%NREG
       end do
    end if
  end subroutine pcrtmw_get_nmonofreq_per_band

  subroutine pcrtmw_get_monofreq(monofreq, nmono) &
       bind(c, name='pcrtmw_get_monofreq')
    real(c_double), intent(out) :: monofreq(nmono)
    integer(c_int), intent(in), value :: nmono
    integer :: c
    if (INIT_STATUS) then
       do c = 1, PCRTM_STND%NM
          monofreq(c) = PCRTM_STND%FRQ(c)
       end do
    end if
  end subroutine pcrtmw_get_monofreq

  ! do the work: forward run.
  subroutine pcrtmw_forward_rt( &
       num_profile_jacob, &
       eof, mono_wn, mono_rad, wn, rad, bt, layer_trans, &
       krad_tsurf, krad_esurf, krad_t, krad_mol, &
       kbt_tsurf, kbt_esurf, kbt_t, kbt_mol, &
       neof, nmono, nchannel, nlevel, nlayer, nmol) &
       bind(c, name='pcrtmw_forward_rt')

    integer(c_int), intent(in), value  :: num_profile_jacob

    real(c_float),  intent(out) :: eof(neof)
    real(c_double), intent(out) :: mono_wn(nmono)
    real(c_float),  intent(out) :: mono_rad(nmono)
    real(c_double), intent(out) :: wn(nchannel)
    real(c_float),  intent(out) :: rad(nchannel)
    real(c_float),  intent(out) :: bt(nchannel)
    real(c_float),  intent(out) :: layer_trans(nchannel,nlayer)
    real(c_float),  intent(out) :: krad_tsurf(nchannel)
    real(c_float),  intent(out) :: krad_esurf(nchannel)
    real(c_float),  intent(out) :: krad_t(nchannel,nlevel)
    real(c_float),  intent(out) :: krad_mol(nchannel,nlevel,nmol)
    real(c_float),  intent(out) :: kbt_tsurf(nchannel)
    real(c_float),  intent(out) :: kbt_esurf(nchannel)
    real(c_float),  intent(out) :: kbt_t(nchannel,nlevel)
    real(c_float),  intent(out) :: kbt_mol(nchannel,nlevel,nmol)

    integer(c_int), intent(in), value :: neof
    integer(c_int), intent(in), value :: nmono
    integer(c_int), intent(in), value :: nchannel
    integer(c_int), intent(in), value :: nlevel
    integer(c_int), intent(in), value :: nlayer
    integer(c_int), intent(in), value :: nmol

    integer :: b, c, w, m

    call PCRTM_FORWARD_RT( &
         num_profile_jacob,       &
         GEOMETRY,                &
         ATM,                     &
         ATM_ABS_COEF,            &
         PCRTM_STND,              &
         ICE_GRID,                &
         WAT_GRID,                &
         RT_SOLUTION,             &
         EOF_SOLUTION,            &
         K_M,                     &
         K_PC,                    &
         K_CH,                    &
         TR_SOLUTION )
    
    w = 1
    do b = 1, SIZE(EOF_SOLUTION)
       do c = 1, EOF_SOLUTION(b)%NPCBND
          eof(w) = EOF_SOLUTION(b)%RADPC(c)
          w = w + 1
       end do
    end do

    do c = 1, PCRTM_STND%NM
       mono_wn(c) = PCRTM_STND%FRQ(c)
       mono_rad(c) = RT_SOLUTION%RADUP(c)
    end do
    
    w = 1
    do b = 1, SIZE(EOF_SOLUTION)
       do c = 1, EOF_SOLUTION(b)%NCHBND

          if (OUTPUT_CH_FLAG) then
             wn(w) = EOF_SOLUTION(b)%FRQCH(c)
             rad(w)= EOF_SOLUTION(b)%RADCH(c)
             if (OUTPUT_BT_FLAG) then
                bt(w) = EOF_SOLUTION(b)%BTCH(c)
             end if
          end if

          if (OUTPUT_JACOB_CH_FLAG) then
             krad_tsurf(w) = K_CH(b)%R_TS(c)
             krad_esurf(w) = K_CH(b)%R_EM(c)
             krad_t(w, :) = K_CH(b)%R_TLEV(c, :)
             do m = 1, nmol
                krad_mol(w,:,m) = K_CH(b)%R_GASLEV(c, :, m)
             end do
          end if

          if (OUTPUT_JACOB_BT_FLAG) then
             kbt_tsurf(w) = K_CH(b)%BT_TS(c)
             kbt_esurf(w) = K_CH(b)%BT_EM(c)
             kbt_t(w, :) = K_CH(b)%BT_TLEV(c, :)
             do m = 1, nmol
                kbt_mol(w,:,m) = K_CH(b)%BT_GASLEV(c, :, m)
             end do
          end if

          if (OUTPUT_TR_FLAG) then
             layer_trans(w, :) = TR_SOLUTION(b)%TRLAY_CH(c, :)
          end if

          w = w + 1

       end do
    end do    

    RUN_STATUS = .TRUE.

  end subroutine pcrtmw_forward_rt

end module pcrtm_wrapper_module
