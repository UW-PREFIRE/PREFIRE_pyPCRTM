import os
import numpy as np
from datetime import datetime

from test_setups import setup_PCRTM

def basic_run_kw():
    setup_kws = dict(
        sensor_id = 2,
        output_jacob_flag=False,
        output_tr_flag=False,
        output_ch_flag=False,
        output_jacob_ch_flag=False,
        output_bt_flag=False,
        output_jacob_bt_flag=False)
    return setup_kws

def run_timing_test(F, num_iter, num_profile_jacob=7):
    t1 = datetime.now()
    tskin = F.tskin
    for n in range(num_iter):
        F.tskin = tskin + 2.0*n/num_iter
        dat = F.forward_rt(num_profile_jacob=num_profile_jacob)
    timing_average = (datetime.now()-t1) / num_iter
    return timing_average

def run_test_sets(num_iter, test_cloudy_jacobians=False):

    timings = []
    descriptions = []

    # cloud parameters for some tests
    k = 90
    cld = np.zeros(100, np.int32)
    cldDe = np.zeros(100, np.float32)
    cldOD = np.zeros(100, np.float32)
    cldP = np.zeros(100, np.float32)
    cld[k] = 2
    cldDe[k] = 30.0
    cldOD[k] = 1.0
    cldP[k] = 800.0

    ######################## 1 - 3 Clear sky forward
    kws = basic_run_kw()
    
    print('#### (1) Minimal run: Mono-v only')
    descriptions.append('Forward,  clear')
    F = setup_PCRTM(**kws)
    timings.append(run_timing_test(F, num_iter))

    print('#### (2) Include channels')
    descriptions.append('Forward,  clear, channels')
    kws['output_ch_flag'] = True
    F = setup_PCRTM(**kws)
    timings.append(run_timing_test(F, num_iter))

    print('#### (3) Include channels and BT')
    descriptions.append('Forward,  clear, channels, BT')
    kws['output_bt_flag'] = True
    F = setup_PCRTM(**kws)
    timings.append(run_timing_test(F, num_iter))

    ######################## 4 - 6 Clear sky Jacobian

    #### Repeat 1-3, with Jacobian
    kws = basic_run_kw()
    kws['output_jacob_flag'] = True

    print('#### (4) Minimal run: Mono-v only')
    descriptions.append('Jacobian, clear')
    F = setup_PCRTM(**kws)
    timings.append(run_timing_test(F, num_iter))

    print('#### (5) Include channels')
    descriptions.append('Jacobian, clear, channels')
    kws['output_jacob_ch_flag'] = True
    kws['output_ch_flag'] = True
    F = setup_PCRTM(**kws)
    timings.append(run_timing_test(F, num_iter))

    print('#### (6) Include channels and BT')
    descriptions.append('Jacobian, clear, channels, BT')
    kws['output_jacob_bt_flag'] = True
    kws['output_bt_flag'] = True
    F = setup_PCRTM(**kws)
    timings.append(run_timing_test(F, num_iter))

    ######################## 7 - 9 Cloudy sky forward
    
    #### Repeat 1-3, with a Cloud
    kws = basic_run_kw()

    print('#### (7) Minimal run: Mono-v only')
    descriptions.append('Forward,  cloud')
    F = setup_PCRTM(**kws)
    F.cld = cld
    F.cldDe = cldDe
    F.cldOD = cldOD
    F.cldP = cldP
    timings.append(run_timing_test(F, num_iter))

    print('#### (8) Include channels')
    descriptions.append('Forward,  cloud, channels')
    kws['output_ch_flag'] = True
    F = setup_PCRTM(**kws)
    F.cld = cld
    F.cldDe = cldDe
    F.cldOD = cldOD
    F.cldP = cldP
    timings.append(run_timing_test(F, num_iter))

    print('#### (9) Include channels and BT')
    descriptions.append('Forward,  cloud, channels, BT')
    kws['output_bt_flag'] = True
    F = setup_PCRTM(**kws)
    F.cld = cld
    F.cldDe = cldDe
    F.cldOD = cldOD
    F.cldP = cldP
    timings.append(run_timing_test(F, num_iter))

    ######################## 10 - 12 Cloudy sky jacobian

    if test_cloudy_jacobians:

        #### Repeat 1-3, with a Cloud and jacobian
        kws = basic_run_kw()
        kws['output_jacob_flag'] = True

        print('#### (10) Minimal run: Mono-v only')
        descriptions.append('Jacobian, cloud')
        F = setup_PCRTM(**kws)
        F.cld = cld
        F.cldDe = cldDe
        F.cldOD = cldOD
        F.cldP = cldP
        timings.append(run_timing_test(F, num_iter))

        #### (11) Include channels
        descriptions.append('Jacobian, cloud, channels')
        kws['output_jacob_ch_flag'] = True
        kws['output_ch_flag'] = True
        F = setup_PCRTM(**kws)
        F.cld = cld
        F.cldDe = cldDe
        F.cldOD = cldOD
        F.cldP = cldP
        timings.append(run_timing_test(F, num_iter))

        #### (12) Include channels and BT
        descriptions.append('Jacobian, cloud, channels, BT')
        kws['output_jacob_bt_flag'] = True
        kws['output_bt_flag'] = True
        F = setup_PCRTM(**kws)
        F.cld = cld
        F.cldDe = cldDe
        F.cldOD = cldOD
        F.cldP = cldP
        timings.append(run_timing_test(F, num_iter))

    ########################## limited jacobians

    #### Repeat 1-3, with Jacobian
    kws = basic_run_kw()
    kws['output_jacob_flag'] = True
    kws['output_jacob_ch_flag'] = True
    kws['output_ch_flag'] = True

    print('#### (13) Minimal run: Mono-v only')
    descriptions.append('Jacobian, clear, channels, 0 profile jac')
    F = setup_PCRTM(**kws)
    timings.append(run_timing_test(F, num_iter, num_profile_jacob=0))

    print('#### (14) Include channels')
    descriptions.append('Jacobian, clear, channels, 2 profile jac')
    F = setup_PCRTM(**kws)
    timings.append(run_timing_test(F, num_iter, num_profile_jacob=2))


    # print results
    print('PCRTM test timings, average on {:d} iterations'.format(num_iter))
    print('PCRTM_HOME:      ', os.environ['PCRTM_HOME'])
    print('PCRTM_INPUT_DIR: ', os.environ['PCRTM_INPUT_DIR'])

    print('Test results on {:d} trials'.format(num_iter))
    for n, timing in enumerate(timings):
        print('Test {:02d}  mean time: {:12.1f} [ms] description: {:s}'.format(
            n+1, timing.total_seconds()*1e3, descriptions[n]))


if __name__ == '__main__':
    import sys
    num_trials = int(sys.argv[1])
    run_test_sets(num_trials)
