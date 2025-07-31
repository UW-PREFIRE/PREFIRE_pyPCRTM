import numpy as np
import pyPCRTM

import expected_data

from test_setups import setup_PCRTM, add_cloud_layer

def test_forward_bt():
    F = setup_PCRTM(
        sensor_id = 2,
        output_jacob_flag=False,
        output_tr_flag=False,
        output_ch_flag=True,
        output_jacob_ch_flag=False,
        output_bt_flag=True,
        output_jacob_bt_flag=False)
    dat = F.forward_rt()
    assert np.allclose(dat['bt'], expected_data.expected_BT_v00), \
        'forward computed BT spectrum (clear) did not match stored result'
    print("forward computed BT spectrum (clear) matched stored result")
    for v in dat:
        assert np.all(np.isfinite(dat[v])), \
            f'non-finite values detected in {v}'
        print(f"All values in '{v}' (clear) are finite.")


def test_forward_bt_withcloud():
    F = setup_PCRTM(
        sensor_id = 2,
        output_jacob_flag=False,
        output_tr_flag=False,
        output_ch_flag=True,
        output_jacob_ch_flag=False,
        output_bt_flag=True,
        output_jacob_bt_flag=False)
    add_cloud_layer(F)
    dat = F.forward_rt()
    assert np.allclose(dat['bt'], expected_data.expected_cloudy_BT_v00), \
        'forward computed BT spectrum (cloudy) did not match stored result'
    print("forward computed BT spectrum (cloudy) matched stored result")
    for v in dat:
        assert np.all(np.isfinite(dat[v])), \
            f'non-finite values detected in {v}'
        print(f"All values in '{v}' (cloudy) are finite.")

if __name__ == "__main__":
    test_forward_bt()
    test_forward_bt_withcloud()
    print("Success!")
