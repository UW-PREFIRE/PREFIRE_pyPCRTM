import numpy as np
import pyPCRTM
import h5py

from test_setups import setup_PCRTM

def create_stored_jacobian_output():
    F = setup_PCRTM(
        sensor_id = 2,
        output_jacob_flag=True,
        output_tr_flag=False,
        output_ch_flag=True,
        output_jacob_ch_flag=True,
        output_bt_flag=False,
        output_jacob_bt_flag=False)
    dat = F.forward_rt()
    with h5py.File('stored_jacobian_output.h5', 'w') as h:
        for v in ('wn', 'rad', 'krad_tsurf', 'krad_esurf', 'krad_t', 'krad_mol'):
            h.create_dataset(
                name = v, shape = dat[v].shape,
                dtype = dat[v].dtype, data = dat[v],
                compression='gzip', compression_opts=2,
                shuffle=True)

def test_jacobian():

    F = setup_PCRTM(
        sensor_id = 2,
        output_jacob_flag=True,
        output_tr_flag=False,
        output_ch_flag=True,
        output_jacob_ch_flag=True,
        output_bt_flag=False,
        output_jacob_bt_flag=False)
    dat = F.forward_rt()

    vlist = ('wn', 'rad', 'krad_tsurf', 'krad_esurf', 'krad_t', 'krad_mol')
    for v in vlist:
        assert np.all(np.isfinite(dat[v])), \
            f'non-finite values detected in {v}'
        print(f"All values in '{v}' are finite.")

    with h5py.File('stored_jacobian_output.h5', 'r') as h:
        for v in vlist:
            if v == 'krad_mol':
                # molecular jacobian have a lot of strange behavior.
                # gfortran vs ifort
                atols = [5e-5, 1e-8, 1e-6, 5e-7, 3e-8, 1e-7]
                # different ifort compiles
                #atols = [6e-6, 1e-8, 1e-6, 2e-7, 3e-8, 5e-8]
                for m in range(6):
                    assert np.allclose(dat[v][:,:,m], h[v][:][:,:,m], \
                                       rtol=1e-5, atol=atols[m]), \
                        f'computation run did not match kmol for mol index {m}'
                    print(f"computation run matched kmol for mol index '{m}'")
            elif v == 'krad_tsurf':
                assert np.allclose(dat[v], h[v][:], rtol=1e-5, atol=1e-6), \
                    f'computation run did not match stored result for variable {v}'
                print(f"computation run matched stored result for variable '{v}'")
            elif v == 'krad_esurf':
                # atol is rather large here, but this is the emissivity jacobian, which is
                # two orders of magnitude larger than the surface temp jacobian.
                assert np.allclose(dat[v], h[v][:], rtol=1e-5, atol=1e-4), \
                    f'computation run did not match stored result for variable {v}'
                print(f"computation run matched stored result for variable '{v}'")
            else:
                # note the relative tolerance is the default (rtol)
                # but the absolute tolerance is slightly increased over
                # default. the T jacobian results appear to have differences
                # of order few x 1e-8 regardless of the amplitude.
                assert np.allclose(dat[v], h[v][:], rtol=1e-5, atol=5e-8), \
                    f'computation run did not match stored result for variable {v}'
                print(f"computation run matched stored result for variable '{v}'")

if __name__ == "__main__":
    test_jacobian()
    print("Success!")
