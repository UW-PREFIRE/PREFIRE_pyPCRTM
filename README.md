# PREFIRE_pyPCRTM

Python wrapper for the PREFIRE_PCRTM_V3.4 package.

The implementation uses an intermediary Fortran module to allow the compound user types defined in PCRTM to be populated with simple C-style variables, annotated in Fortran with ISO_C_BINDING types. The annotated functions can then be called from Python with ctypes calls. The Python code can then automate much of the ctypes labeling, and add docstrings, etc, to make the code much more usable.

This code is released under the terms of this [LICENSE](LICENSE).  The version of this package can be found in [VERSION.txt](VERSION.txt).

# Installation

## Requirements

This package requires compilation with a modern Fortran compiler. Both the Intel Fortran (ifort) and GNU Fortran (gfortran) compilers have been used and tested.

Python version 3.6+ is also required, along with the following third-party Python packages (for the testing step): numpy

## Python Environment Setup

It is recommended to install the above Python packages in a dedicated conda environment (or something similar).  The packages used (and their versions) can be found in [conda_env.list](conda_env.list).

For example, using conda (and specifying Python 3.10.x from the conda-forge channel):

```
conda create --name for_PREFIRE_pyPCRTM -c conda-forge python=3.10;
conda activate for_PREFIRE_pyPCRTM;
conda install -c conda-forge numpy;
```

## Build process

### Preliminary steps and information

First, the PREFIRE_PCRTM_V3.4 package itself needs to be built (see the README.md in that repository). Make a note of the root directory (that contains include/ and lib/) that the built PREFIRE_PCRTM_V3.4 files were installed to, as this will be needed to build the Python wrapper.  For example:

`/data/RT_tools/PCRTM/gfortran_build/PREFIRE_PCRTM_V3.4/`

The Python wrapper includes a Fortran 90 "adapter", compiled into a shared object, that performs two main functions: (1) stores the initialized PCRTM coefficient arrays for re-use, when running the model over large ensembles of profiles; and (2) converts plain C-friendly inputs (which can be easily mapped to Python NumPy arrays) into the Fortran derived-type variables that are defined by PCRTM. The main wrapper for PREFIRE is the PCRTM_wrapper target.

### Prepare for the Fortran wrapper code build:

`cd pyPCRTM;`

Depending on the desired Fortran compiler to use, edit `Makefile.intel` or `Makefile.gfortran` to set compiler, compiler flags, etc.

Load any compiler modules (if needed).  For example:

```
module load license_intel intel;   For Intel compilers
  -OR-
module load gcc;   For GNU compilers
```

Clean any prior build detritus:

`make clean;`

### Now build the Fortran wrapper:

The following example is compiled with gfortran (substitute "intel" for "gfortran" to use the Intel compiler). The Makefiles assume that an environment variable PCRTM_HOME is defined, that is set to the directory containing the `lib/` and `include/` subdirectories of the PCRTM installation, containing the static archive and module files created by building PCRTM. The user will likely need to change the value of PCRTM_HOME in the example below to reflect where their PREFIRE_PCRTM_V3.4 installation actually is.

```
PCRTM_HOME=/data/RT_tools/PCRTM/gfortran_build/PREFIRE_PCRTM_V3.4 make gfortran;
   -OR-
export PCRTM_HOME=/data/RT_tools/PCRTM/gfortran_build/PREFIRE_PCRTM_V3.4;
make gfortran;
```

# Testing

The main target here is PCRTM_wrapper_module.so. The Python wrapper code loads this shared object file using the ctypes.CDLL mechanism.

The Python code also assumes that the home directory of the repository is on the
PYTHONPATH (or sys.path), in order to be importable within Python. The `pyPCRTM/` subdirectory contains the __init__.py file, so `import pyPCRTM` would be the import statement.

There is an automated test script in the `test/` subdirectory. It will check if the wrapper can reproduce a stored calculation result. Note that it requires the NumPy Python library. Examples of how to run this test:

```
cd ../test;

ln -s /data/RT_tools/PCRTM/PCRTM_V3.4/INPUTDIR INPUTDIR;   To allow the script to find the static ancillary data

python test_forward_run.py;
python test_jacobian_run.py;
    -OR-
PYTHONPATH=$HOME/projects/current/PREFIRE_pyPCRTM python test_forward_run.py;
PYTHONPATH=$HOME/projects/current/PREFIRE_pyPCRTM python test_jacobian_run.py;
```

A successful run should print some PCRTM log messages to console, followed by "Success!". A failed run would produce a Python exception (e.g., "AssertionError ...", if the numbers do not match the stored results).

## Environment summary for using PREFIRE_pyPCRTM

For example:

```
export PCRTM_HOME=/data/RT_tools/PCRTM/gfortran_build/PREFIRE_PCRTM_V3.4;
export PCRTM_INPUT_DIR=/data/RT_tools/PCRTM/PCRTM_V3.4/INPUTDIR/;   # must end with '/'
export PYTHONPATH=/data/RT_tools/PREFIRE_pyPCRTM/gfortran_build;
```

To use your own build, the PYTHONPATH and/or PCRTM_HOME variables would probably need to be changed. The PCRTM_INPUT_DIR should probably not be duplicated, unless you want to make a copy of that directory structure and all of the (quite large volume of) binary files within.

## _The creation of this code was supported by NASA, as part of the PREFIRE (Polar Radiant Energy in the Far-InfraRed Experiment) CubeSat mission._
