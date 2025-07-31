PCRTM_DIR=/data/rttools/PCRTM/PCRTM_V3.4

ifort -fPIC -c PCRTM_wrapper_module.f90 -o PCRTM_wrapper_module.o \
	-I${PCRTM_DIR}/include -L${PCRTM_DIR}/lib

ifort PCRTM_wrapper_module.o ${PCRTM_DIR}/lib/libPCRTM.a -shared \
	-I${PCRTM_DIR}/include -L${PCRTM_DIR}/lib \
	-o PCRTM_wrapper_module.so
