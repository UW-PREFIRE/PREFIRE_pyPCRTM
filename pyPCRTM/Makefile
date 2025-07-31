THIS_PROV1 = ../data/this_prov1.txt
THIS_PROV2 = ../data/this_prov2.txt
PCRTM_PROV = $(PCRTM_HOME)/include/build_provenance.txt
BUILD_PROV = ../data/build_provenance.txt

intel: determine_provenance Makefile.ifort
	make -f Makefile.ifort

gfortran: determine_provenance Makefile.gfortran
	make -f Makefile.gfortran

clean:
	rm -f PCRTM_wrapper_module.o PCRTM_wrapper_module.so \
		pcrtm_wrapper_module.mod \
		PCRTM_SOLAR_wrapper_module.o PCRTM_SOLAR_wrapper_module.so \
		pcrtm_solar_wrapper_module.mod $(BUILD_PROV) \
		$(THIS_PROV1) $(THIS_PROV2)

determine_provenance:
	@git rev-parse --short=8 --verify HEAD 2>/dev/null 1>$(THIS_PROV1) || (echo "unknown" > $(THIS_PROV1)); \
	this_prov_tmp=`cat $(THIS_PROV1)`; \
	if [ $$this_prov_tmp != "unknown" ]; then \
	   cp $(THIS_PROV1) $(THIS_PROV2); \
	   git diff --quiet || (echo "$$this_prov_tmp(modified)" > $(THIS_PROV2)); \
	else \
	   echo '' > $(THIS_PROV2); \
	fi ; \
	echo $(PCRTM_HOME) > $(THIS_PROV1); \
	cat $(THIS_PROV2) $(THIS_PROV1) $(PCRTM_PROV) | tr '\n' ' ' > $(BUILD_PROV); \
	rm -f $(THIS_PROV1) $(THIS_PROV2)
