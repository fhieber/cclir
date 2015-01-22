bin_PROGRAMS = retrieve retrieve_multithreading bowfd

retrieve_SOURCES = retrieve.cpp
retrieve_LDADD   = $(top_srcdir)/src/core/libclir.a $(CDEC_ROOT)/utils/libutils.a -lz

retrieve_multithreading_SOURCES = retrieve_multithreading.cpp
retrieve_multithreading_CXXFLAGS = $(OPENMP_CXXFLAGS)
retrieve_multithreading_LDADD   = $(top_srcdir)/src/core/libclir.a $(CDEC_ROOT)/utils/libutils.a -lz

bowfd_SOURCES = bowfd.cpp
bowfd_CXXFLAGS = $(OPENMP_CXXFLAGS)
bowfd_LDADD = $(top_srcdir)/src/core/libclir.a $(CDEC_ROOT)/decoder/libcdec.a $(CDEC_ROOT)/mteval/libmteval.a $(CDEC_ROOT)/utils/libutils.a $(CDEC_ROOT)/klm/search/libksearch.a $(CDEC_ROOT)/klm/lm/libklm.a $(CDEC_ROOT)/klm/util/libklm_util.a $(CDEC_ROOT)/klm/util/double-conversion/libklm_util_double.a -lz

# DOES NOT COMPILE RIGHT NOW (!)
#baseline_retrieval_SOURCES = baseline_retrieval.cpp
#baseline_retrieval_LDADD   = $(top_srcdir)/src/core/libclir.a $(CDEC_ROOT)/utils/libutils.a -lz

AM_CPPFLAGS = -W -Wall -Wno-sign-compare -I$(CDEC_ROOT)/utils -I$(CDEC_ROOT)/decoder