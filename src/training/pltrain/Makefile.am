bin_PROGRAMS = generate-instances view-instances retrieval set-gold-permutation sgd

generate_instances_SOURCES = generate-instances.cpp
generate_instances_LDADD = $(top_srcdir)/src/core/libclir.a $(CDEC_ROOT)/decoder/libcdec.a $(CDEC_ROOT)/mteval/libmteval.a $(CDEC_ROOT)/utils/libutils.a $(CDEC_ROOT)/klm/search/libksearch.a $(CDEC_ROOT)/klm/lm/libklm.a $(CDEC_ROOT)/klm/util/libklm_util.a $(CDEC_ROOT)/klm/util/double-conversion/libklm_util_double.a -lz

retrieval_SOURCES = retrieval.cpp
retrieval_LDADD = $(top_srcdir)/src/core/libclir.a $(CDEC_ROOT)/utils/libutils.a -lz
retrieval_CXXFLAGS = $(OPENMP_CXXFLAGS)

set_gold_permutation_SOURCES = set-gold-permutation.cpp
set_gold_permutation_LDADD = $(top_srcdir)/src/core/libclir.a $(CDEC_ROOT)/utils/libutils.a -lz

sgd_SOURCES = sgd.cpp 
sgd_LDADD = $(top_srcdir)/src/core/libclir.a $(CDEC_ROOT)/utils/libutils.a -lz
sgd_CXXFLAGS = $(OPENMP_CXXFLAGS)

view_instances_SOURCES = view-instances.cpp
view_instances_LDADD = $(top_srcdir)/src/core/libclir.a $(CDEC_ROOT)/utils/libutils.a -lz

AM_CPPFLAGS = -W -Wall -Wno-sign-compare -I$(CDEC_ROOT)/utils -I$(CDEC_ROOT)/decoder