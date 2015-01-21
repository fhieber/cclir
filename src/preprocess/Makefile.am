bin_PROGRAMS = vectorize generate_psqs interpolate_queries build_index

vectorize_SOURCES = vectorize.cpp
vectorize_LDADD   = $(top_srcdir)/src/libclir.a $(CDEC_ROOT)/utils/libutils.a -lz

generate_psqs_SOURCES = generate_psqs.cpp
generate_psqs_LDADD = $(top_srcdir)/src/libclir.a $(CDEC_ROOT)/decoder/libcdec.a $(CDEC_ROOT)/mteval/libmteval.a $(CDEC_ROOT)/utils/libutils.a $(CDEC_ROOT)/klm/search/libksearch.a $(CDEC_ROOT)/klm/lm/libklm.a $(CDEC_ROOT)/klm/util/libklm_util.a $(CDEC_ROOT)/klm/util/double-conversion/libklm_util_double.a -lz

interpolate_queries_SOURCES = interpolate_queries.cpp
interpolate_queries_LDADD = $(top_srcdir)/src/libclir.a $(CDEC_ROOT)/utils/libutils.a -lz

build_index_SOURCES = build_index.cpp
build_index_LDADD = $(top_srcdir)/src/libclir.a $(CDEC_ROOT)/utils/libutils.a -lz

AM_CPPFLAGS = -W -Wall -Wno-sign-compare -I$(CDEC_ROOT)/utils -I$(CDEC_ROOT)/decoder