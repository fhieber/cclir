bin_PROGRAMS = vectorize build_index

vectorize_SOURCES = vectorize.cpp
vectorize_LDADD   = $(top_srcdir)/src/libclir.a $(CDEC_ROOT)/utils/libutils.a -lz

build_index_SOURCES = build_index.cpp
build_index_LDADD = $(top_srcdir)/src/libclir.a $(CDEC_ROOT)/utils/libutils.a -lz

AM_CPPFLAGS = -W -Wall -Wno-sign-compare -I$(CDEC_ROOT)/utils -I$(CDEC_ROOT)/decoder