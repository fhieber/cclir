bin_PROGRAMS = vectorize

vectorize_SOURCES = vectorize.cpp
vectorize_LDADD   = $(top_srcdir)/src/libclir.a $(CDEC_ROOT)/utils/libutils.a -lz

AM_CPPFLAGS = -W -Wall -Wno-sign-compare -I$(CDEC_ROOT)/utils -I$(CDEC_ROOT)/decoder