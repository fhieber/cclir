SUBDIRS = core preprocess retrieval training

bin_PROGRAMS = get-tf-vectors

noinst_LIBRARIES = libclir.a 
libclir_a_SOURCES = \
	util.h \
	util.cpp \
	stopwords.cpp \
	stopwords.h \
	pugixml.hpp \
	pugixml.cpp \
	ttable.h \
	lexical-ttable.h \
	nbest-ttable.h \
	dftable.cpp \
	document.h \
	document.cpp \
	query.h
	scoring.h \
	clir.h \
	distclir.h \
	clir-scoring.h \
	serialization.h \
	observer.h \
	index.h

get_tf_vectors_SOURCES = get-tf-vectors.cpp
get_tf_vectors_LDADD   = $(top_srcdir)/src/libclir.a $(CDEC_ROOT)/utils/libutils.a -lz

AM_CPPFLAGS = -W -Wall -Wno-sign-compare -I$(CDEC_ROOT)/utils -I$(CDEC_ROOT)/decoder
