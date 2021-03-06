AC_CONFIG_MACRO_DIR([m4])
AC_INIT([cclir],[2014-10-09])
AC_CONFIG_SRCDIR([src])
AM_INIT_AUTOMAKE
AC_CONFIG_HEADERS(config.h)

#cdec
AM_CONDITIONAL([HAVE_CDEC], false)
AC_ARG_WITH([cdec],
	[AC_HELP_STRING([--with-cdec=PATH], [(required) path to cdec repository])],
	[CDEC_ROOT=$withval],
	[CDEC_ROOT=no]
	)
if test "x$CDEC_ROOT" != 'xno'
then
	AC_SUBST(CDEC_ROOT,"$CDEC_ROOT")
	AM_CONDITIONAL([HAVE_CDEC], true)
	AC_MSG_NOTICE([CDEC_ROOT set to $CDEC_ROOT;])
else
	AC_MSG_ERROR([cannot find cdec. Specify with --with-cdec])
	exit 1
fi

AC_PROG_LIBTOOL
AC_PROG_CC
AC_PROG_CXX
AX_CXX_COMPILE_STDCXX_11([],[mandatory])
AC_LANG_CPLUSPLUS
AC_OPENMP
AC_CHECK_LIB(rt, clock_gettime)
BOOST_REQUIRE([1.44])
BOOST_FILESYSTEM
BOOST_PROGRAM_OPTIONS
BOOST_SYSTEM
BOOST_REGEX
BOOST_SERIALIZATION
BOOST_TEST
BOOST_THREADS
AM_PATH_PYTHON
AC_CHECK_LIB(dl, dlopen)

AC_CHECK_HEADERS(zlib.h,
AC_CHECK_LIB(z, gzread,[
AC_DEFINE(HAVE_ZLIB,[],[Do we have zlib])
ZLIBS="$ZLIBS -lz"
]))

AC_CHECK_HEADERS(bzlib.h,
AC_CHECK_LIB(bz2, BZ2_bzReadOpen,[
AC_DEFINE(HAVE_BZLIB,[],[Do we have bzlib])
ZLIBS="$ZLIBS -lbz2"
]))

AC_CHECK_HEADERS(lzma.h,
AC_CHECK_LIB(lzma, lzma_code,[
AC_DEFINE(HAVE_XZLIB,[],[Do we have lzma])
ZLIBS="$ZLIBS -llzma"
]))

#BOOST_THREADS
CPPFLAGS="$CPPFLAGS $BOOST_CPPFLAGS"
LDFLAGS="$LDFLAGS $BOOST_PROGRAM_OPTIONS_LDFLAGS $BOOST_SERIALIZATION_LDFLAGS $BOOST_SYSTEM_LDFLAGS $BOOST_FILESYSTEM_LDFLAGS"
# $BOOST_THREAD_LDFLAGS"
LIBS="$LIBS $BOOST_PROGRAM_OPTIONS_LIBS $BOOST_SERIALIZATION_LIBS $BOOST_SYSTEM_LIBS $BOOST_FILESYSTEM_LIBS $ZLIBS"
# $BOOST_THREAD_LIBS"


CPPFLAGS="-DPIC $CPPFLAGS -DHAVE_CONFIG_H"
CXXFLAGS="$CXX11_SWITCH $CXXFLAGS -fPIC -g -O3"
CFLAGS="$CFLAGS -fPIC -g -O3"

#zmq
AC_CHECK_HEADER(zmq.hpp)
if test "$ac_cv_header_zmq_hpp" = no
then
	AC_MSG_WARN([zmq.hpp not found. Will not build server/client programs!])
	AM_CONDITIONAL([HAVE_ZMQ], false)
else
	AM_CONDITIONAL([HAVE_ZMQ], true)
	AC_SUBST(HAVE_ZMQ,1)
fi

AC_CONFIG_FILES([Makefile])
AC_CONFIG_FILES([src/Makefile])
AC_CONFIG_FILES([src/core/Makefile])
AC_CONFIG_FILES([src/preprocess/Makefile])
AC_CONFIG_FILES([src/search/Makefile])
AC_CONFIG_FILES([src/training/Makefile])
AC_CONFIG_FILES([src/training/bow_fd/Makefile])
AC_CONFIG_FILES([src/training/reldec/Makefile])
AC_CONFIG_FILES([src/training/pltrain/Makefile])
AC_CONFIG_FILES([src/training/adarank/Makefile])

# end
AC_OUTPUT
