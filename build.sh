#/bin/bash
################################################################################
#Input:
# $1: Specified compiler
#     (e.g. riscv32-unknown-elf-gcc, riscv32-unknown-elf-clang
#      or riscv64-unknown-elf-gcc)
# $2: Extra compilation flags for the library
#
# Example:
#   ./build.sh "riscv32-unknown-elf-gcc" "-O0 -g3"
###############################################################################

# set the common compilation flags
CFLAG="-Wall -Werror -O3 -ffunction-sections -fdata-sections -fno-strict-aliasing"

# set the compiler
CC="${1}"

# get the extra compilation flags
if [ -n "${2}" ]; then
    CFLAG="${CFLAG} ${2}"
fi

# get the toolchain prefix
if [[ ${CC} == *-* ]]; then
    TOOLCHAIN_PREFIX="${CC%-*}-"
else
    TOOLCHAIN_PREFIX=""
fi

LIB_ROOT=$(readlink -f `dirname ${0}`)
INCLUDE="-I${LIB_ROOT}/Include -I${LIB_ROOT}/internal"
CFLAG="${CFLAG} ${INCLUDE}"
RELEASE_FOLDER="${LIB_ROOT}/release"

echo "**********************************************"
echo " Building libnn library ..."
echo "**********************************************"
echo
make clean -f ${LIB_ROOT}/Makefile_lib.mak
make -f ${LIB_ROOT}/Makefile_lib.mak CFLAGS="${CFLAG}" CROSS_COMPILE="${TOOLCHAIN_PREFIX}" CC="${CC}"
ret=$?
echo
echo "**********************************************"
echo "Building libnn library done!"
echo

exit $ret
