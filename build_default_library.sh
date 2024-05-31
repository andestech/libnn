#!/usr/bin/env bash

################################################################################
#
# This script is used to build multi-lib NN libraries with Andes toolchain.
#
# $1: library install path
# $2: toolchain prefix
# $3: header files install path
# $4: compiler name (it could be "gcc" or "clang")
#
# Example-1: to build and instll the NN library for riscv64-unknown-elf toolcahin
# ./build_default_library.sh <RISCV64-UNKNOWN-ELF_TOOLCHAIN_PATH>/riscv64-elf/lib riscv64-elf <RISCV64-UNKNOWN-ELF_TOOLCHAIN_PATH>/riscv64-elf/include gcc
################################################################################

#########################
# The command line help
#########################
show_help()
{
    echo "Usage: $0 lib_insatll_path tool_prefix header_install_path compiler_name " >&2
    echo
    echo "where:"
    echo "    lib_insatll_path      library install path"
    echo "    tool_prefix           toolchain prefix"
    echo "    header_install_path   header files install path"
    echo "    compiler_name         compiler name (it could be "gcc" or "clang")"
    echo
}

#=========== main ==============
LIB_INSTALL_PATH=${1}
TOOLCHAIN_PREFIX=${2}
HEADER_INSTALL_PATH=${3}
if [[ -n ${TOOLCHAIN_PREFIX} ]]; then
    CC="${TOOLCHAIN_PREFIX}-${4}"
else
    CC="${4}"
fi

# check are the parameters valid
if [[ -z "${LIB_INSTALL_PATH}" ]]; then
    echo "Error! The library install path is empty." | tee build_lib.log
    show_help
    exit 1
fi
if [[ -z "${TOOLCHAIN_PREFIX}" ]]; then
    echo "Error! The toolchain prefix is empty." | tee build_lib.log
    show_help
    exit 1
fi
if [[ -z "${HEADER_INSTALL_PATH}" ]]; then
    echo "Error! The header files install path is empty." | tee build_lib.log
    show_help
    exit 1
fi
if [[ -z "${4}" ]]; then
    echo "Error! The compiler name is empty." | tee build_lib.log
    show_help
    exit 1
fi
if ! ${CC} -v &> /dev/null; then
    echo "Error! The compiler(${CC}) could not be found." | tee build_lib.log
    exit 1
fi


# set compiling flags and misc global variables
DEFAULT_MAKE_FLAGS="-j16"
LIB_ROOT=$(dirname $0)
HEADER_PATH="${LIB_ROOT}/Include"
INCLUDE_DIR="-I${LIB_ROOT}/Include -I${LIB_ROOT}/internal"
DEFAULT_COMPILE_FLAGS="-O3 -ffunction-sections -fdata-sections -Wall -Werror ${INCLUDE_DIR}"
DEFAULT_DYNAMIC_LIB_FLAG="-fpic -shared -lm"
if [[ ${TOOLCHAIN_PREFIX} == "nds32"* ]] ; then
    DEFAULT_COMPILE_FLAGS="-std=c99 ${DEFAULT_COMPILE_FLAGS}"
elif [[ ${TOOLCHAIN_PREFIX} =~ ^riscv(32|64)* ]] ; then
    DEFAULT_COMPILE_FLAGS+=" -fno-strict-aliasing"
fi
DSP_COMPILE_FLAGS="-mext-dsp"
VEC_COMPILE_FLAGS="-mext-vector -mtune=andes-45-series -mcmodel=large -DENA_VEC_ISA -DMAX_VLEN=1024 -DNDS_VEC_RVV_VERSION=1000 -DENA_NDS_V5_VEC_DOT_PROD"
CUR_COMPILE_FLAG=""
CUR_LOG_FILE="build_lib.log"
CWD=$(pwd)
LIB_ROOT=$(readlink -f `dirname ${0}`)
LIB_OBJ_DIR="${LIB_ROOT}/lib_objs"
LIB_NAME="libnn"
LIB_NAME_A="${LIB_NAME}.a"
LIB_NAME_P_A="${LIB_NAME}_p.a"
LIB_NAME_V_ELEN32_A="${LIB_NAME}_v.a"
LIB_NAME_V_ELEN64_A="${LIB_NAME}_v_elen64.a"
LIB_NAME_V_ELEN32_SEG_A="${LIB_NAME}_v_seg.a"
LIB_NAME_V_ELEN64_SEG_A="${LIB_NAME}_v_elen64_seg.a"
LIB_NAME_SO="${LIB_NAME}.so"
LIB_NAME_P_SO="${LIB_NAME}_p.so"
LIB_NAME_V_ELEN32_SO="${LIB_NAME}_v.so"
LIB_NAME_V_ELEN64_SO="${LIB_NAME}_v_elen64.so"
LIB_NAME_V_ELEN32_SEG_SO="${LIB_NAME}_v_seg.so"
LIB_NAME_V_ELEN64_SEG_SO="${LIB_NAME}_v_elen64_seg.so"
MAKE="make -f ${LIB_ROOT}/Makefile_lib.mak CROSS_COMPILE="${TOOLCHAIN_PREFIX}-" CC="${CC}""

# zol flags for V3 CPUs with DSP
if [[ ${TOOLCHAIN_PREFIX} == "nds32"* ]] ; then
    DSP_COMPILE_FLAGS+=" -mext-zol"
fi

# change directory to LIB_ROOT to make sure the git repository could be found
# (git repository is required as generating nds_version.c)
cd ${LIB_ROOT}

# generate nds_version.c to make library version number
# (remember to move nds_version.c to LIB_ROOT)
sh ${LIB_ROOT}/nds_autogen_info.sh "libnn"
mv nds_version.c ${LIB_ROOT}

# change directory back to CWD
cd ${CWD}

# start building process
if [[ ${TOOLCHAIN_PREFIX} == *"linux"* ]]; then
    echo "Building multi-lib of ${LIB_NAME} for ${TOOLCHAIN_PREFIX} toolchain ..."
    MULTI_LIB_STR=`${CC} -print-multi-lib`
    # workaound for the latest linker issue
    if [[ ${TOOLCHAIN_PREFIX} =~ riscv32* ]]; then
        MULTI_LIB_STR=`echo ${MULTI_LIB_STR} | sed 's/\.;/lib32\/ilp32d;/g'`
    else
        MULTI_LIB_STR=`echo ${MULTI_LIB_STR} | sed 's/\.;/lib64\/lp64d;/g'`
    fi

    for MULTI_LIB_LINE in ${MULTI_LIB_STR}
    do
        TEMP_SUB_DIR=`echo ${MULTI_LIB_LINE}| cut -d ';' -f1`
        TEMP_FLAG=`echo ${MULTI_LIB_LINE}| cut -d ';' -f2`
        CUR_LOG_FILE="build_multi_lib_`echo $TEMP_SUB_DIR | sed 's/\//_/g'`.log"
        CUR_LIB_PATH="${LIB_INSTALL_PATH}/${TEMP_SUB_DIR}"
        CUR_COMPILE_FLAG=`echo $TEMP_FLAG | sed 's/@/ -/g'`

        # automatically enable the zfh extension if the toolcahin supports it
        # Note. We should check the zfh extension could be enabled or not with
        # EACH "CUR_COMPILE_FLAG". The zfh extension may be supported with the
        # toolchain; however, it's disabled by the "CUR_COMPILE_FLAG".
        DEFAULT_COMPILE_FLAGS_TMP="${DEFAULT_COMPILE_FLAGS}"
        ${CC} -mzfh -E -dM ${CUR_COMPILE_FLAG} - < /dev/null &> /dev/null
        if [ $? -eq 0 ]; then
            DEFAULT_COMPILE_FLAGS_TMP="${DEFAULT_COMPILE_FLAGS_TMP} -mzfh"
        fi

        mkdir -p ${CUR_LIB_PATH}
        echo "${TEMP_FLAG} --> ${CUR_LIB_PATH}"
        echo "    Building static libraries ..."

        # (1) static + plain C
        echo -e "        Building plain C version ... \c"
        ${MAKE} clean > /dev/null
        ${MAKE} ${DEFAULT_MAKE_FLAGS} CFLAGS="${DEFAULT_COMPILE_FLAGS_TMP} ${CUR_COMPILE_FLAG}" >> ${CUR_LOG_FILE} 2>&1
        if [ $? == 0 ]; then
            echo "success"
            cp -pf ${LIB_ROOT}/${LIB_NAME_A} ${CUR_LIB_PATH}
        else
            echo "FAIL"
        fi

        # (2) static + DSP
        echo -e "        Building dsp version ... \c"
        ${MAKE} clean > /dev/null
        ${MAKE} ${DEFAULT_MAKE_FLAGS} CFLAGS="${DEFAULT_COMPILE_FLAGS_TMP} ${DSP_COMPILE_FLAGS} ${CUR_COMPILE_FLAG}" >> ${CUR_LOG_FILE} 2>&1
        if [ $? == 0 ]; then
            echo "success"
            cp -pf ${LIB_ROOT}/${LIB_NAME_A} ${CUR_LIB_PATH}/${LIB_NAME_P_A}
        else
            echo "FAIL"
        fi

        echo "    Building dynamic libraries ..."

        # (3) dynamic + plain C
        echo -e "        Building plain C version ... \c"
        ${MAKE} clean > /dev/null
        ${MAKE} ${DEFAULT_MAKE_FLAGS} CFLAGS="${DEFAULT_COMPILE_FLAGS_TMP} ${DEFAULT_DYNAMIC_LIB_FLAG} ${CUR_COMPILE_FLAG} " CCASFLAGS="${DEFAULT_DYNAMIC_LIB_FLAG}" >> ${CUR_LOG_FILE} 2>&1
        if [[ $? -eq 0 ]]; then
            ALL_OBJ_FILES=`ls ${LIB_OBJ_DIR}/*.o`
            ${CC} ${DEFAULT_DYNAMIC_LIB_FLAG} ${CUR_COMPILE_FLAG} -o ${CUR_LIB_PATH}/${LIB_NAME_SO} ${ALL_OBJ_FILES} >> ${CUR_LOG_FILE}
            if [[ $? -eq 0 ]]; then
                echo "success"
            else
                echo "FAIL"
            fi
        else
            echo "FAIL"
        fi

        # (4) dynamic + DSP
        echo -e "        Building dsp version ... \c"
        ${MAKE} clean > /dev/null
        ${MAKE} ${DEFAULT_MAKE_FLAGS} CFLAGS="${DEFAULT_COMPILE_FLAGS_TMP} ${DSP_COMPILE_FLAGS} ${DEFAULT_DYNAMIC_LIB_FLAG} ${CUR_COMPILE_FLAG}" CCASFLAGS="${DEFAULT_DYNAMIC_LIB_FLAG}">> ${CUR_LOG_FILE} 2>&1
        if [[ $? -eq 0 ]]; then
            ALL_OBJ_FILES=`ls ${LIB_OBJ_DIR}/*.o`
            ${CC} ${DEFAULT_DYNAMIC_LIB_FLAG} ${CUR_COMPILE_FLAG} -o ${CUR_LIB_PATH}/${LIB_NAME_P_SO} ${ALL_OBJ_FILES} >> ${CUR_LOG_FILE}
            if [[ $? -eq 0 ]]; then
                echo "success"
            else
                echo "FAIL"
            fi
        else
            echo "FAIL"
        fi
    done

else

    #elf toolchains
    echo "Building multi-lib of ${LIB_NAME} for ${TOOLCHAIN_PREFIX} toolchain ..."
    echo "--------------------"
    echo "The multi-lib configurations are listed as:"
    ${CC} -print-multi-lib \
    && MULTI_LIB_STR=`${CC} -print-multi-lib` \
    || echo "Fail to get multi-lib configuration with '${CC} -print-multi-lib'."

    echo "--------------------"
    echo "Begin building ${LIB_NAME} ..."
    echo

    for MULTI_LIB_LINE in ${MULTI_LIB_STR}
    do
        TEMP_SUB_DIR=`echo ${MULTI_LIB_LINE}| cut -d ';' -f1`
        TEMP_FLAG=`echo ${MULTI_LIB_LINE}| cut -d ';' -f2`
        CUR_LOG_FILE="build_multi_lib_`echo $TEMP_SUB_DIR | sed 's/\//_/g'`.log"
        CUR_LIB_PATH="${LIB_INSTALL_PATH}/${TEMP_SUB_DIR}"
        CUR_COMPILE_FLAG=`echo $TEMP_FLAG | sed 's/@/ -/g'`
        CUR_COMPILE_FLAG=`echo $CUR_COMPILE_FLAG | sed 's/-mext-zol//g'`

        # automatically enable the zfh extension if the toolcahin supports it
        # Note. We should check the zfh extension could be enabled or not with
        # EACH "CUR_COMPILE_FLAG". The zfh extension may be supported with the
        # toolchain; however, it's disabled by the "CUR_COMPILE_FLAG".
        DEFAULT_COMPILE_FLAGS_TMP="${DEFAULT_COMPILE_FLAGS}"
        ${CC} -mzfh -E -dM ${CUR_COMPILE_FLAG} - < /dev/null &> /dev/null
        if [ $? -eq 0 ]; then
            DEFAULT_COMPILE_FLAGS_TMP="${DEFAULT_COMPILE_FLAGS_TMP} -mzfh"
        fi

        ## When building the multi-lib with -mno-nds option, the -mext-dsp otion should be removed.
        if [[ $CUR_COMPILE_FLAG == *"-mno-nds"* ]]; then
            if [[ $CUR_COMPILE_FLAG == *"-mext-dsp"* ]]; then
                CUR_COMPILE_FLAG=`echo $CUR_COMPILE_FLAG | sed 's/-mext-dsp//g'`
            fi
        fi

        if [[ ${TOOLCHAIN_PREFIX} == nds32* ]]; then
            CUR_COMPILE_FLAG=`echo $CUR_COMPILE_FLAG | sed 's/-mext-dsp/-mext-dsp -mext-zol -mcpu=d10/g'`
        fi

        # optimized flags for n15/d15
        CUR_COMPILE_FLAG=`echo $CUR_COMPILE_FLAG | \
        sed 's/-mcpu=graywolf/-mcpu=d15 -funroll-loops --param max-unroll-times=4 -fsched-pressure --param sched-pressure-algorithm=2 -fno-auto-inc-dec/g'`

        CUR_COMPILE_FLAG=`echo $CUR_COMPILE_FLAG | \
        sed 's/-mcpu=n15/-mcpu=n15 -funroll-loops --param max-unroll-times=4 -fsched-pressure --param sched-pressure-algorithm=2 -fno-auto-inc-dec/g'`

        # setup the cflags
        CFLAGS="${DEFAULT_COMPILE_FLAGS_TMP} ${CUR_COMPILE_FLAG}"

        echo "${TEMP_FLAG} --> ${CUR_LIB_PATH}"
        echo "    Making ${LIB_NAME_A} with CFLAGS=${CFLAGS}"

        ${MAKE} clean > /dev/null
        ${MAKE} ${DEFAULT_MAKE_FLAGS} CFLAGS="${CFLAGS}" 2>&1 1>${CUR_LOG_FILE}
        if [ $? -eq 0 ]; then
            mkdir -p ${CUR_LIB_PATH} \
            && mv ${LIB_ROOT}/${LIB_NAME_A} ${CUR_LIB_PATH} 2>&1 | tee -a ${CUR_LOG_FILE}
            if [ ${PIPESTATUS[0]} == 0 ]; then
                echo "Move ${LIB_NAME_A} to ${CUR_LIB_PATH}"
            fi
        else
            echo -e "Build ${LIB_NAME_A} failed.\nPlease see ${CUR_LOG_FILE} for detailed error messages."
            exit 1
        fi
        echo "--------------------"
    done

    # v-ext version library is not for RVE toolchain now
    ${CC} -E -dM - < /dev/null | grep -qw "__riscv_abi_rve"
    is_rve=$?
    if [ ${is_rve} -ne 0 ] && [[ ${TOOLCHAIN_PREFIX} == "riscv64"* ]]; then

        # check the float abi
        ${CC} -E -dM - < /dev/null | grep -qw "__riscv_float_abi_soft"
        is_soft_ft_abi=$?
        if [ ${is_soft_ft_abi} -eq 0 ]; then
            ELE32_CONFIG="@mext-vector=zve32x"
        else
            ELE32_CONFIG="@mext-vector=zve32f"
        fi
        ELE32_CONFIG_LOG_NAME=${ELE32_CONFIG/@/}
        ELE32_CONFIG_LOG_NAME=${ELE32_CONFIG_LOG_NAME/=/-}
        LIB_V_COMBINATION=("mext-vector/${ELE32_CONFIG_LOG_NAME};${ELE32_CONFIG};${LIB_NAME_V_ELEN32_A}"
                           "mext-vector/${ELE32_CONFIG_LOG_NAME}/ENA_VEC_ISA_ZVLSSEG;${ELE32_CONFIG}@DENA_VEC_ISA_ZVLSSEG;${LIB_NAME_V_ELEN32_SEG_A}"
                           "mext-vector;;${LIB_NAME_V_ELEN64_A}"
                           "mext-vector/ENA_VEC_ISA_ZVLSSEG;@DENA_VEC_ISA_ZVLSSEG;${LIB_NAME_V_ELEN64_SEG_A}")

        for lib_v_option in "${LIB_V_COMBINATION[@]}"; do
           TEMP_SUB_DIR=`echo ${lib_v_option}| cut -d ';' -f1`
           TEMP_FLAG=`echo ${lib_v_option}| cut -d ';' -f2`
           CUR_LIB_NAME_V=`echo ${lib_v_option}| cut -d ';' -f3`
           CUR_LOG_FILE="build_multi_lib_`echo $TEMP_SUB_DIR | sed 's/\//_/g'`.log"
           CUR_LIB_PATH="${LIB_INSTALL_PATH}"
           CUR_COMPILE_FLAG=`echo $TEMP_FLAG | sed 's/@/ -/g'`
           CFLAGS="${DEFAULT_COMPILE_FLAGS_TMP} ${CUR_COMPILE_FLAG} ${VEC_COMPILE_FLAGS}"

           echo "${TEMP_FLAG} --> ${CUR_LIB_PATH}"
           echo "    Making ${CUR_LIB_NAME_V} with CFLAGS=${CFLAGS}"
           ${MAKE} clean > /dev/null
           ${MAKE} ${DEFAULT_MAKE_FLAGS} CFLAGS="${CFLAGS}" 2>&1 1>>${CUR_LOG_FILE}

           if [ $? -eq 0 ]; then
               mv ${LIB_ROOT}/${LIB_NAME_A} ${LIB_INSTALL_PATH}/${CUR_LIB_NAME_V} 2>&1 | tee -a ${CUR_LOG_FILE}
               if [ ${PIPESTATUS[0]} == 0 ]; then
                   echo "Move ${CUR_LIB_NAME_V} to ${CUR_LIB_PATH}"
               fi
           else
               echo -e "Build ${CUR_LIB_NAME_V} failed.\nPlease see ${CUR_LOG_FILE} for detailed error messages."
               exit 1
           fi
           echo "--------------------"
        done
    fi
fi

mkdir -p ${HEADER_INSTALL_PATH}
cp ${HEADER_PATH}/*.h ${HEADER_INSTALL_PATH} && rm -f ${HEADER_INSTALL_PATH}/*_nn_support.h
echo "Done!"
