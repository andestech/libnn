/***************************************************************************
 *  Copyright (C) 2018-2020 Andes Technology Corporation                   *
 *  All rights reserved.                                                   *
 ***************************************************************************/

/** @file*/

#ifndef __RISCV_NN_FUNCS_H__
#define __RISCV_NN_FUNCS_H__

#ifdef __cplusplus
extern    "C"
{
#endif

#include "riscv_math_types.h"
#include <string.h>

/**
 * @defgroup Support Support Functions
 * @brief Perform vector multiplication and data type conversion for Neural Network.
 *
 * @{
 */

/**
 * @brief           Duplicate the elements in a Q7 vector to a Q15 vector.
 *                  This is internally used function.
 * @param[in]       src         pointer of Q7 input vector
 * @param[out]      dst         pointer of Q15 output vector
 * @param[in]       size        element numbers in input/output vector
 * @return          None
 *
 * @b Example:
 * @code
 * #define SIZE 10
 * q7_t in_data[SIZE] = {...};
 * q15_t out_data[SIZE];
 *
 * riscv_nn_dup_s8_s16(in_data, out_data, SIZE);
 * @endcode
 */

void riscv_nn_dup_s8_s16(const q7_t * src, q15_t * dst, uint32_t size);

/**
 * @brief           Duplicate and reorder every two elements in a Q7 vector to
 *                  a Q15 vector. This is an internally used function.
 * @param[in]       src         pointer of Q7 input vector
 * @param[out]      dst         pointer of Q15 output vector
 * @param[in]       size        element numbers in input/output vector
 * @return          None
 *
 * @b Example:
 * @code
 * #define SIZE 10
 * q7_t in_data[SIZE] = {...};
 * q15_t out_data[SIZE];
 *
 * riscv_nn_dup_s8_s16_reordered(in_data, out_data, SIZE);
 * @endcode
 */

void riscv_nn_dup_s8_s16_reordered(const q7_t * src, q15_t * dst, uint32_t size);

void riscv_nn_dup_u8_u16_reordered(const u8_t * src, u16_t * dst, uint32_t size);

void riscv_nn_dup_s8_s16_offset(const q7_t *src,
                            q15_t *dst,
                            uint32_t block_size,
                            q15_t offset);

void static inline riscv_nn_dup_s16(const q15_t * src, q15_t * dst, uint32_t size)
{
#ifdef ENA_VEC_ISA
    long vl;
    while(size > 0)
    {
        NDS_VEC_VSETVLI(vl, size, NDS_VEC_VTYPE_SEW_E16, NDS_VEC_VTYPE_LMUL_M8);
        NDS_VEC_VLH_V(NDS_VEC_V0, src);
        NDS_VEC_VSH_V(NDS_VEC_V0, dst);
        src += vl;
        dst += vl;
        size -= vl;
    }
#elif defined(NDS_TOOLCHAIN_RISCV)
    memcpy(dst, src, sizeof(*src) * size);
#endif
}

void static inline riscv_nn_dup_s8(const q7_t * src, q7_t * dst, uint32_t size)
{
#ifdef ENA_VEC_ISA
    //used vector registers: v0
    long vl;
    while(size > 0)
    {
        NDS_VEC_VSETVLI(vl, size, NDS_VEC_VTYPE_SEW_E8, NDS_VEC_VTYPE_LMUL_M8);
        NDS_VEC_VLB_V(NDS_VEC_V0, src);
        NDS_VEC_VSB_V(NDS_VEC_V0, dst);
        src += vl;
        dst += vl;
        size -= vl;
    }
#else
    memcpy(dst, src, size);
#endif
}

//customized function for the VPU with the configuration of SIMD=VLEN/2
#ifdef ENA_VEC_ISA
void static inline riscv_nn_dup_s8_v2(const q7_t * src, q7_t * dst, uint32_t size)
{
    //used vector registers: v0
    long vl;
    while(size > 0)
    {
        NDS_VEC_VSETVLI(vl, size, NDS_VEC_VTYPE_SEW_E8, NDS_VEC_VTYPE_LMUL_M4);
        NDS_VEC_VLB_V(NDS_VEC_V0, src);
        NDS_VEC_VSB_V(NDS_VEC_V0, dst);
        src += vl;
        dst += vl;
        size -= vl;
    }
}
#endif

void static inline riscv_nn_dup_u8(const u8_t * src, u8_t * dst, uint32_t size)
{
#ifdef ENA_VEC_ISA
    //used vector registers: v0
    int32_t vl;
    while(size > 0)
    {
        NDS_VEC_VSETVLI_E8(vl, size);
        NDS_VEC_VLBU_V(NDS_VEC_V0, src);
        NDS_VEC_VSB_V(NDS_VEC_V0, dst);
        src += vl;
        dst += vl;
        size -= vl;
    }
#else
    memcpy(dst, src, size);
#endif
}

void static inline riscv_nn_set_zero_s16(q15_t *dst, uint32_t size)
{
#ifdef ENA_VEC_ISA
    long vl;
    NDS_VEC_VSETVLI(vl, size, NDS_VEC_VTYPE_SEW_E16, NDS_VEC_VTYPE_LMUL_M8);
    NDS_VEC_VAND_VI(NDS_VEC_V24, NDS_VEC_V24, 0);
    while(size > 0)
    {
        NDS_VEC_VSH_V(NDS_VEC_V24, dst);
        dst += vl;
        size -= vl;
        NDS_VEC_VSETVLI(vl, size, NDS_VEC_VTYPE_SEW_E16, NDS_VEC_VTYPE_LMUL_M8);
    }
#else
    // while(size-- > 0)
    // {
    //     *dst++ = 0;
    // }
    memset(dst, 0, sizeof(int16_t) * size);
#endif
}

void static inline riscv_nn_set_zero_s8(q7_t *dst, uint32_t size)
{
#ifdef ENA_VEC_ISA
    //Note. vector register v31 must be set to zero before calling this function
    long vl;
    NDS_VEC_VSETVLI(vl, size, NDS_VEC_VTYPE_SEW_E8, NDS_VEC_VTYPE_LMUL_M8);
    NDS_VEC_VAND_VI(NDS_VEC_V24, NDS_VEC_V24, 0);
    while(size > 0)
    {
        NDS_VEC_VSB_V(NDS_VEC_V24, dst);
        dst += vl;
        size -= vl;
        NDS_VEC_VSETVLI(vl, size, NDS_VEC_VTYPE_SEW_E8, NDS_VEC_VTYPE_LMUL_M8);
    }
#else
    // while(size-- > 0)
    // {
    //     *dst++ = 0;
    // }
    memset(dst, 0, size);
#endif
}

void static inline riscv_nn_set_zero_u8(u8_t *dst, uint32_t size)
{
#ifdef ENA_VEC_ISA
    //Note. vector register v31 must be set to zero before calling this function
    int32_t vl;
    NDS_VEC_VSETVLI_E8(vl, size);
    NDS_VEC_VAND_VI(NDS_VEC_V31, NDS_VEC_V31, 0);
    while(size > 0)
    {
        NDS_VEC_VSETVLI_E8(vl, size);
        NDS_VEC_VSB_V(NDS_VEC_V31, dst);
        dst += vl;
        size -= vl;
    }
#else
    memset(dst, 0, size);
#endif
}

void static inline riscv_nn_set_val_s8(q7_t *dst, q7_t val, uint32_t size)
{
#ifdef ENA_VEC_ISA
    uint32_t vl;
    NDS_VEC_VSETVLI_E8(vl, size);
    NDS_VEC_VAND_VI(NDS_VEC_V31, NDS_VEC_V31, 0x0);
    NDS_VEC_VADD_VX(NDS_VEC_V31, NDS_VEC_V31, val);

    //Note. vector register v31 must be set to val before calling this function
    while(size > 0)
    {
        NDS_VEC_VSETVLI_E8(vl, size);
        NDS_VEC_VSB_V(NDS_VEC_V31, dst);
        dst += vl;
        size -= vl;
    }
#else
    // while(size-- > 0)
    // {
    //     *dst++ = 0;
    // }
    memset(dst, val, size);
#endif
}

// Following is a customized function for q7 fully-connected.
// This function will read every two inputs (in[n] and in[n+1]), duplicate them
// 4 times and store them to the destination with the reordered ordering of
// (in[n] in[n] in[n+1] in[n+1] in[n] in[n] in[n+1] in[n+1])
// (left hand side elements have lower index).
// The rest (size mod 3) elements are copied to the destination directly.
#ifdef ENA_VEC_ISA
void static inline riscv_nn_dup_s8_x4_reordered(const q7_t * src, q7_t * dst, uint32_t size)
{
#if 0   //version-1 (generate the load index at run time)
    int32_t vl, vl2;
    uint32_t sft_bit = 8;
    uint32_t size2 = (size >> 2) << 2;

    //generate load index (swap the elements at index of 4n+1 and 4n+2)
    NDS_VEC_VSETVLI_E8(vl, size2);
    NDS_VEC_VID_V(NDS_VEC_V1);      //temp index vector
    NDS_VEC_VSETVLI_E32(vl, vl>>2);
    //clear 4n+1 and 4n+2 index elements
    NDS_VEC_VAND_VX(NDS_VEC_V0, NDS_VEC_V1, 0xFF0000FF);
    //select 4n+2 index elements
    NDS_VEC_VAND_VX(NDS_VEC_V2, NDS_VEC_V1, 0x00FF0000);
    //shift 4n+2 index elements to 4n+1 position
    NDS_VEC_VSRL_VX(NDS_VEC_V2, NDS_VEC_V2, sft_bit);
    NDS_VEC_VOR_VV(NDS_VEC_V0, NDS_VEC_V0, NDS_VEC_V2);
    //select 4n+1 index elements
    NDS_VEC_VAND_VX(NDS_VEC_V2, NDS_VEC_V1, 0x0000FF00);
    //shift 4n+1 index elements to 4n+2 position
    NDS_VEC_VSLL_VX(NDS_VEC_V2, NDS_VEC_V2, sft_bit);
    //V0: load ordering index
    NDS_VEC_VOR_VV(NDS_VEC_V0, NDS_VEC_V0, NDS_VEC_V2);

    q7_t *dst2 = dst+4;
    while(size2 > 0)
    {
        NDS_VEC_VSETVLI_E8(vl, size2);
        NDS_VEC_VLXBU_V(NDS_VEC_V1, src, NDS_VEC_V0);
        NDS_VEC_VWADDU_VX(NDS_VEC_V2, NDS_VEC_V1, 0);
        NDS_VEC_VSETVLI_E16_M2(vl2, size2);
        NDS_VEC_VSLL_VX(NDS_VEC_V4, NDS_VEC_V2, 8);
        NDS_VEC_VOR_VV(NDS_VEC_V2, NDS_VEC_V2, NDS_VEC_V4);

        //since the AndeSim now doesn't support SEW=64
        //thus, here we strided store the results to the destination twice
        NDS_VEC_VSETVLI_E32_M2(vl2, size2>>1);
        NDS_VEC_VSSW_V(NDS_VEC_V2, dst, 8);
        NDS_VEC_VSSW_V(NDS_VEC_V2, dst2, 8);

        size2 -= vl;
        src += vl;
        dst += vl<<2;
        dst2 += vl<<2;
    }

    //rest
    size2 = size & 0x3u;
    while (size2 > 0u)
    {
        *dst++ = *src++;
        size2--;
    }
#elif 0     //version-2 (plain C)
    uint32_t size2 = size >> 2;
    while(size2-- > 0)
    {
        q7_t in0 = *src++,
             in1 = *src++,
             in2 = *src++,
             in3 = *src++;

        *dst++ = in0;
        *dst++ = in0;
        *dst++ = in2;
        *dst++ = in2;
        *dst++ = in0;
        *dst++ = in0;
        *dst++ = in2;
        *dst++ = in2;

        *dst++ = in1;
        *dst++ = in1;
        *dst++ = in3;
        *dst++ = in3;
        *dst++ = in1;
        *dst++ = in1;
        *dst++ = in3;
        *dst++ = in3;
    }

    //rest
    size2 = size & 0x3u;
    while (size2 > 0u)
    {
        *dst++ = *src++;
        size2--;
    }
#elif 1

//VLEN checking
#ifndef VLEN
    #error "[Error]Please define VLEN!"
#elif (VLEN > 512)
    #error "[Error]The algorithm doesn't support VLEN > 512 now!"
#endif

    //assume the maximun supported VLEN=512b
    const uint8_t index[] = {  0,  0,  2,  2,  0,  0,  2,  2,  1,  1,  3,  3,  1,  1,  3,  3,
                               4,  4,  6,  6,  4,  4,  6,  6,  5,  5,  7,  7,  5,  5,  7,  7,
                               8,  8, 10, 10,  8,  8, 10, 10,  9,  9, 11, 11,  9,  9, 11, 11,
                              12, 12, 14, 14, 12, 12, 14, 14, 13, 13, 15, 15, 13, 13, 15, 15};
    uint32_t vl, size2 = size & 0xfffffffc;
    size2 = size2 << 2;     //output length is 4x of input length (since they will be duplicated 4 times)
    NDS_VEC_VSETVLI_E8(vl, size2);
    NDS_VEC_VLB_V(NDS_VEC_V0, index);

    // while(size2 >= 64*4)
    // {
    //     NDS_VEC_VSETVLI_E8(vl, size2);
    //     NDS_VEC_VLB_V(NDS_VEC_V1, src);

    //     // 1st 1/4 input data
    //     NDS_VEC_VRGATHER_VV(NDS_VEC_V2, NDS_VEC_V1, NDS_VEC_V0);
    //     NDS_VEC_VSB_V(NDS_VEC_V2, dst);
    //     dst += vl;

    //     // 2nd 1/4 input data
    //     NDS_VEC_VSLIDEDOWN_VI(NDS_VEC_V1, NDS_VEC_V1, 16);
    //     NDS_VEC_VRGATHER_VV(NDS_VEC_V2, NDS_VEC_V1, NDS_VEC_V0);
    //     NDS_VEC_VSB_V(NDS_VEC_V2, dst);
    //     dst += vl;

    //     // 3rd 1/4 input data
    //     NDS_VEC_VSLIDEDOWN_VI(NDS_VEC_V1, NDS_VEC_V1, 16);
    //     NDS_VEC_VRGATHER_VV(NDS_VEC_V2, NDS_VEC_V1, NDS_VEC_V0);
    //     NDS_VEC_VSB_V(NDS_VEC_V2, dst);
    //     dst += vl;

    //     // 4th 1/4 input data
    //     NDS_VEC_VSLIDEDOWN_VI(NDS_VEC_V1, NDS_VEC_V1, 16);
    //     NDS_VEC_VRGATHER_VV(NDS_VEC_V2, NDS_VEC_V1, NDS_VEC_V0);
    //     NDS_VEC_VSB_V(NDS_VEC_V2, dst);
    //     dst += vl;

    //     size2 -= vl << 2;
    //     src += vl;
    // }

    while(size2 > 0)
    {
        NDS_VEC_VSETVLI_E8(vl, size2 >> 2);
        NDS_VEC_VLB_V(NDS_VEC_V1, src);
        NDS_VEC_VSETVLI_E8(vl, size2);
        NDS_VEC_VRGATHER_VV(NDS_VEC_V2, NDS_VEC_V1, NDS_VEC_V0);
        NDS_VEC_VSB_V(NDS_VEC_V2, dst);
        size2 -= vl;
        src += vl >> 2;
        dst += vl;
    }

    //rest
    size2 = size & 0x3u;
    while (size2 > 0u)
    {
        *dst++ = *src++;
        size2--;
    }
#endif
}

// void static inline riscv_nn_dup_u8_x4_reordered(const u8_t * src, u8_t * dst, uint32_t size)
// {
//     int32_t vl, vl2;
//     uint32_t sft_bit = 8;
//     uint32_t size2 = (size >> 2) << 2;

//     //generate load index (swap the elements at index of 4n+1 and 4n+2)
//     NDS_VEC_VSETVLI_E8(vl, size2);
//     NDS_VEC_VID_V(NDS_VEC_V1);      //temp index vector
//     NDS_VEC_VSETVLI_E32(vl, vl>>2);
//     //clear 4n+1 and 4n+2 index elements
//     NDS_VEC_VAND_VX(NDS_VEC_V0, NDS_VEC_V1, 0xFF0000FF);
//     //select 4n+2 index elements
//     NDS_VEC_VAND_VX(NDS_VEC_V2, NDS_VEC_V1, 0x00FF0000);
//     //shift 4n+2 index elements to 4n+1 position
//     NDS_VEC_VSRL_VX(NDS_VEC_V2, NDS_VEC_V2, sft_bit);
//     NDS_VEC_VOR_VV(NDS_VEC_V0, NDS_VEC_V0, NDS_VEC_V2);
//     //select 4n+1 index elements
//     NDS_VEC_VAND_VX(NDS_VEC_V2, NDS_VEC_V1, 0x0000FF00);
//     //shift 4n+1 index elements to 4n+2 position
//     NDS_VEC_VSLL_VX(NDS_VEC_V2, NDS_VEC_V2, sft_bit);
//     //V0: load ordering index
//     NDS_VEC_VOR_VV(NDS_VEC_V0, NDS_VEC_V0, NDS_VEC_V2);

//     u8_t *dst2 = dst+4;
//     while(size2 > 0)
//     {
//         NDS_VEC_VSETVLI_E8(vl, size2);
//         NDS_VEC_VLXBU_V(NDS_VEC_V1, src, NDS_VEC_V0);
//         NDS_VEC_VWADDU_VX(NDS_VEC_V2, NDS_VEC_V1, 0);
//         NDS_VEC_VSETVLI_E16_M2(vl2, size2);
//         NDS_VEC_VSLL_VX(NDS_VEC_V4, NDS_VEC_V2, 8);
//         NDS_VEC_VOR_VV(NDS_VEC_V2, NDS_VEC_V2, NDS_VEC_V4);

//         //since the AndeSim now doesn't support SEW=64
//         //thus, here we strided store the results to the destination twice
//         NDS_VEC_VSETVLI_E32_M2(vl2, size2>>1);
//         NDS_VEC_VSSW_V(NDS_VEC_V2, dst, 8);
//         NDS_VEC_VSSW_V(NDS_VEC_V2, dst2, 8);

//         size2 -= vl;
//         src += vl;
//         dst += vl<<2;
//         dst2 += vl<<2;
//     }

//     //rest
//     size2 = size & 0x3u;
//     while (size2 > 0u)
//     {
//         *dst++ = *src++;
//         size2--;
//     }
// }

// Following is a customized function for fc_q15_fast.
// This function will read every two inputs (in[n] and in[n+1]), duplicate them
// 4 times and store them to the destination with the reordered ordering of
// (in[n] in[n+1] in[n] in[n+1] in[n] in[n+1] in[n] in[n+1]).
void static inline riscv_nn_dup_s16_x4_reordered(const q15_t * src, q15_t * dst, uint32_t size)
{
#if 0   //version-1: vlxh
    int32_t vl;
    uint32_t size2 = (size >> 1) << 3;  //x4

    //assume VLEN <= 512 (*2 -> convert to byte offset)
    //      load_index   = {0,   1,   0,   1,   0,   1,   0,   1,   2,   3,   2,   3,   2,   3,   2,   3,   4,   5,   4,   5,   4,   5,   4,   5,   6,   7,   6,   7,   6,   7,   6,   7}
    int16_t load_index[] = {0*2, 1*2, 0*2, 1*2, 0*2, 1*2, 0*2, 1*2, 2*2, 3*2, 2*2, 3*2, 2*2, 3*2, 2*2, 3*2, 4*2, 5*2, 4*2, 5*2, 4*2, 5*2, 4*2, 5*2, 6*2, 7*2, 6*2, 7*2, 6*2, 7*2, 6*2, 7*2,};

    NDS_VEC_VSETVLI_E16(vl, size2);
    NDS_VEC_VLH_V(NDS_VEC_V0, load_index);

    while(size2 > 0)
    {
        NDS_VEC_VSETVLI_E16(vl, size2);
        NDS_VEC_VLXH_V(NDS_VEC_V1, src, NDS_VEC_V0);
        NDS_VEC_VSH_V(NDS_VEC_V1, dst);
        size2 -= vl ;
        src += vl>>2;
        dst += vl;
    }

    //rest one
    //Note. if the size is odd, the last element needs to be processed specially
    if(size & 0x1)
    {
        *dst++ = *src;
        *dst++ = *src;
        *dst++ = *src;
        *dst++ = *src;
    }
#else   //version-2: vrgather (only support VLEN=512)

//VLEN checking
#ifndef VLEN
    #error "[Error]Please define VLEN!"
#elif (VLEN > 512)
    #error "[Error]The algorithm doesn't support VLEN > 512 now!"
#endif

    const int16_t load_index[] = { 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7};
    int32_t vl;
    uint32_t size2 = (size >> 1) << 3;  //x4
    NDS_VEC_VSETVLI_E16(vl, size2);
    NDS_VEC_VLH_V(NDS_VEC_V0, load_index);

    while(size2 > 0)
    {
        NDS_VEC_VSETVLI_E16(vl, size2 >> 2);
        NDS_VEC_VLH_V(NDS_VEC_V1, src);
        NDS_VEC_VSETVLI_E16(vl, size2);
        NDS_VEC_VRGATHER_VV(NDS_VEC_V2, NDS_VEC_V1, NDS_VEC_V0);
        NDS_VEC_VSH_V(NDS_VEC_V2, dst);
        size2 -= vl;
        src += vl >> 2;
        dst += vl;
    }

    //rest one
    //Note. if the size is odd, the last element needs to be processed specially
    if(size & 0x1)
    {
        *dst++ = *src;
        *dst++ = *src;
        *dst++ = *src;
        *dst++ = *src;
    }
#endif
}

// Following is a customized function for fc_mat_q7_vec_q15_fast.
// This function will read every two inputs (in[n] and in[n+1]), duplicate them
// 4 times and store them to the destination with the reordered ordering of
// (in[n] in[n] in[n+1] in[n+1] in[n] in[n] in[n+1] in[n+1]).
void static inline riscv_nn_dup_s16_x4_reordered_v2(const q15_t * src, q15_t * dst, uint32_t size)
{
#if 0   //version-1: vlxh
    int32_t vl;
    uint32_t size2 = (size >> 1) << 3;  //x4

    //assume VLEN <= 512 (*2 -> convert to byte offset)
    //      load_index   = {0,   0,   1,   1,   0,   0,   1,   1,   2,   2,   3,   3,   2,   2,   3,   3,   4,   4,   5,   5,   4,   4,   5,   5,   6,   6,   7,   7,   6,   6,   7,   7}
    int16_t load_index[] = {0*2, 0*2, 1*2, 1*2, 0*2, 0*2, 1*2, 1*2, 2*2, 2*2, 3*2, 3*2, 2*2, 2*2, 3*2, 3*2, 4*2, 4*2, 5*2, 5*2, 4*2, 4*2, 5*2, 5*2, 6*2, 6*2, 7*2, 7*2, 6*2, 6*2, 7*2, 7*2};

    NDS_VEC_VSETVLI_E16(vl, size2);
    NDS_VEC_VLH_V(NDS_VEC_V0, load_index);

    while(size2 > 0)
    {
        NDS_VEC_VSETVLI_E16(vl, size2);
        NDS_VEC_VLXH_V(NDS_VEC_V1, src, NDS_VEC_V0);
        NDS_VEC_VSH_V(NDS_VEC_V1, dst);
        size2 -= vl ;
        src += vl>>2;
        dst += vl;
    }

    //rest one
    //Note. if the size is odd, the last element needs to be processed specially
    if(size & 0x1)
    {
        *dst++ = *src;
        *dst++ = *src;
        *dst++ = *src;
        *dst++ = *src;
    }
#else   //version-2: vrgather

//VLEN checking
#ifndef VLEN
    #error "[Error]Please define VLEN!"
#elif (VLEN > 512)
    #error "[Error]The algorithm doesn't support VLEN > 512 now!"
#endif

    const int16_t load_index[] = { 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 5, 5, 6, 6, 7, 7, 6, 6, 7, 7};
    int32_t vl;
    uint32_t size2 = (size >> 1) << 3;  //x4
    NDS_VEC_VSETVLI_E16(vl, size2);
    NDS_VEC_VLH_V(NDS_VEC_V0, load_index);

    // while(size2 >= 32*4)
    // {
    //     NDS_VEC_VSETVLI_E16(vl, size2);
    //     NDS_VEC_VLH_V(NDS_VEC_V1, src);

    //     // 1st 1/4 input data
    //     NDS_VEC_VRGATHER_VV(NDS_VEC_V2, NDS_VEC_V1, NDS_VEC_V0);
    //     NDS_VEC_VSH_V(NDS_VEC_V2, dst);
    //     dst += vl;

    //     // 2nd 1/4 input data
    //     NDS_VEC_VSLIDEDOWN_VX(NDS_VEC_V1, NDS_VEC_V1, vl>>2);
    //     NDS_VEC_VRGATHER_VV(NDS_VEC_V2, NDS_VEC_V1, NDS_VEC_V0);
    //     NDS_VEC_VSH_V(NDS_VEC_V2, dst);
    //     dst += vl;

    //     // 3rd 1/4 input data
    //     NDS_VEC_VSLIDEDOWN_VX(NDS_VEC_V1, NDS_VEC_V1, vl>>2);
    //     NDS_VEC_VRGATHER_VV(NDS_VEC_V2, NDS_VEC_V1, NDS_VEC_V0);
    //     NDS_VEC_VSH_V(NDS_VEC_V2, dst);
    //     dst += vl;

    //     // 4th 1/4 input data
    //     NDS_VEC_VSLIDEDOWN_VX(NDS_VEC_V1, NDS_VEC_V1, vl>>2);
    //     NDS_VEC_VRGATHER_VV(NDS_VEC_V2, NDS_VEC_V1, NDS_VEC_V0);
    //     NDS_VEC_VSH_V(NDS_VEC_V2, dst);
    //     dst += vl;

    //     size2 -= vl << 2;
    //     src += vl;
    // }

    while(size2 > 0)
    {
        NDS_VEC_VSETVLI_E16(vl, size2 >> 2);
        NDS_VEC_VLH_V(NDS_VEC_V1, src);
        NDS_VEC_VSETVLI_E16(vl, size2);
        NDS_VEC_VRGATHER_VV(NDS_VEC_V2, NDS_VEC_V1, NDS_VEC_V0);
        NDS_VEC_VSH_V(NDS_VEC_V2, dst);
        size2 -= vl;
        src += vl >> 2;
        dst += vl;
    }

    //rest one
    //Note. if the size is odd, the last element needs to be processed specially
    if(size & 0x1)
    {
        *dst++ = *src;
        *dst++ = *src;
        *dst++ = *src;
        *dst++ = *src;
    }
#endif
}
#endif

/**
 * @brief           Multiply two Q7 vectors, right shift the results with
 *                  variable shift and saturate the results into Q7 range.
 * @param[in]       src1        pointer of the first input vector
 * @param[in]       src2        pointer of the second input vector
 * @param[out]      dst         pointer of the output vector
 * @param[in]       out_rshift  right shift amounts for output
 * @param[in]       size        element numbers in first input, second input or
 *                              output vector
 * @return          None
 *
 * @note
 * The multiplication results will be saturated into Q7 range [0x80, 0x7F].
 *
 * @b Example:
 * @code
 * #define SIZE 10
 * #define OUT_RSHIFT 2
 * q7_t src1[SIZE] = {...};
 * q7_t src2[SIZE] = {...};
 * q7_t dst[SIZE];
 *
 * riscv_nn_mul_q7(src1, src22, dst, OUT_RSHIFT, SIZE);
 * @endcode
 */

void riscv_nn_mul_q7(q7_t * src1,
                q7_t * src2,
                q7_t * dst,
                const uint16_t out_rshift,
                uint32_t size);

/**
 * @brief           Multiply two Q15 vectors, right shift the results with
 *                  variable shift and saturated the results into Q15 range.
 * @param[in]       src1        pointer of the first input vector
 * @param[in]       src2        pointer of the second input vector
 * @param[out]      dst         pointer of the output vector
 * @param[in]       out_rshift  right shift amounts for output
 * @param[in]       size        element numbers in first input, second input or
 *                              output vector
 * @return          None
 *
 * @note
 * The multiplication results will be saturated into Q15 range [0x8000, 0x7FFF].
 */

void riscv_nn_mul_q15(q15_t * src1,
                    q15_t * src2,
                    q15_t * dst,
                    const uint16_t out_rshift,
                    uint32_t size);

int32_t riscv_nn_mat_mult_nt_t_s8(const q7_t *lhs,
                                   const q7_t *rhs,
                                   const q31_t *bias,
                                   q7_t *dst,
                                   const int32_t *dst_multipliers,
                                   const int32_t *dst_shifts,
                                   const int32_t lhs_rows,
                                   const int32_t rhs_rows,
                                   const int32_t rhs_cols,
                                   const int32_t lhs_offset,
                                   const int32_t dst_offset,
                                   const int32_t activation_min,
                                   const int32_t activation_max);

//========== sub-functions for convolution ==========
// following are internal sub-functions called by NN convolution functions

/**
 * @brief           Multiply two Q7 matrices for convolution.
 * @param[in]       src1            pointer of first matrix
 * @param[in]       src2            pointer of second matrix (consists of 2
 *                                  column vectors)
 * @param[in]       out_tensor_ch   channels of output tensor (or row
 *                                  numbers of first matrix)
 * @param[in]       col_src1        columns of first matrix
 * @param[in]       bias_lshift     left shift amounts for bias
 * @param[in]       out_rshift      right shift amounts for output
 * @param[in]       bias            pointer of bias vector
 * @param[in,out]   out             pointer of output vector
 * @return          This function returns the incremented pointer of output
 *                  vector.
 *
 * @note
 * The second matrix consists of two column vectors from im2col.
 *
 * @b Example:
 * @code
 *  #define IN_CH 3
 *  #define KER_DIM 5
 *  #define OUT_CH 32
 *  #define COL_SRC1 (IN_CH * KER_DIM * KER_DIM)
 *  #define BIAS_LSHIFT 6
 *  #define OUT_RSHIFT 9
 *
 *  q7_t wt[IN_CH * KER_DIM * KER_DIM * OUT_CH] = {...};
 *  q7_t buf[2* COL_SRC1] = {...};
 *  q7_t bias[OUT_CH] = {...}
 *  q7_t tmp_buf[40960];
 *  q7_t *out = tmp_buf;
 *
 *  out = riscv_nn_mat_mul_kernel_q7(wt, buf, OUT_CH, COL_SRC1,
 *                              BIAS_LSHIFT, OUT_RSHIFT, bias, out);
 * @endcode
 */

q7_t *riscv_nn_mat_mul_kernel_q7(const q7_t * src1,
                               const q7_t * src2,
                               const uint16_t out_tensor_ch,
                               const uint16_t col_src1,
                               const uint16_t bias_lshift,
                               const uint16_t out_rshift,
                               const q7_t * bias,
                               q7_t * out);

q7_t *riscv_nn_mat_mul_kernel_q7_bias_2sft(const q7_t * src1,
                                        const q7_t * src2,
                                        const uint16_t out_tensor_ch,
                                        const uint16_t col_src1,
                                        const uint16_t pre_rshift,
                                        const uint16_t out_scale,
                                        const uint16_t post_rshift,
                                        const q31_t * bias,
                                        q7_t * out);

q15_t *riscv_nn_mat_mul_kernel_q7_q15_bias_2sft(const q7_t * src1,
                                              const q7_t * src2,
                                              const uint16_t out_tensor_ch,
                                              const uint16_t col_src1,
                                              const uint16_t pre_rshift,
                                              const uint16_t out_scale,
                                              const uint16_t post_rshift,
                                              const q31_t * bias,
                                              q15_t * out);

u8_t *riscv_nn_mat_mul_kernel_u8_bias_2sft(const q7_t * src1,
                                    const u8_t * src2,
                                    const uint16_t out_tensor_ch,
                                    const uint16_t col_src1,
                                    const uint16_t pre_rshift,
                                    const uint16_t out_scale,
                                    const uint16_t post_rshift,
                                    const q31_t * bias,
                                    u8_t * out);

q7_t *riscv_nn_mat_mul_kernel_u8_q7_bias_2sft(const q7_t * src1,
                                    const u8_t * src2,
                                    const uint16_t out_tensor_ch,
                                    const uint16_t col_src1,
                                    const uint16_t pre_rshift,
                                    const uint16_t out_scale,
                                    const uint16_t post_rshift,
                                    const q31_t * bias,
                                    q7_t * out);

q15_t *riscv_nn_mat_mul_kernel_u8_q15_bias_2sft(const q7_t * src1,
                                    const u8_t * src2,
                                    const uint16_t out_tensor_ch,
                                    const uint16_t col_src1,
                                    const uint16_t pre_rshift,
                                    const uint16_t out_scale,
                                    const uint16_t post_rshift,
                                    const q31_t * bias,
                                    q15_t * out);

q7_t *riscv_nn_mat_mul_kernel_q7_2sft(const q7_t * src1,
                                    const q7_t * src2,
                                    const uint16_t out_tensor_ch,
                                    const uint16_t col_src1,
                                    const uint16_t pre_rshift,
                                    const uint16_t out_scale,
                                    const uint16_t post_rshift,
                                    q7_t * out);

q15_t *riscv_nn_mat_mul_kernel_q7_q15_2sft(const q7_t * src1,
                                    const q7_t * src2,
                                    const uint16_t out_tensor_ch,
                                    const uint16_t col_src1,
                                    const uint16_t pre_rshift,
                                    const uint16_t out_scale,
                                    const uint16_t post_rshift,
                                    q15_t * out);

u8_t *riscv_nn_mat_mul_kernel_u8_2sft(const q7_t * src1,
                                    const u8_t * src2,
                                    const uint16_t out_tensor_ch,
                                    const uint16_t col_src1,
                                    const uint16_t pre_rshift,
                                    const uint16_t out_scale,
                                    const uint16_t post_rshift,
                                    u8_t * out);

q7_t *riscv_nn_mat_mul_kernel_u8_q7_2sft(const q7_t * src1,
                                    const u8_t * src2,
                                    const uint16_t out_tensor_ch,
                                    const uint16_t col_src1,
                                    const uint16_t pre_rshift,
                                    const uint16_t out_scale,
                                    const uint16_t post_rshift,
                                    q7_t * out);

q15_t *riscv_nn_mat_mul_kernel_u8_q15_2sft(const q7_t * src1,
                                    const u8_t * src2,
                                    const uint16_t out_tensor_ch,
                                    const uint16_t col_src1,
                                    const uint16_t pre_rshift,
                                    const uint16_t out_scale,
                                    const uint16_t post_rshift,
                                    q15_t * out);
/**
 * @brief           Multiply a Q7 matrix by a Q15 matrix for convolution.
 * @param[in]       src1            pointer of first matrix
 * @param[in]       src2            pointer of second matrix (consists of 2
 *                                  column vectors)
 * @param[in]       out_tensor_ch   channels of output tensor (or row
 *                                  numbers of first matrix)
 * @param[in]       col_src1        columns of first matrix
 * @param[in]       bias_lshift     left shift amounts for bias
 * @param[in]       out_rshift      right shift amounts for output
 * @param[in]       bias            pointer of bias vector
 * @param[in,out]   out             pointer of output vector
 * @return          This function returns the incremented pointer of output
 *                  vector.
 *
 * @note
 * The second matrix consists of two column vectors from im2col.
 */

  q7_t *riscv_nn_mat_mul_kernel_q7_q15(const q7_t * src1,
                                    const q15_t * src2,
                                    const uint16_t out_tensor_ch,
                                    const uint16_t col_src1,
                                    const uint16_t bias_lshift,
                                    const uint16_t out_rshift,
                                    const q7_t * bias,
                                    q7_t * out);

/**
 * @brief           Multiply two Q15 matrices for convolution.
 * @param[in]       src1            pointer of first matrix
 * @param[in]       src2            pointer of second matrix (consists of 2
 *                                  column vectors)
 * @param[in]       out_tensor_ch   channels of output tensor (or row
 *                                  numbers of first matrix)
 * @param[in]       col_src1        column numbers of second matrix
 * @param[in]       bias_lshift     left shift amounts for bias
 * @param[in]       out_rshift      right shift amounts for output
 * @param[in]       bias            pointer of bias vector
 * @param[in,out]   out             pointer to output vector
 * @return          This function returns the incremented pointer of output
 *                  vector.
 *
 * @note
 * The second matrix consists of two column vectors from im2col.
 */

  q7_t *riscv_nn_mat_mul_kernel_q15(const q15_t * src1,
                                const q15_t * src2,
                                const uint16_t out_tensor_ch,
                                const uint16_t col_src1,
                                const uint16_t bias_lshift,
                                const uint16_t out_rshift,
                                const q7_t * bias,
                                q7_t * out);

/**
 * @brief           Multiply a Q7 matrix by a Q15 matrix with reordered columns
 *                  for convolution.
 * @param[in]       src1            pointer of first matrix
 * @param[in]       src2            pointer of second matrix (consists of 2
 *                                  column vectors)
 * @param[in]       out_tensor_ch   channels of output tensor (or row
 *                                  numbers of first matrix)
 * @param[in]       col_src1        column numbers of first matrix
 * @param[in]       bias_lshift     left shift amounts for bias
 * @param[in]       out_rshift      right shift amounts for output
 * @param[in]       bias            pointer of bias vector
 * @param[in,out]   out             pointer of output vector
 * @return          This function returns the incremented pointer of output
 *                  vector.
 *
 * @note
 * The second matrix consists of two column vectors from im2col.
 */

  q7_t *riscv_nn_mat_mul_kernel_q7_q15_reordered(const q7_t * src1,
                                            const q15_t * src2,
                                            const uint16_t out_tensor_ch,
                                            const uint16_t col_src1,
                                            const uint16_t bias_lshift,
                                            const uint16_t out_rshift,
                                            const q7_t * bias,
                                            q7_t * out);

q7_t *riscv_nn_mat_mul_kernel_q7_q15_q7_reordered_bias_2sft(const q7_t * src1,
                                                    const q15_t * src2,
                                                    const uint16_t out_tensor_ch,
                                                    const uint16_t col_src1,
                                                    const uint16_t pre_rshift,
                                                    const uint16_t out_scale,
                                                    const uint16_t post_rshift,
                                                    const q31_t * bias,
                                                    q7_t * out);

q15_t *riscv_nn_mat_mul_kernel_q7_q15_q15_reordered_bias_2sft(const q7_t * src1,
                                                    const q15_t * src2,
                                                    const uint16_t out_tensor_ch,
                                                    const uint16_t col_src1,
                                                    const uint16_t pre_rshift,
                                                    const uint16_t out_scale,
                                                    const uint16_t post_rshift,
                                                    const q31_t * bias,
                                                    q15_t * out);

u8_t *riscv_nn_mat_mul_kernel_q7_u16_u8_reordered_bias_2sft(const q7_t * src1,
                                                    const u16_t * src2,
                                                    const uint16_t out_tensor_ch,
                                                    const uint16_t col_src1,
                                                    const uint16_t pre_rshift,
                                                    const uint16_t out_scale,
                                                    const uint16_t post_rshift,
                                                    const q31_t * bias,
                                                    u8_t * out);

q7_t *riscv_nn_mat_mul_kernel_q7_u16_q7_reordered_bias_2sft(const q7_t * src1,
                                                    const u16_t * src2,
                                                    const uint16_t out_tensor_ch,
                                                    const uint16_t col_src1,
                                                    const uint16_t pre_rshift,
                                                    const uint16_t out_scale,
                                                    const uint16_t post_rshift,
                                                    const q31_t * bias,
                                                    q7_t * out);

q15_t *riscv_nn_mat_mul_kernel_q7_u16_q15_reordered_bias_2sft(const q7_t * src1,
                                                    const u16_t * src2,
                                                    const uint16_t out_tensor_ch,
                                                    const uint16_t col_src1,
                                                    const uint16_t pre_rshift,
                                                    const uint16_t out_scale,
                                                    const uint16_t post_rshift,
                                                    const q31_t * bias,
                                                    q15_t * out);

q7_t *riscv_nn_mat_mul_kernel_q7_q15_q7_reordered_2sft(const q7_t * src1,
                                                    const q15_t * src2,
                                                    const uint16_t out_tensor_ch,
                                                    const uint16_t col_src1,
                                                    const uint16_t pre_rshift,
                                                    const uint16_t out_scale,
                                                    const uint16_t post_rshift,
                                                    q7_t * out);

q15_t *riscv_nn_mat_mul_kernel_q7_q15_q15_reordered_2sft(const q7_t * src1,
                                                    const q15_t * src2,
                                                    const uint16_t out_tensor_ch,
                                                    const uint16_t col_src1,
                                                    const uint16_t pre_rshift,
                                                    const uint16_t out_scale,
                                                    const uint16_t post_rshift,
                                                    q15_t * out);

u8_t *riscv_nn_mat_mul_kernel_q7_u16_u8_reordered_2sft(const q7_t * src1,
                                                    const u16_t * src2,
                                                    const uint16_t out_tensor_ch,
                                                    const uint16_t col_src1,
                                                    const uint16_t pre_rshift,
                                                    const uint16_t out_scale,
                                                    const uint16_t post_rshift,
                                                    u8_t * out);

q7_t *riscv_nn_mat_mul_kernel_q7_u16_q7_reordered_2sft(const q7_t * src1,
                                                    const u16_t * src2,
                                                    const uint16_t out_tensor_ch,
                                                    const uint16_t col_src1,
                                                    const uint16_t pre_rshift,
                                                    const uint16_t out_scale,
                                                    const uint16_t post_rshift,
                                                    q7_t * out);

q15_t *riscv_nn_mat_mul_kernel_q7_u16_q15_reordered_2sft(const q7_t * src1,
                                                    const u16_t * src2,
                                                    const uint16_t out_tensor_ch,
                                                    const uint16_t col_src1,
                                                    const uint16_t pre_rshift,
                                                    const uint16_t out_scale,
                                                    const uint16_t post_rshift,
                                                    q15_t * out);

q7_t *riscv_nn_mat_mul_kernel_q15_q15_q7_bias_2sft(const q15_t * src1,
                                                const q15_t * src2,
                                                const uint16_t out_tensor_ch,
                                                const uint16_t col_src1,
                                                const uint16_t pre_rshift,
                                                const uint16_t out_scale,
                                                const uint16_t post_rshift,
                                                const q31_t * bias,
                                                q7_t * out);

q15_t *riscv_nn_mat_mul_kernel_q15_q15_q15_bias_2sft(const q15_t * src1,
                                                const q15_t * src2,
                                                const uint16_t out_tensor_ch,
                                                const uint16_t col_src1,
                                                const uint16_t pre_rshift,
                                                const uint16_t out_scale,
                                                const uint16_t post_rshift,
                                                const q31_t * bias,
                                                q15_t * out);

u8_t *riscv_nn_mat_mul_kernel_q15_q15_u8_bias_2sft(const q15_t * src1,
                                                const q15_t * src2,
                                                const uint16_t out_tensor_ch,
                                                const uint16_t col_src1,
                                                const uint16_t pre_rshift,
                                                const uint16_t out_scale,
                                                const uint16_t post_rshift,
                                                const q31_t * bias,
                                                u8_t * out);

q7_t *riscv_nn_mat_mul_kernel_q15_q15_q7_2sft(const q15_t * src1,
                                            const q15_t * src2,
                                            const uint16_t out_tensor_ch,
                                            const uint16_t col_src1,
                                            const uint16_t pre_rshift,
                                            const uint16_t out_scale,
                                            const uint16_t post_rshift,
                                            q7_t * out);

q15_t *riscv_nn_mat_mul_kernel_q15_q15_q15_2sft(const q15_t * src1,
                                            const q15_t * src2,
                                            const uint16_t out_tensor_ch,
                                            const uint16_t col_src1,
                                            const uint16_t pre_rshift,
                                            const uint16_t out_scale,
                                            const uint16_t post_rshift,
                                            q15_t * out);

u8_t *riscv_nn_mat_mul_kernel_q15_q15_u8_2sft(const q15_t * src1,
                                            const q15_t * src2,
                                            const uint16_t out_tensor_ch,
                                            const uint16_t col_src1,
                                            const uint16_t pre_rshift,
                                            const uint16_t out_scale,
                                            const uint16_t post_rshift,
                                            u8_t * out);

q7_t *riscv_nn_mat_mult_kernel_s8_s16(const q7_t *input_a,
                                    const q15_t *input_b,
                                    const uint16_t output_ch,
                                    const int32_t *out_shift,
                                    const int32_t *out_mult,
                                    const int32_t out_offset,
                                    const int16_t activation_min,
                                    const int16_t activation_max,
                                    const uint16_t num_col_a,
                                    const int32_t *const output_bias,
                                    q7_t *out_0);

q7_t *riscv_nn_mat_mult_kernel_s8_offset(const q7_t *input_a,
                                    const q7_t *input_b,
                                    const uint16_t output_ch,
                                    const int32_t *out_shift,
                                    const int32_t *out_mult,
                                    const int32_t input_offset,
                                    const int32_t out_offset,
                                    const int16_t activation_min,
                                    const int16_t activation_max,
                                    const uint16_t num_col_a,
                                    const int32_t *const output_bias,
                                    q7_t *out_0);

int32_t riscv_nn_vec_mat_mult_t_s8(const q7_t *lhs,
                                    const q7_t *rhs,
                                    const q31_t *bias,
                                    q7_t *dst,
                                    const int32_t lhs_offset,
                                    const int32_t rhs_offset,
                                    const int32_t dst_offset,
                                    const int32_t dst_multiplier,
                                    const int32_t dst_shift,
                                    const int32_t rhs_cols,
                                    const int32_t rhs_rows,
                                    const int32_t activation_min,
                                    const int32_t activation_max);

int32_t riscv_nn_vec_mat_mult_t_s8_v2(const q7_t *lhs,
                                    const q7_t *rhs,
                                    const q31_t *bias,
                                    q7_t *dst,
                                    const int32_t lhs_offset,
                                    const int32_t rhs_offset,
                                    const int32_t dst_offset,
                                    const int32_t dst_multiplier,
                                    const int32_t dst_shift,
                                    const int32_t rhs_cols,
                                    const int32_t rhs_rows,
                                    const int32_t activation_min,
                                    const int32_t activation_max);

int riscv_nn_vec_mat_mult_t_svdf_s8(const q7_t *lhs,
                                    const q7_t *rhs,
                                    q15_t *dst,
                                    const int32_t lhs_offset,
                                    const int32_t rhs_offset,
                                    const int32_t dst_offset,
                                    const int32_t dst_multiplier,
                                    const int32_t dst_shift,
                                    const int32_t rhs_cols,
                                    const int32_t rhs_rows,
                                    const int32_t activation_min,
                                    const int32_t activation_max);
/**
 *   * @}
 */
#ifdef __cplusplus
}
#endif

#endif
