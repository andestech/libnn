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
#include "riscv_nn_types.h"
#include "riscv_nn_activation.h"
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
#if   defined(NDS_TOOLCHAIN_RISCV)
    memcpy(dst, src, sizeof(*src) * size);
#endif
}

void static inline riscv_nn_dup_s8(const q7_t * src, q7_t * dst, uint32_t size)
{
    memcpy(dst, src, size);
}

void static inline riscv_nn_dup_s8_v2(const q7_t * src, q7_t * dst, uint32_t size)
{
#ifdef ENA_VEC_INTRINSIC
    long vl;
    long size2 = size;
    while(size2 > 0)
    {
        vl = __riscv_vsetvl_e8m4(size2);
        vint8m4_t vSrc = __riscv_vle8_v_i8m4(src, vl);
        __riscv_vse8(dst, vSrc, vl);
        src += vl;
        dst += vl;
        size2 -= vl;
    }
#else
    memcpy(dst, src, size);
#endif
}

void static inline riscv_nn_dup_u8(const u8_t * src, u8_t * dst, uint32_t size)
{
    memcpy(dst, src, size);
}

void static inline riscv_nn_set_zero_s16(q15_t *dst, uint32_t size)
{
    // while(size-- > 0)
    // {
    //     *dst++ = 0;
    // }
    memset(dst, 0, sizeof(int16_t) * size);
}

void static inline riscv_nn_set_zero_s8(q7_t *dst, uint32_t size)
{
    // while(size-- > 0)
    // {
    //     *dst++ = 0;
    // }
    memset(dst, 0, size);
}

void static inline riscv_nn_set_zero_u8(u8_t *dst, uint32_t size)
{
    memset(dst, 0, size);
}

void static inline riscv_nn_set_val_s8(q7_t *dst, q7_t val, uint32_t size)
{
    // while(size-- > 0)
    // {
    //     *dst++ = 0;
    // }
    memset(dst, val, size);
}

// Following is a customized function for q7 fully-connected.
// This function will read every two inputs (in[n] and in[n+1]), duplicate them
// 4 times and store them to the destination with the reordered ordering of
// (in[n] in[n] in[n+1] in[n+1] in[n] in[n] in[n+1] in[n+1])
// (left hand side elements have lower index).
// The rest (size mod 3) elements are copied to the destination directly.

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
                                   const int32_t activation_max,
                                   const int32_t lhs_cols_offset);

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

q7_t *riscv_nn_mat_mul_kernel_q7_unroll4(const q7_t * src1,
                                    const q7_t * src2,
                                    const uint16_t out_tensor_ch,
                                    const uint16_t col_src1,
                                    const uint16_t bias_lshift,
                                    const uint16_t out_rshift,
                                    const q7_t * bias,
                                    q7_t * out);

q7_t *riscv_nn_mat_mul_kernel_q7_bias_2sft_unroll4(const q7_t * src1,
                                        const q7_t * src2,
                                        const uint16_t out_tensor_ch,
                                        const uint16_t col_src1,
                                        const uint16_t pre_rshift,
                                        const uint16_t out_scale,
                                        const uint16_t post_rshift,
                                        const q31_t * bias,
                                        q7_t * out);

q15_t *riscv_nn_mat_mul_kernel_q7_q15_bias_2sft_unroll4(const q7_t * src1,
                                                    const q7_t * src2,
                                                    const uint16_t out_tensor_ch,
                                                    const uint16_t col_src1,
                                                    const uint16_t pre_rshift,
                                                    const uint16_t out_scale,
                                                    const uint16_t post_rshift,
                                                    const q31_t * bias,
                                                    q15_t * out);

u8_t *riscv_nn_mat_mul_kernel_u8_bias_2sft_unroll4(const q7_t * src1,
                                                const u8_t * src2,
                                                const uint16_t out_tensor_ch,
                                                const uint16_t col_src1,
                                                const uint16_t pre_rshift,
                                                const uint16_t out_scale,
                                                const uint16_t post_rshift,
                                                const q31_t * conv_out,
                                                u8_t * out);

q7_t *riscv_nn_mat_mul_kernel_u8_q7_bias_2sft_unroll4(const q7_t * src1,
                                                    const u8_t * src2,
                                                    const uint16_t out_tensor_ch,
                                                    const uint16_t col_src1,
                                                    const uint16_t pre_rshift,
                                                    const uint16_t out_scale,
                                                    const uint16_t post_rshift,
                                                    const q31_t * bias,
                                                    q7_t * out);

q15_t *riscv_nn_mat_mul_kernel_u8_q15_bias_2sft_unroll4(const q7_t * src1,
                                                    const u8_t * src2,
                                                    const uint16_t out_tensor_ch,
                                                    const uint16_t col_src1,
                                                    const uint16_t pre_rshift,
                                                    const uint16_t out_scale,
                                                    const uint16_t post_rshift,
                                                    const q31_t * bias,
                                                    q15_t * out);

q7_t *riscv_nn_mat_mul_kernel_q7_2sft_unroll4(const q7_t * src1,
                                            const q7_t * src2,
                                            const uint16_t out_tensor_ch,
                                            const uint16_t col_src1,
                                            const uint16_t pre_rshift,
                                            const uint16_t out_scale,
                                            const uint16_t post_rshift,
                                            q7_t * out);

q15_t *riscv_nn_mat_mul_kernel_q7_q15_2sft_unroll4(const q7_t * src1,
                                                const q7_t * src2,
                                                const uint16_t out_tensor_ch,
                                                const uint16_t col_src1,
                                                const uint16_t pre_rshift,
                                                const uint16_t out_scale,
                                                const uint16_t post_rshift,
                                                q15_t * out);

u8_t *riscv_nn_mat_mul_kernel_u8_2sft_unroll4(const q7_t * src1,
                                            const u8_t * src2,
                                            const uint16_t out_tensor_ch,
                                            const uint16_t col_src1,
                                            const uint16_t pre_rshift,
                                            const uint16_t out_scale,
                                            const uint16_t post_rshift,
                                            u8_t * out);

q7_t *riscv_nn_mat_mul_kernel_u8_q7_2sft_unroll4(const q7_t * src1,
                                            const u8_t * src2,
                                            const uint16_t out_tensor_ch,
                                            const uint16_t col_src1,
                                            const uint16_t pre_rshift,
                                            const uint16_t out_scale,
                                            const uint16_t post_rshift,
                                            q7_t * out);

q15_t *riscv_nn_mat_mul_kernel_u8_q15_2sft_unroll4(const q7_t * src1,
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

int32_t riscv_nn_vec_mat_mult_t_s8_v3(const q7_t *lhs,
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
                                    const int32_t activation_max,
                                    const int32_t out_addr_offset);

int32_t vec_mat_mult_acc_t_s8_s16(const q7_t *lhs,
                                  const q7_t *rhs,
                                  const q31_t *bias,
                                  q15_t *dst,
                                  const int32_t lhs_offset,   //value is in the range of [-127, 128]
                                  const int32_t rhs_offset,   //value is in the range of [-127, 128]
                                  const int32_t dst_offset,   //value is in the range of [-128, 127]
                                  const int32_t dst_multiplier,
                                  const int32_t dst_shift,
                                  const int32_t rhs_cols,
                                  const int32_t rhs_rows,
                                  const int32_t activation_min,
                                  const int32_t activation_max,
                                  const int32_t batch);

q15_t *riscv_nn_mat_mult_kernel_s16(const q7_t *ker_wt,
                                    const q15_t *in_tensor,
                                    const int32_t output_ch,
                                    const int32_t *out_shift,
                                    const int32_t *out_mult,
                                    const int32_t act_min,
                                    const int32_t act_max,
                                    const int32_t col_count,
                                    const int64_t *const out_bias,
                                    q15_t *out_tensor);

q15_t *riscv_nn_mat_mult_kernel_s16_acc_s64(const q7_t *ker_wt,
                                            const q15_t *in_tensor,
                                            const int32_t output_ch,
                                            const int32_t *out_shift,
                                            const int32_t *out_mult,
                                            const int32_t act_min,
                                            const int32_t act_max,
                                            const int32_t col_count,
                                            const int64_t *const out_bias,
                                            q15_t *out_tensor);

int32_t riscv_nn_vec_mat_mult_t_s16(const int16_t *lhs,
                                const int8_t *rhs,
                                const int64_t *bias,
                                int16_t *dst,
                                // const int32_t lhs_offset,
                                // const int32_t rhs_offset,
                                // const int32_t dst_offset,
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

//----- sub-functions for lstm_begin -----
void lstm_update_cell_state_and_output_s16_s8(const int32_t cell_state_scale,
                                              int16_t *cell_state,
                                              riscv_nn_lstm_context *scratch_buffers,
                                              const riscv_nn_scaling hidden_scaling,
                                              const int32_t hidden_offset,
                                              int8_t *output_state,
                                              const int n_batch,
                                              const int n_cell,
                                              const int n_output,
                                              int8_t* output);

void lstm_update_cell_state_s16(const int32_t n_block,
                                const int32_t cell_state_scale,
                                int16_t *cell_state,
                                const int16_t *input_gate,
                                const int16_t *forget_gate,
                                const int16_t *cell_gate);

void lstm_update_output_s16_s8(const int n_batch,
                               const int n_cell,
                               int16_t *cell_state,
                               const int32_t cell_state_scale,
                               const int16_t *output_gate,
                               const riscv_nn_scaling hidden_scaling,
                               const int32_t hidden_offset,
                               int8_t *output_state,
                               int16_t *cell_gate_scratch);

void lstm_calculate_gate_s8_s16(const int8_t *input,
                                const int8_t *input_to_gate_weights,
                                const int32_t *input_to_gate_bias,
                                const riscv_nn_scaling input_to_gate_scaling,
                                const int8_t *output_state,
                                const int8_t *recurrent_to_gate_weights,
                                const int32_t *recurrent_to_gate_bias,
                                const riscv_nn_scaling recurrent_to_gate,
                                const int32_t n_batch,
                                const int32_t n_input,
                                const int32_t n_output,
                                const int32_t n_cell,
                                const riscv_nn_activation_fun activation_type,
                                int16_t *gate);

int lstm_step_s8(const int8_t *input,
                 const int8_t *input_to_input_weight,
                 const int8_t *input_to_forget_weight,
                 const int8_t *input_to_cell_weight,
                 const int8_t *input_to_output_weight,
                 const int8_t *recurrent_to_input_weight,
                 const int8_t *recurrent_to_forget_weight,
                 const int8_t *recurrent_to_cell_weight,
                 const int8_t *recurrent_to_output_weight,
                 const riscv_nn_lstm_params *lstm,
                 const int n_batch,
                 const int n_cell,
                 const int n_input,
                 const int n_output,
                 int8_t *output_state,
                 int16_t *cell_state,
                 int8_t *output,
                 riscv_nn_lstm_context *scratch_buffers);
//----- sub-functions for lstm_end -----

#ifdef __riscv_zfh
float16_t *riscv_nn_mat_mul_kernel_fp16_unroll4(const float16_t * src1,
                                            const float16_t * src2,
                                            const uint16_t out_tensor_ch,
                                            const uint16_t col_src1,
                                            const float16_t * bias,
                                            float16_t * out);

float16_t *riscv_nn_mat_mul_kernel_f16(const float16_t * src1,
                                            const float16_t * src2,
                                            const uint16_t out_tensor_ch,
                                            const uint16_t col_src1,
                                            const float16_t * bias,
                                            float16_t * out);

float16_t *riscv_nn_mat_mul_kernel_tiling_f16(const float16_t * src1,
                                          const float16_t * src2,
                                          float16_t * out,
                                          const uint16_t row,
                                          const uint16_t col,
                                          const uint16_t col2,
                                          const float16_t * bias,
                                          const uint16_t tiling_size);

void riscv_nn_mat_mul_kernel_tiling_transpose_f16(const float16_t * src,
                                                  float16_t * dst,
                                                  const long row,
                                                  const long col);
#endif
/**
 *   * @}
 */
#ifdef __cplusplus
}
#endif

#endif
