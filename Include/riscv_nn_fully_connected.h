/******************************************************************************
 * Copyright (C) 2010-2018 Arm Limited or its affiliates. All rights reserved.*
 * Copyright (C) 2018-2024 Andes Technology Corporation. All rights reserved. *
 *                                                                            *
 * SPDX-License-Identifier: Apache-2.0                                        *
 *                                                                            *
 * Licensed under the Apache License, Version 2.0 (the License); you may      *
 * not use this file except in compliance with the License.                   *
 * You may obtain a copy of the License at                                    *
 *                                                                            *
 * www.apache.org/licenses/LICENSE-2.0                                        *
 *                                                                            *
 * Unless required by applicable law or agreed to in writing, software        *
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT    *
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.           *
 * See the License for the specific language governing permissions and        *
 * limitations under the License.                                             *
 ******************************************************************************/

/** @file*/

#ifndef __RISCV_NN_FULLY_CONNECTED_H__
#define __RISCV_NN_FULLY_CONNECTED_H__

#ifdef __cplusplus
extern    "C"
{
#endif

#include "riscv_math_types.h"

/**
 * @defgroup FullyConnect Fully-Connected Layer Functions
 * @brief Fully-connected layer functions multiply the input vector by a weight
 *        matrix and add a bias, if any, to the results.
 *        The supported combinations of input vector and weight matrix are
 *        (signed 8-bit integer, signed 8-bit integer), (unsigned 8-bit integer,
 *        signed 8-bit integer), (signed 16-bit integer, signed 8-bit integer),
 *        (signed 16-bit integer, signed 16-bit integer), and (half-precision
 *        floating-point, half-precision floating-point).
 *
 * @{
 */

/**
 * @brief           This function performs calculation on signed 8-bit integers
 *                  for inputs, applying shift-based quantization to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       bias_lshift     Left shift amount for the bias
 * @param[in]       out_rshift      Right shift amount for the output
 * @param[in]       bias            Pointer to the bias vector
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       in_tmp_buf      Dummy
 * @return          This function only returns 0.
 *
 * @b Example:
 * @code
 * #define IN_SIZE 2048
 * #define OUT_SIZE 256
 * #define BIAS_LSHIFT 9        //Scale up the bias by 2^9
 * #define OUT_RSHIFT 9         //Scale down the outputs by 1/2^9
 *
 * q7_t in_vec[IN_SIZE] = {...};;
 * q7_t wt_mat[IN_SIZE * OUT_SIZE] {...};
 * q7_t bias[OUT_SIZE] = {...};
 * q7_t out_vec[OUT_SIZE];
 *
 * riscv_nn_fc_s8_s8_s8_sft_bias(in_vec, wt_mat, IN_SIZE, OUT_SIZE, BIAS_LSHIFT,
 *     OUT_RSHIFT, bias, out_vec, NULL);
 * @endcode
 */
int32_t riscv_nn_fc_s8_s8_s8_sft_bias(const q7_t * in_vec,
                                    const q7_t * wt_mat,
                                    const uint16_t size,
                                    const uint16_t wt_row_num,
                                    const uint16_t bias_lshift,
                                    const uint16_t out_rshift,
                                    const q7_t * bias,
                                    q7_t * out_vec,
                                    q15_t * in_tmp_buf);

/**
 * @brief           This function performs interleaved multiplication on signed
 *                  8-bit integers for inputs, applying shift-based quantization
 *                  to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       bias_lshift     Left shift amount for the bias
 * @param[in]       out_rshift      Right shift amount for the output
 * @param[in]       bias            Pointer to the bias vector
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       in_tmp_buf      Temporary buffer for the input vector. It is
 *                                  required when -mext-vector is enabled and
 *                                  its size must be "2 * size".
 * @return          This function only returns 0.
 *
 * @note
 * The input vector is multiplied by a weight matrix, which is in interleaved
 * format and obtained from riscv_nn_fc_s8_wt_converter.
 */
int32_t riscv_nn_fc_s8_s8_s8_sft_bias_fast(const q7_t * in_vec,
                                        const q7_t * wt_mat,
                                        const uint16_t size,
                                        const uint16_t wt_row_num,
                                        const uint16_t bias_lshift,
                                        const uint16_t out_rshift,
                                        const q7_t * bias,
                                        q7_t * out_vec,
                                        q15_t * in_tmp_buf);

/**
 * @brief           This function performs calculation on signed 16-bit integers
 *                  for inputs, applying shift-based quantization to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       bias_lshift     Left shift amount for the bias
 * @param[in]       out_rshift      Right shift amount for the output
 * @param[in]       bias            Pointer to the bias
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       tmp_buf         Dummy
 * @return          This function only returns 0.
 */
int32_t riscv_nn_fc_s16_s16_s16_sft_bias(const q15_t * in_vec,
                                        const q15_t * wt_mat,
                                        const uint16_t size,
                                        const uint16_t wt_row_num,
                                        const uint16_t bias_lshift,
                                        const uint16_t out_rshift,
                                        const q15_t * bias,
                                        q15_t * out_vec,
                                        q15_t * tmp_buf);

/**
 * @brief           This function performs interleaved multiplication on signed
 *                  16-bit integers for inputs, applying shift-based
 *                  quantization to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       bias_lshift     Left shift amount for the bias
 * @param[in]       out_rshift      Right shift amount for the output
 * @param[in]       bias            Pointer to the bias
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       in_tmp_buf      Temporary buffer for the input vector. It is
 *                                  required when -mext-vector is enabled and
 *                                  its size must be 4 * size.
 * @return          This function only returns 0.
 *
 *
 * @note
 * The input vector is multiplied by a weight matrix, which is in interleaved
 * format and obtained from riscv_nn_fc_s16_wt_converter.
 */
int32_t riscv_nn_fc_s16_s16_s16_sft_bias_fast(const q15_t * in_vec,
                                            const q15_t * wt_mat,
                                            const uint16_t size,
                                            const uint16_t wt_row_num,
                                            const uint16_t bias_lshift,
                                            const uint16_t out_rshift,
                                            const q15_t * bias,
                                            q15_t * out_vec,
                                            q15_t * in_tmp_buf);

/**
 * @brief           This function multiplies a signed 16-bit integer input
 *                  vector by a signed 8-bit integer weight matrix, and applies
 *                  shift-based quantization to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       bias_lshift     Left shift amount for the bias
 * @param[in]       out_rshift      Right shift amount for the output
 * @param[in]       bias            Pointer to the bias
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       tmp_buf         Dummy
 * @return          This function only returns 0.
 */
int32_t riscv_nn_fc_mat_vec_s16_s16_s8_sft_bias(const q15_t * in_vec,
                                                const q7_t * wt_mat,
                                                const uint16_t size,
                                                const uint16_t wt_row_num,
                                                const uint16_t bias_lshift,
                                                const uint16_t out_rshift,
                                                const q7_t * bias,
                                                q15_t * out_vec,
                                                q15_t * tmp_buf);

/**
 * @brief           This function multiplies a signed 16-bit integer input
 *                  vector by a signed 8-bit integer weight matrix in
 *                  interleaved format, then applies shift-based quantization to
 *                  the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       bias_lshift     Left shift amount for the bias
 * @param[in]       out_rshift      Right shift amount for the output
 * @param[in]       bias            Pointer to the bias
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       tmp_buf         Dummy
 * @return          This function only returns 0.
 *
 * @note
 * The input vector is multiplied by a weight matrix, which is in interleaved
 * format and obtained from riscv_nn_fc_mat_vec_s8_wt_converter.
 */
int32_t riscv_nn_fc_mat_vec_s16_s16_s8_sft_bias_fast(const q15_t * in_vec,
                                                    const q7_t * wt_mat,
                                                    const uint16_t size,
                                                    const uint16_t wt_row_num,
                                                    const uint16_t bias_lshift,
                                                    const uint16_t out_rshift,
                                                    const q7_t * bias,
                                                    q15_t * out_vec,
                                                    q15_t * tmp_buf);

/**
 * @brief           This function performs calculation on signed 8-bit integers
 *                  for both inputs and outputs, incorporating bias inputs and
 *                  applying symmetric quantization to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       pre_rshift      Right shift amount for the output before the
 *                                  scaling
 * @param[in]       out_scale       Scaling value for the output
 * @param[in]       post_rshift     Right shift amount for the output after the
 *                                  scaling
 * @param[in]       bias            Pointer to the bias vector
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       in_tmp_buf      Temporary buffer for the input vector. It is
 *                                  required when -mext-dsp is enabled and its
 *                                  size must be "size".
 * @return          This function only returns 0.
 *
 * @note
 * The outputs will be two-stage shifted before being stored, for example:
 * out = ((out >> pre_rshift) * out_scale) >> post_rshift.
 */
int32_t riscv_nn_fc_s8_s8_s8_sym_bias(const q7_t * in_vec,
                                    const q7_t * wt_mat,
                                    const uint16_t size,
                                    const uint16_t wt_row_num,
                                    const uint16_t pre_rshift,
                                    const uint16_t out_scale,
                                    const uint16_t post_rshift,
                                    const q31_t * bias,
                                    q7_t * out_vec,
                                    q15_t * in_tmp_buf);

/**
 * @brief           This function performs calculation on signed 8-bit integers
 *                  for inputs and signed 16-bit integers for outputs,
 *                  incorporating bias inputs and applying symmetric
 *                  quantization to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       pre_rshift      Right shift amount for the output before
 *                                  scaling
 * @param[in]       out_scale       Scaling value for the output
 * @param[in]       post_rshift     Right shift amount for the output after
 *                                  scaling
 * @param[in]       bias            Pointer to the bias vector
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       in_tmp_buf      Temporary buffer for the input vector. It is
 *                                  required when -mext-dsp is enabled and its
 *                                  size must be "size".
 * @return          This function only returns 0.
 *
 * @note
 * The outputs will be two-stage shifted before being stored, for example:
 * out = ((out >> pre_rshift) * out_scale) >> post_rshift.
 */
int32_t riscv_nn_fc_s8_s16_s8_sym_bias(const q7_t * in_vec,
                                    const q7_t * wt_mat,
                                    const uint16_t size,
                                    const uint16_t wt_row_num,
                                    const uint16_t pre_rshift,
                                    const uint16_t out_scale,
                                    const uint16_t post_rshift,
                                    const q31_t * bias,
                                    q15_t * out_vec,
                                    q15_t * in_tmp_buf);

/**
 * @brief           This function performs calculation on unsigned 8-bit
 *                  integers for both inputs and outputs, incorporating bias
 *                  inputs and applying symmetric quantization to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       pre_rshift      Right shift amount for the output before
 *                                  scaling
 * @param[in]       out_scale       Scaling value for the output
 * @param[in]       post_rshift     Right shift amount for the output after
 *                                  scaling
 * @param[in]       bias            Pointer to the bias vector
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       in_tmp_buf      Temporary buffer for the input vector. It is
 *                                  required when -mext-dsp is enabled and its
 *                                  size must be "size".
 * @return          This function only returns 0.
 *
 * @note
 * The outputs will be two-stage shifted before being stored, for example:
 * out = ((out >> pre_rshift) * out_scale) >> post_rshift.
 */
int32_t riscv_nn_fc_u8_u8_s8_sym_bias(const u8_t * in_vec,
                                    const q7_t * wt_mat,
                                    const uint16_t size,
                                    const uint16_t wt_row_num,
                                    const uint16_t pre_rshift,
                                    const uint16_t out_scale,
                                    const uint16_t post_rshift,
                                    const q31_t * bias,
                                    u8_t * out_vec,
                                    q15_t * in_tmp_buf);

/**
 * @brief           This function performs calculation on unsigned 8-bit
 *                  integers for inputs and signed 8-bit integers for outputs,
 *                  incorporating bias inputs and applying symmetric
 *                  quantization to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       pre_rshift      Right shift amount for the output before
 *                                  scaling
 * @param[in]       out_scale       Scaling value for the output
 * @param[in]       post_rshift     Right shift amount for the output after
 *                                  scaling
 * @param[in]       bias            Pointer to the bias vector
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       in_tmp_buf      Temporary buffer for input vector. It is
 *                                  required when -mext-dsp is enabled and its
 *                                  size must be "size".
 * @return          This function only returns 0.
 *
 * @note
 * The outputs will be two-stage shifted before being stored, for example:
 * out = ((out >> pre_rshift) * out_scale) >> post_rshift.
 */
int32_t riscv_nn_fc_u8_s8_s8_sym_bias(const u8_t * in_vec,
                                    const q7_t * wt_mat,
                                    const uint16_t size,
                                    const uint16_t wt_row_num,
                                    const uint16_t pre_rshift,
                                    const uint16_t out_scale,
                                    const uint16_t post_rshift,
                                    const q31_t * bias,
                                    q7_t * out_vec,
                                    q15_t * in_tmp_buf);

/**
 * @brief           This function performs calculation on unsigned 8-bit
 *                  integers for inputs and signed 16-bit integers for outputs,
 *                  incorporating bias inputs and applying symmetric
 *                  quantization to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       pre_rshift      Right shift amount for the output before
 *                                  scaling
 * @param[in]       out_scale       Scaling value for the output
 * @param[in]       post_rshift     Right shift amount for the output after
 *                                  scaling
 * @param[in]       bias            Pointer to the bias vector
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       in_tmp_buf      Temporary buffer for the input vector. It is
 *                                  required when -mext-dsp is enabled and its
 *                                  size must be "size".
 * @return          This function only returns 0.
 *
 * @note
 * The outputs will be two-stage shifted before being stored, for example:
 * out = ((out >> pre_rshift) * out_scale) >> post_rshift.
 */
int32_t riscv_nn_fc_u8_s16_s8_sym_bias(const u8_t * in_vec,
                                    const q7_t * wt_mat,
                                    const uint16_t size,
                                    const uint16_t wt_row_num,
                                    const uint16_t pre_rshift,
                                    const uint16_t out_scale,
                                    const uint16_t post_rshift,
                                    const q31_t * bias,
                                    q15_t * out_vec,
                                    q15_t * in_tmp_buf);

/**
 * @brief           This function performs calculation on signed 8-bit integers
 *                  for both inputs and outputs, applying symmetric quantization
 *                  to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       pre_rshift      Right shift amount for the output before
 *                                  scaling
 * @param[in]       out_scale       Scaling value for the output
 * @param[in]       post_rshift     Right shift amount for the output after
 *                                  scaling
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       in_tmp_buf      Temporary buffer for the input vector. It is
 *                                  required when -mext-dsp is enabled and its
 *                                  size must be "size".
 * @return          This function only returns 0.
 *
 * @note
 * The outputs will be two-stage shifted before being stored, for example:
 * out = ((out >> pre_rshift) * out_scale) >> post_rshift.
 */
int32_t riscv_nn_fc_s8_s8_s8_sym(const q7_t * in_vec,
                                const q7_t * wt_mat,
                                const uint16_t size,
                                const uint16_t wt_row_num,
                                const uint16_t pre_rshift,
                                const uint16_t out_scale,
                                const uint16_t post_rshift,
                                q7_t * out_vec,
                                q15_t * in_tmp_buf);

/**
 * @brief           This function performs calculation on signed 8-bit integers
 *                  for inputs and signed 16-bit integers for outputs, applying
 *                  symmetric quantization to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       pre_rshift      Right shift amount for the output before
 *                                  scaling
 * @param[in]       out_scale       Scaling value for the output
 * @param[in]       post_rshift     Right shift amount for the output after
 *                                  scaling
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       in_tmp_buf      Temporary buffer for the input vector. It is
 *                                  required when -mext-dsp is enabled and its
 *                                  size must be "size".
 * @return          This function only returns 0.
 *
 * @note
 * The outputs will be two-stage shifted before being stored, for example:
 * out = ((out >> pre_rshift) * out_scale) >> post_rshift.
 */
int32_t riscv_nn_fc_s8_s16_s8_sym(const q7_t * in_vec,
                                const q7_t * wt_mat,
                                const uint16_t size,
                                const uint16_t wt_row_num,
                                const uint16_t pre_rshift,
                                const uint16_t out_scale,
                                const uint16_t post_rshift,
                                q15_t * out_vec,
                                q15_t * in_tmp_buf);

/**
 * @brief           This function performs calculation on unsigned 8-bit
 *                  integers for both inputs and outputs, applying symmetric
 *                  quantization to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       pre_rshift      Right shift amount for the output before
 *                                  scaling
 * @param[in]       out_scale       Scaling value for the output
 * @param[in]       post_rshift     Right shift amount for the output after
 *                                  scaling
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       in_tmp_buf      Temporary buffer for the input vector. It is
 *                                  required when -mext-dsp is enabled and its
 *                                  size must be "size".
 * @return          This function only returns 0.
 *
 * @note
 * The outputs will be two-stage shifted before being stored, for example:
 * out = ((out >> pre_rshift) * out_scale) >> post_rshift.
 */
int32_t riscv_nn_fc_u8_u8_s8_sym(const u8_t * in_vec,
                                const q7_t * wt_mat,
                                const uint16_t size,
                                const uint16_t wt_row_num,
                                const uint16_t pre_rshift,
                                const uint16_t out_scale,
                                const uint16_t post_rshift,
                                u8_t * out_vec,
                                q15_t * in_tmp_buf);

/**
 * @brief           This function performs calculation on unsigned 8-bit
 *                  integers for inputs and signed 8-bit integers for outputs,
 *                  applying symmetric quantization to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       pre_rshift      Right shift amount for the output before
 *                                  scaling
 * @param[in]       out_scale       Scaling value for the output
 * @param[in]       post_rshift     Right shift amount for the output after
 *                                  scaling
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       in_tmp_buf      Temporary buffer for the input vector. It is
 *                                  required when -mext-dsp is enabled and its
 *                                  size must be "size".
 * @return          This function only returns 0.
 *
 * @note
 * The outputs will be two-stage shifted before being stored, for example:
 * out = ((out >> pre_rshift) * out_scale) >> post_rshift.
 */
int32_t riscv_nn_fc_u8_s8_s8_sym(const u8_t * in_vec,
                                const q7_t * wt_mat,
                                const uint16_t size,
                                const uint16_t wt_row_num,
                                const uint16_t pre_rshift,
                                const uint16_t out_scale,
                                const uint16_t post_rshift,
                                q7_t * out_vec,
                                q15_t * in_tmp_buf);

/**
 * @brief           This function performs calculation on unsigned 8-bit
 *                  integers for inputs and signed 16-bit integers for outputs,
 *                  applying symmetric quantization to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       pre_rshift      Right shift amount for the output before
 *                                  scaling
 * @param[in]       out_scale       Scaling value for the output
 * @param[in]       post_rshift     Right shift amount for the output after
 *                                  scaling
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       in_tmp_buf      Temporary buffer for the input vector. It is
 *                                  required when -mext-dsp is enabled and its
 *                                  size must be "size".
 * @return          This function only returns 0.
 *
 * @note
 * The outputs will be two-stage shifted before being stored, for example:
 * out = ((out >> pre_rshift) * out_scale) >> post_rshift.
 */
int32_t riscv_nn_fc_u8_s16_s8_sym(const u8_t * in_vec,
                                const q7_t * wt_mat,
                                const uint16_t size,
                                const uint16_t wt_row_num,
                                const uint16_t pre_rshift,
                                const uint16_t out_scale,
                                const uint16_t post_rshift,
                                q15_t * out_vec,
                                q15_t * in_tmp_buf);

/**
 * @brief           This function performs interleaved multiplication on signed
 *                  8-bit integers for both inputs and outputs, incorporating
 *                  bias inputs and applying symmetric quantization to the
 *                  outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       pre_rshift      Right shift amount for the output before
 *                                  scaling
 * @param[in]       out_scale       Scaling value for the output
 * @param[in]       post_rshift     Right shift amount for the output after
 *                                  scaling
 * @param[in]       bias            Pointer to the bias vector
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       in_tmp_buf      Temporary buffer for the input vector. It is
 *                                  required when -mext-vector is enabled and
 *                                  its size must be "2 * size".
 * @return          This function only returns 0.
 *
 * @note
 *  - The input vector is multiplied by a weight matrix, which is in interleaved
 *    format and obtained from riscv_nn_fc_s8_wt_converter.
 *  - The outputs will be two-stage shifted before being stored, for example:
 *    out = ((out >> pre_rshift) * out_scale) >> post_rshift.
 */
int32_t riscv_nn_fc_s8_s8_s8_sym_bias_fast(const q7_t * in_vec,
                                        const q7_t * wt_mat,
                                        const uint16_t size,
                                        const uint16_t wt_row_num,
                                        const uint16_t pre_rshift,
                                        const uint16_t out_scale,
                                        const uint16_t post_rshift,
                                        const q31_t * bias,
                                        q7_t * out_vec,
                                        q15_t * in_tmp_buf);

/**
 * @brief           This function performs interleaved multiplication on signed
 *                  8-bit integers for inputs and signed 16-bit integers for
 *                  outputs, incorporating bias inputs and applying symmetric
 *                  quantization to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       pre_rshift      Right shift amount for the output before
 *                                  scaling
 * @param[in]       out_scale       Scaling value for the output
 * @param[in]       post_rshift     Right shift amount for the output after
 *                                  scaling
 * @param[in]       bias            Pointer to the bias vector
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       in_tmp_buf      Temporary buffer for the input vector. It is
 *                                  required when -mext-vector is enabled and
 *                                  its size must be "2 * size".
 * @return          This function only returns 0.
 *
 * @note
 *  - The input vector is multiplied by a weight matrix, which is in interleaved
 *    format and obtained from riscv_nn_fc_s8_wt_converter.
 *  - The outputs will be two-stage shifted before being stored, for example:
 *    out = ((out >> pre_rshift) * out_scale) >> post_rshift.
 */
int32_t riscv_nn_fc_s8_s16_s8_sym_bias_fast(const q7_t * in_vec,
                                            const q7_t * wt_mat,
                                            const uint16_t size,
                                            const uint16_t wt_row_num,
                                            const uint16_t pre_rshift,
                                            const uint16_t out_scale,
                                            const uint16_t post_rshift,
                                            const q31_t * bias,
                                            q15_t * out_vec,
                                            q15_t * in_tmp_buf);

/**
 * @brief           This function performs interleaved multiplication on
 *                  unsigned 8-bit integers for both inputs and outputs,
 *                  incorporating bias inputs and applying symmetric
 *                  quantization to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       pre_rshift      Right shift amount for the output before
 *                                  scaling
 * @param[in]       out_scale       Scaling value for the output
 * @param[in]       post_rshift     Right shift amount for the output after
 *                                  scaling
 * @param[in]       bias            Pointer to the bias vector
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       in_tmp_buf      Temporary buffer for the input vector. It is
 *                                  required when -mext-vector is enabled and
 *                                  its size must be "2 * size".
 * @return          This function only returns 0.
 *
 * @note
 *  - The input vector is multiplied by a weight matrix, which is in interleaved
 *    format and obtained from riscv_nn_fc_s8_wt_converter.
 *  - The outputs will be two-stage shifted before being stored, for example:
 *    out = ((out >> pre_rshift) * out_scale) >> post_rshift.
 */
int32_t riscv_nn_fc_u8_u8_s8_sym_bias_fast(const u8_t * in_vec,
                                        const q7_t * wt_mat,
                                        const uint16_t size,
                                        const uint16_t wt_row_num,
                                        const uint16_t pre_rshift,
                                        const uint16_t out_scale,
                                        const uint16_t post_rshift,
                                        const q31_t * bias,
                                        u8_t * out_vec,
                                        q15_t * in_tmp_buf);

/**
 * @brief           This function performs interleaved multiplication on
 *                  unsigned 8-bit integers for inputs and signed 8-bit integers
 *                  for outputs, incorporating bias inputs and applying
 *                  symmetric quantization to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       pre_rshift      Right shift amount for the output before
 *                                  scaling
 * @param[in]       out_scale       Scaling value for the output
 * @param[in]       post_rshift     Right shift amount for the output after
 *                                  scaling
 * @param[in]       bias            Pointer to the bias vector
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       in_tmp_buf      Temporary buffer for the input vector. It is
 *                                  required when -mext-vector is enabled and
 *                                  its size must be "2 * size".
 * @return          This function only returns 0.
 *
 * @note
 *  - The input vector is multiplied by a weight matrix, which is in interleaved
 *    format and obtained from riscv_nn_fc_s8_wt_converter.
 *  - The outputs will be two-stage shifted before being stored, i.e.,
 *    out = ((out >> pre_rshift) * out_scale) >> post_rshift.
 */
int32_t riscv_nn_fc_u8_s8_s8_sym_bias_fast(const u8_t * in_vec,
                                        const q7_t * wt_mat,
                                        const uint16_t size,
                                        const uint16_t wt_row_num,
                                        const uint16_t pre_rshift,
                                        const uint16_t out_scale,
                                        const uint16_t post_rshift,
                                        const q31_t * bias,
                                        q7_t * out_vec,
                                        q15_t * in_tmp_buf);

/**
 * @brief           This performs interleaved multiplication on unsigned 8-bit
 *                  integers for inputs and signed 16-bit integers for outputs,
 *                  incorporating bias inputs and applying symmetric
 *                  quantization to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       pre_rshift      Right shift amount for the output before
 *                                  scaling
 * @param[in]       out_scale       Value of scaling for the output
 * @param[in]       post_rshift     Right shift amount for the output after
 *                                  scaling
 * @param[in]       bias            Pointer to the bias vector
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       in_tmp_buf      Temporary buffer for the input vector. It is
 *                                  required when -mext-vector is enabled and
 *                                  its size must be "2 * size".
 * @return          This function only returns 0.
 *
 * @note
 *  - The input vector is multiplied by a weight matrix, which is in interleaved
 *    format and obtained from riscv_nn_fc_s8_wt_converter.
 *  - The outputs will be two-stage shifted before being stored, for example:
 *    out = ((out >> pre_rshift) * out_scale) >> post_rshift.
 */
int32_t riscv_nn_fc_u8_s16_s8_sym_bias_fast(const u8_t * in_vec,
                                        const q7_t * wt_mat,
                                        const uint16_t size,
                                        const uint16_t wt_row_num,
                                        const uint16_t pre_rshift,
                                        const uint16_t out_scale,
                                        const uint16_t post_rshift,
                                        const q31_t * bias,
                                        q15_t * out_vec,
                                        q15_t * in_tmp_buf);

/**
 * @brief           This function performs interleaved multiplication on signed
 *                  8-bit integers for both inputs and outputs, applying
 *                  symmetric quantization to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       pre_rshift      Right shift amount for the output before
 *                                  scaling
 * @param[in]       out_scale       Scaling value for the output
 * @param[in]       post_rshift     Right shift amount for the output after
 *                                  scaling
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       in_tmp_buf      Temporary buffer for the input vector. It is
 *                                  required when -mext-vector is enabled and
 *                                  its size must be "2 * size".
 * @return          This function only returns 0.
 *
 * @note
 *  - The input vector is multiplied by a weight matrix, which is in interleaved
 *    format and obtained from riscv_nn_fc_s8_wt_converter.
 *  - The outputs will be two-stage shifted before being stored, for example:
 *    out = ((out >> pre_rshift) * out_scale) >> post_rshift.
 */
int32_t riscv_nn_fc_s8_s8_s8_sym_fast(const q7_t * in_vec,
                                    const q7_t * wt_mat,
                                    const uint16_t size,
                                    const uint16_t wt_row_num,
                                    const uint16_t pre_rshift,
                                    const uint16_t out_scale,
                                    const uint16_t post_rshift,
                                    q7_t * out_vec,
                                    q15_t * in_tmp_buf);

/**
 * @brief           This is a fully connected layer function for signed 8-bit
 *                  integer inputs and signed 16-bit integer outputs with
 *                  interleaved multiplication and symmetric quantization on the
 *                  outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       pre_rshift      Right shift amount for the output before the
 *                                  scaling
 * @param[in]       out_scale       Scaling value for the output
 * @param[in]       post_rshift     Right shift amount for the output after the
 *                                  scaling
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       in_tmp_buf      Temporary buffer for the input vector. It is
 *                                  required when -mext-vector is enabled and
 *                                  its size must be "2 * size".
 * @return          This function only returns 0.
 *
 * @note
 *  - The input vector is multiplied by a weight matrix, which is in interleaved
 *    format and obtained from riscv_nn_fc_s8_wt_converter.
 *  - The outputs will be two-stage shifted before being stored, for example:
 *    out = ((out >> pre_rshift) * out_scale) >> post_rshift.
 */
int32_t riscv_nn_fc_s8_s16_s8_sym_fast(const q7_t * in_vec,
                                    const q7_t * wt_mat,
                                    const uint16_t size,
                                    const uint16_t wt_row_num,
                                    const uint16_t pre_rshift,
                                    const uint16_t out_scale,
                                    const uint16_t post_rshift,
                                    q15_t * out_vec,
                                    q15_t * in_tmp_buf);

/**
 * @brief           This function performs interleaved multiplication on
 *                  unsigned 8-bit integers for both inputs and outputs,
 *                  applying symmetric quantization to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       pre_rshift      Right shift amount for the output before
 *                                  scaling
 * @param[in]       out_scale       Scaling value for the output
 * @param[in]       post_rshift     Right shift amount for the output after
 *                                  scaling
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       in_tmp_buf      Temporary buffer for the input vector. It is
 *                                  required when -mext-vector is enabled and
 *                                  its size must be "2 * size".
 * @return          This function only returns 0.
 *
 * @note
 *  - The input vector is multiplied by a weight matrix, which is in interleaved
 *    format and obtained from riscv_nn_fc_s8_wt_converter.
 *  - The outputs will be two-stage shifted before being stored, for example:
 *    out = ((out >> pre_rshift) * out_scale) >> post_rshift.
 */
int32_t riscv_nn_fc_u8_u8_s8_sym_fast(const u8_t * in_vec,
                                    const q7_t * wt_mat,
                                    const uint16_t size,
                                    const uint16_t wt_row_num,
                                    const uint16_t pre_rshift,
                                    const uint16_t out_scale,
                                    const uint16_t post_rshift,
                                    u8_t * out_vec,
                                    q15_t * in_tmp_buf);

/**
 * @brief           This function performs interleaved multiplication on
 *                  unsigned 8-bit integers for inputs and signed 8-bit integers
 *                  for outputs, applying symmetric quantization to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       pre_rshift      Right shift amount for the output before
 *                                  scaling
 * @param[in]       out_scale       Scaling value for the output
 * @param[in]       post_rshift     Right shift amount for the output after
 *                                  scaling
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       in_tmp_buf      Temporary buffer for the input vector. It is
 *                                  required when -mext-vector is enabled and
 *                                  its size must be "2 * size".
 * @return          This function only returns 0.
 *
 * @note
 *  - The input vector is multiplied by a weight matrix, which is in interleaved
 *    format and obtained from riscv_nn_fc_s8_wt_converter.
 *  - The outputs will be two-stage shifted before being stored, for example:
 *    out = ((out >> pre_rshift) * out_scale) >> post_rshift.
 */
int32_t riscv_nn_fc_u8_s8_s8_sym_fast(const u8_t * in_vec,
                                    const q7_t * wt_mat,
                                    const uint16_t size,
                                    const uint16_t wt_row_num,
                                    const uint16_t pre_rshift,
                                    const uint16_t out_scale,
                                    const uint16_t post_rshift,
                                    q7_t * out_vec,
                                    q15_t * in_tmp_buf);

/**
 * @brief           This function performs interleaved multiplication on
 *                  unsigned 8-bit integers for inputs and signed 16-bit
 *                  integers for outputs, applying symmetric quantization to the
 *                  outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       pre_rshift      Right shift amount for the output before
 *                                  scaling
 * @param[in]       out_scale       Scaling value for the output
 * @param[in]       post_rshift     Right shift amount for the output after
 *                                  scaling
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       in_tmp_buf      Temporary buffer for the input vector. It is
 *                                  required when -mext-vector is enabled and
 *                                  its size must be "2 * size".
 * @return          This function only returns 0.
 *
 * @note
 *  - The input vector is multiplied by a weight matrix, which is in interleaved
 *    format and obtained from riscv_nn_fc_s8_wt_converter.
 *  - The outputs will be two-stage shifted before being stored, for example:
 *    out = ((out >> pre_rshift) * out_scale) >> post_rshift.
 */
int32_t riscv_nn_fc_u8_s16_s8_sym_fast(const u8_t * in_vec,
                                    const q7_t * wt_mat,
                                    const uint16_t size,
                                    const uint16_t wt_row_num,
                                    const uint16_t pre_rshift,
                                    const uint16_t out_scale,
                                    const uint16_t post_rshift,
                                    q15_t * out_vec,
                                    q15_t * in_tmp_buf);

/**
 * @brief           This is a weight converter for fully-connected layer
 *                  functions that use signed 8-bit weight data and are named
 *                  with a _fast suffix.
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[out]      wt_mat_out      Pointer to the weight matrix stored in
 *                                  specific ordering
 * @return          None
 */
void riscv_nn_fc_s8_wt_converter(const q7_t *wt_mat,
                                const uint32_t size,
                                const uint32_t wt_row_num,
                                q7_t *wt_mat_out);

/**
 * @brief           This is a weight converter for fully-connected layer
 *                  functions that use signed 16-bit weight data and are named
 *                  with a _fast suffix.
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[out]      wt_mat_out      Pointer to the weight matrix stored in
 *                                  specific ordering
 * @return          None
 */
void riscv_nn_fc_s16_wt_converter(const q15_t *wt_mat,
                                const uint32_t size,
                                const uint32_t wt_row_num,
                                q15_t *wt_mat_out);

/**
 * @brief           This is a weight converter for
 *                  riscv_nn_fc_mat_vec_s16_s16_s8_sft_bias_fast.
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[out]      wt_mat_out      Pointer to the weight matrix stored in
 *                                  specific ordering
 * @return          None
 */
void riscv_nn_fc_mat_vec_s8_wt_converter(const q7_t *wt_mat,
                                        const uint32_t size,
                                        const uint32_t wt_row_num,
                                        q7_t *wt_mat_out);

/**
 * @brief           This function performs calculation on signed 8-bit integers
 *                  for inputs, incorporating bias inputs and applying
 *                  asymmetric quantization to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the transposed weight matrix
 * @param[in]       in_vec_col      Number of columns in the input vector (or
 *                                  transposed weight matrix)
 * @param[in]       wt_mat_row      Number of rows in the transposed weight
 *                                  matrix
 * @param[in]       in_vec_batch    Size of the input vector batches
 * @param[in]       in_offset       Offset value to be added to the input tensor
 *                                  . It should be in the range of -127 to 128.
 * @param[in]       wt_offset       Offset value to be added to the weight. It
 *                                  should be in the range of -127 to 128.
 * @param[in]       out_scale       Scaling value for the quantization on the
 *                                  outputs
 * @param[in]       out_shift       Shift amount for the quantization on the
 *                                  outputs
 * @param[in]       out_offset      Offset value to be added to the output
 *                                  tensor. It should be in the range of -128 to
 *                                  127.
 * @param[in]       bias            Pointer to the bias vector
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       act_min         Minimum value that the output tensor is
 *                                  limited to. It should be in the range of
 *                                  -128 to 127.
 * @param[in]       act_max         Maximum value that the output tensor is
 *                                  limited to. It should be in the range of
 *                                  -128 to 127.
 * @param[in]       tmp_buf         Dummy
 * @return          This function only returns 0.
 *
 * @note
 *  - bias could be a null pointer as the bias vector is optional for this
 *    function.
 *  - During the quantization process, a positive out_shift value is used to
 *    left shift calculation results whereas a negative one is used to right
 *    shift.
 */
int32_t riscv_nn_fc_s8_s8_s8_asym_bias(const int8_t *in_vec,
                                    const int8_t *wt_mat,
                                    const uint16_t in_vec_col,
                                    const uint16_t wt_mat_row,
                                    const uint16_t in_vec_batch,
                                    const int32_t in_offset,
                                    const int32_t wt_offset,
                                    const int32_t out_scale,
                                    const int32_t out_shift,
                                    const int32_t out_offset,
                                    const int32_t *bias,
                                    int8_t *out_vec,
                                    const int32_t act_min,
                                    const int32_t act_max,
                                    q15_t *tmp_buf);

/**
 * @brief           This function calculates the required size (in bytes) for
 *                  the temporary buffer needed for riscv_nn_fc_s8_s8_s8_asym_bias.
 * @param[in]       in_vec_col      Number of columns in the input vector (or
 *                                  transposed weight matrix)
 * @return          This function returns the required size of the temporary
 *                  buffer.
 */
int32_t riscv_nn_fc_s8_s8_s8_asym_bias_get_buffer_size(const uint16_t in_vec_col);

/**
 * @brief           This function performs calculation on signed 16-bit integers
 *                  for inputs, incorporating bias inputs and applying
 *                  asymmetric quantization to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the transposed weight matrix
 * @param[in]       in_vec_col      Number of columns in the input vector (or
 *                                  transposed weight matrix)
 * @param[in]       wt_mat_row      Number of rows in the transposed weight
 *                                  matrix
 * @param[in]       in_vec_batch    Size of input vector batches
 * @param[in]       in_offset       Dummy
 * @param[in]       wt_offset       Dummy
 * @param[in]       out_scale       Scaling value for the quantization on the
 *                                  outputs
 * @param[in]       out_shift       Shift amount for the quantization on the
 *                                  outputs
 * @param[in]       out_offset      Dummy
 * @param[in]       bias            Pointer to the bias vector
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       act_min         Minimum value that the output tensor is
 *                                  limited to. It should be in the range of
 *                                  -32768 to 32767.
 * @param[in]       act_max         Maximum value that the output tensor is
 *                                  limited to. It should be in the range of
 *                                  -32768 to 32767.
 * @param[in]       tmp_buf         Dummy
 * @return          This function only returns 0.
 *
 * @note
 *  - bias could be a null pointer as the bias vector is optional for this
 *    function.
 *  - During the quantization process, a positive out_shift value is used to
 *    left shift calculation results whereas a negative one is used to right
 *    shift.
 */
int32_t riscv_nn_fc_s16_s16_s8_asym_bias(const int16_t *in_vec,
    const int8_t *wt_mat,
    const int32_t in_vec_col,
    const int32_t wt_mat_row,
    const int32_t in_vec_batch,
    const int32_t in_offset,
    const int32_t wt_offset,
    const int32_t out_scale,
    const int32_t out_shift,
    const int32_t out_offset,
    const int64_t *bias,
    int16_t *out_vec,
    const int32_t act_min,
    const int32_t act_max,
    int16_t *tmp_buf);

/**
 * @brief           This function calculates the required size (in bytes) for
 *                  the temporary buffer needed for riscv_nn_fc_s16_s16_s8_asym_bias.
 * @param[in]       in_vec_col      Number of columns in the input vector (or
 *                                  transposed weight matrix)
 * @return          This function returns the required size of the temporary
 *                  buffer.
 */
int32_t riscv_nn_fc_s16_s16_s8_asym_bias_get_buffer_size(const uint16_t in_vec_col);

#ifdef __riscv_zfh
/**
 * @brief           This function performs calculation on half-precision
 *                  floating-point inputs and outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       wt_mat          Pointer to the weight matrix
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       wt_row_num      Number of rows in the weight matrix
 * @param[in]       bias            Pointer to the bias vector
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       tmp_buf         Dummy
 * @return          This function only returns 0.
 */
int32_t riscv_nn_fc_f16_f16_f16_bias(const float16_t * in_vec,
                                    const float16_t * wt_mat,
                                    const uint16_t size,
                                    const uint16_t wt_row_num,
                                    const float16_t * bias,
                                    float16_t * out_vec,
                                    float16_t * tmp_buf);
#endif

/**
 *   * @}
 */

#ifdef __cplusplus
}
#endif

#endif
