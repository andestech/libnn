/******************************************************************************
 * Copyright (C) 2010-2018 Arm Limited or its affiliates. All rights reserved.*
 * Copyright (C) 2018-2023 Andes Technology Corporation. All rights reserved. *
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

#ifndef __RISCV_NN_SOFTMAX_H__
#define __RISCV_NN_SOFTMAX_H__

#ifdef __cplusplus
extern    "C"
{
#endif

#include "riscv_math_types.h"

/**
 * @defgroup Softmax Softmax Functions
 * @brief The softmax functions are exponential functions with base 2.
 *
 * @{
 */

/**
 * @brief           This is a softmax function for signed 8-bit integer input
 *                  vectors.
 * @param[in]       in_vec      pointer of the input vector
 * @param[in]       size        number of elements in the input vector
 * @param[out]      out_vec     pointer of the output vector
 * @return          None
 *
 * @note            The outputs from inputs with very small values will be zero,
 *                  and this makes the summation of dequant outputs far from 1.
 *
 * @b Example:
 * @code
 * #define LENGTH 10
 * q7_t in_data[LENGTH] = {...};
 * q7_t out_data[LENGTH];
 *
 * riscv_nn_softmax_s8_fast(in_data, LENGTH, out_data);
 * @endcode
 */
void riscv_nn_softmax_s8_fast(const q7_t * in_vec,
                            const uint16_t size,
                            q7_t * out_vec);

/**
 * @brief           This is a softmax function for signed 16-bit integer input
 *                  vectors.
 * @param[in]       in_vec      pointer of the input vector
 * @param[in]       size        number of elements in the input vector
 * @param[out]      out_vec     pointer of the output vector
 * @return          None
 *
 * @note            The outputs from inputs with very small values will be zero,
 *                  and this makes the summation of dequant outputs far from 1.
 */
void riscv_nn_softmax_s16_fast(const q15_t * in_vec,
                            const uint16_t size,
                            q15_t * out_vec);

/**
 * @brief           This is a softmax function for signed 8-bit integer input/
 *                  output tensor with high precision algorithm.
 * @param[in]       in_tensor       pointer of the input tensor
 * @param[in]       in_tensor_row   number of rows in the input tensor
 * @param[in]       in_tensor_col   number of columns in the input tensor
 * @param[in]       scale           scaling value for input quantization
 * @param[in]       lshift          left shift amount for input quantization
 * @param[in]       diff_min        minimum threshold to perform the quantized
 *                                  exponential operation. The difference can be
 *                                  obtained by subtracting the input from the
 *                                  maximum in row.
 * @param[out]      out_tensor      pointer of the output tensor
 * @return          None
 */
void riscv_nn_softmax_s8_hp(const int8_t *in_tensor,
                            const int32_t in_tensor_row,
                            const int32_t in_tensor_col,
                            const int32_t scale,
                            const int32_t lshift,
                            const int32_t diff_min,
                            int8_t *out_tensor);

/**
 * @brief           This is a softmax function for signed 8-bit integer input
 *                  tensor and signed 16-bit integer output tensor with high
 *                  precision algorithm.
 * @param[in]       in_tensor       pointer of the input tensor
 * @param[in]       in_tensor_row   number of rows in the input tensor
 * @param[in]       in_tensor_col   number of columns in the input tensor
 * @param[in]       scale           scaling value for input quantization
 * @param[in]       lshift          left shift amount for input quantization
 * @param[in]       diff_min        minimum threshold to perform the quantized
 *                                  exponential operation. The difference can be
 *                                  obtained by subtracting the input from the
 *                                  maximum in row.
 * @param[out]      out_tensor      pointer of the output tensor
 * @return          None
 */
void riscv_nn_softmax_s8_s16_hp(const int8_t *in_tensor,
                                const int32_t in_tensor_row,
                                const int32_t in_tensor_col,
                                const int32_t scale,
                                const int32_t lshift,
                                const int32_t diff_min,
                                int16_t *out_tensor);

/**
 * @brief           This is a softmax function for unsigned 8-bit integer input
 *                  tensor with high precision algorithm.
 * @param[in]       in_tensor       pointer of the input tensor
 * @param[in]       in_tensor_row   number of rows in the input tensor
 * @param[in]       in_tensor_col   number of columns in the input tensor
 * @param[in]       scale           scaling value for input quantization
 * @param[in]       lshift          left shift amount for input quantization
 * @param[in]       diff_min        minimum threshold to perform the quantized
 *                                  exponential operation. The difference can be
 *                                  obtained by subtracting the input from the
 *                                  maximum in row.
 * @param[out]      out_tensor      pointer of the output tensor
 * @return          None
 */
void riscv_nn_softmax_u8_hp(const uint8_t *in_tensor,
                            const int32_t in_tensor_row,
                            const int32_t in_tensor_col,
                            const int32_t scale,
                            const int32_t lshift,
                            const int32_t diff_min,
                            uint8_t *out_tensor);

/**
 * @brief           This is a softmax function for signed 16-bit integer input
 *                  tensor with high precision algorithm.
 * @param[in]       in_tensor       pointer of the input tensor
 * @param[in]       in_tensor_row   number of rows in the input tensor
 * @param[in]       in_tensor_col   number of columns in the input tensor
 * @param[in]       scale           scaling value for input quantization
 * @param[in]       shift           left shift amount for input quantization
 * @param[in]       exp_lut         pointer of the lookup table for exp(x),
                                    where x is uniformly distributed in [10, 0].
 * @param[in]       one_by_one_lut  pointer of the lookup table for (1/(1+x)),
                                    where x is uniformly distributed in [0, 1].
 * @param[out]      out_tensor      pointer of the output tensor
 * @return          This function only returns 0.
 */
int riscv_nn_softmax_s16_hp(const int16_t *in_tensor,
                             const int32_t in_tensor_row,
                             const int32_t in_tensor_col,
                             const int32_t scale,
                             const int32_t shift,
                             const int16_t *exp_lut,
                             const int16_t *one_by_one_lut,
                             int16_t *out_tensor);

#ifdef __riscv_zfh
/**
 * @brief           This is a softmax function for half-precision floating point
 *                  input vectors.
 * @param[in]       in_vec          pointer of the input vector
 * @param[in]       size            number of elements in the input vector
 * @param[out]      out_vec         pointer of the output vector
 * @return          This function only returns 0.
 */
int32_t riscv_nn_softmax_f16(const float16_t * in_vec,
                            const uint32_t size,
                            float16_t * out_vec);
#endif

/**
 *   * @}
 */
#ifdef __cplusplus
}
#endif

#endif
