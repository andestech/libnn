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

#ifndef __RISCV_NN_SOFTMAX_H__
#define __RISCV_NN_SOFTMAX_H__

#ifdef __cplusplus
extern    "C"
{
#endif

#include "riscv_math_types.h"

/**
 * @defgroup Softmax Softmax Functions
 * @brief   Softmax functions are mathematical functions that calculate
 *          probability distributions for one-dimensional (1D) or
 *          two-dimensional (2D) inputs.
 *
 * @{
 */

/**
 * @brief           This function performs softmax calculations on signed 8-bit
 *                  integer input vectors.
 * @param[in]       in_vec      Pointer to the input vector
 * @param[in]       size        Number of elements in the input vector
 * @param[out]      out_vec     Pointer to the output vector
 * @return          None
 *
 * @note
 *  - Outputs from inputs with very small values will be zero, which may cause
 *    the sum of the dequantized outputs to deviate from 1.
 *  - This is a 2-based softmax function.
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
 * @brief           This function performs softmax calculations on signed 16-bit
 *                  integer input vectors.
 * @param[in]       in_vec      Pointer to the input vector
 * @param[in]       size        Number of elements in the input vector
 * @param[out]      out_vec     Pointer to the output vector
 * @return          None
 *
 * @note
 *  - Outputs from inputs with very small values will be zero, which may cause
 *    the sum of the dequantized outputs to deviate from 1.
 *  - This is a 2-based softmax function.
 */
void riscv_nn_softmax_s16_fast(const q15_t * in_vec,
                            const uint16_t size,
                            q15_t * out_vec);

/**
 * @brief           This function performs softmax calculations on signed 8-bit
 *                  integer input/output tensors using a high-precision
 *                  algorithm.
 * @param[in]       in_tensor       Pointer to the input tensor
 * @param[in]       in_tensor_row   Number of rows in the input tensor
 * @param[in]       in_tensor_col   Number of columns in the input tensor
 * @param[in]       scale           Scaling value for input quantization
 * @param[in]       lshift          Left shift amount for input quantization
 * @param[in]       diff_min        Minimum threshold to perform the quantized
 *                                  exponential operation. The difference can be
 *                                  obtained by subtracting the input from the
 *                                  maximum in row.
 * @param[out]      out_tensor      Pointer to the output tensor
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
 * @param[in]       in_tensor       Pointer to the input tensor
 * @param[in]       in_tensor_row   Number of rows in the input tensor
 * @param[in]       in_tensor_col   Number of columns in the input tensor
 * @param[in]       scale           Scaling value for input quantization
 * @param[in]       lshift          Left shift amount for input quantization
 * @param[in]       diff_min        Minimum threshold to perform the quantized
 *                                  exponential operation. The difference can be
 *                                  obtained by subtracting the input from the
 *                                  maximum in row.
 * @param[out]      out_tensor      Pointer to the output tensor
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
 * @param[in]       in_tensor       Pointer to the input tensor
 * @param[in]       in_tensor_row   Number of rows in the input tensor
 * @param[in]       in_tensor_col   Number of columns in the input tensor
 * @param[in]       scale           Scaling value for input quantization
 * @param[in]       lshift          Left shift amount for input quantization
 * @param[in]       diff_min        Minimum threshold to perform the quantized
 *                                  exponential operation. The difference can be
 *                                  obtained by subtracting the input from the
 *                                  maximum in row.
 * @param[out]      out_tensor      Pointer to the output tensor
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
 * @brief           This function performs softmax calculations on signed 16-bit
 *                  integer input/output tensors using a high-precision
 *                  algorithm.
 * @param[in]       in_tensor       Pointer to the input tensor
 * @param[in]       in_tensor_row   Number of rows in the input tensor
 * @param[in]       in_tensor_col   Number of columns in the input tensor
 * @param[in]       scale           Scaling value for input quantization
 * @param[in]       shift           Left shift amount for input quantization
 * @param[in]       exp_lut         Pointer to the lookup table for exp(x),
                                    where x is uniformly distributed in [10, 0].
 * @param[in]       one_by_one_lut  Pointer to the lookup table for (1/(1+x)),
                                    where x is uniformly distributed in [0, 1].
 * @param[out]      out_tensor      Pointer to the output tensor
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
 * @brief           This function performs softmax calculations on
 *                  half-precision floating-point input vectors.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       size            Number of elements in the input vector
 * @param[out]      out_vec         Pointer to the output vector
 * @return          This function only returns 0.
 */
int32_t riscv_nn_softmax_f16(const float16_t * in_vec,
                            const uint32_t size,
                            float16_t * out_vec);

int32_t riscv_nn_softmax_f16_2pass(const float16_t * in_vec,
                             const uint32_t size,
                             float16_t * out_vec);
#endif


/**
 * @brief           This function performs softmax calculations on
 *                  single-precision floating-point input vectors.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       size            Number of elements in the input vector
 * @param[out]      out_vec         Pointer to the output vector
 * @return          This function only returns 0.
 */
int32_t riscv_nn_softmax_f32(const float32_t * in_vec,
                             const uint32_t size,
                             float32_t * out_vec);

/**
 * @brief           This function performs softmax calculations on
 *                  single-precision floating-point input vectors using a
 *                  two-pass algorithm.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       out_vec         Pointer to the output vector
 * @return          This function only returns 0.
 */
int32_t riscv_nn_softmax_f32_2pass(const float32_t * in_vec,
                             const uint32_t size,
                             float32_t * out_vec);

#ifdef __riscv_zfh
/**
 * @brief           This function applies calculations to each row of a
 *                  two-dimensional, half-precision floating-point buffer.
 * @param[in]       in_buf          Pointer to the input buffer
 * @param[in]       row             Number of rows in the two-dimension buffer
 * @param[in]       col             Number of columns in the two-dimension
 *                                  buffer
 * @param[out]      out_buf         Pointer to the output buffer
 * @param[in]       tmp_buf         Temporary buffer for calculations. It is
 *                                  required when -mext-vector is enabled and
 *                                  its size must be the same as input buffer.
 * @return          This function only returns 0.
 */
int32_t riscv_nn_softmax2d_f16(const float16_t * in_buf,
                             uint32_t row,
                             uint32_t col,
                             float16_t * out_buf,
                             float16_t * tmp_buf);
#endif

/**
 * @brief           This function applies calculations to each row of a
 *                  two-dimensional single-precision floating-point buffer.
 * @param[in]       in_buf          Pointer to the input buffer
 * @param[in]       row             Number of rows in the two-dimension buffer
 * @param[in]       col             Number of columns in the two-dimension
 *                                  buffer
 * @param[out]      out_buf         Pointer to the output buffer
 * @param[in]       tmp_buf         Temporary buffer for calculations. It is
 *                                  required when -mext-vector is enabled and
 *                                  its size must be the same as input buffer.
 * @return          This function only returns 0.
 */
int32_t riscv_nn_softmax2d_f32(const float32_t * in_buf,
                             uint32_t row,
                             uint32_t col,
                             float32_t * out_buf,
                             float32_t * tmp_buf);

/**
 *   * @}
 */
#ifdef __cplusplus
}
#endif

#endif
