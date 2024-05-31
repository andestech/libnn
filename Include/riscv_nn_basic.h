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

#ifndef __RISCV_NN_BASIC_OPERATION_H__
#define __RISCV_NN_BASIC_OPERATION_H__

#ifdef __cplusplus
extern    "C"
{
#endif

#include "riscv_math_types.h"

/**
 * @defgroup Basic Basic Functions
 * @brief Basic functions are used to perform elementwise basic arithmetic
 *        operations.
 *
 * @{
 */

/**
 * @brief           This function performs elementwise addition on signed 8-bit
 *                  integer input vectors, applying symmetric quantization to
 *                  the outputs.
 * @param[in]       in_vec1     Pointer to the first input vector
 * @param[in]       in_vec2     Pointer to the second input vector
 * @param[in]       scale1      Pointer to the first scaling vector
 * @param[in]       scale2      Pointer to the second scaling vector
 * @param[in]       size        Number of elements in the input vectors
 * @param[in]       pre_rshift  Right shift amount for the accumulator before
 *                              scaling
 * @param[in]       out_scale   Scaling value for the accumulator
 * @param[in]       post_rshift Right shift amount for the accumulator after
 *                              scaling
 * @param[out]      out_vec     Pointer to the element-wise addition results
 * @return          None
 *
 * @note
 * Figure 1 illustrates the calculation of each element, where rectangles
 * represent inputs or outputs, circles denote arithmetic operations, solid
 * lines indicate the required steps, and dashed lines represents optional ones.
 *
 * @image html riscv_nn_add_s8_sym_algorithm_flowchart.png "Figure 1. riscv_nn_add_s8_sym Algorithm Flowchart" width=600px
 *
 * @b Example:
 * @code
 * #define SIZE 1024
 * uint16_t pre_rshift = 8;        // The addition results of both scaled input
 *                                 // tensors are in the range of 24-bit; thus, the
 *                                 // pre_rshift should be in the range of [0, 24].
 *                                 // Here we scale down the results into 16-bit
 *                                 // range.
 * uint16_t out_scale = 3;         // Scale up the result into 18-bit range.
 * uint16_t post_rshift = 11;      // Scale down the result into 7-bit range.
 *
 * q7_t in_vec1[SIZE] = {...};
 * q7_t in_vec2[SIZE] = {...};
 * int16_t scale1[SIZE] = {...};
 * int16_t scale2[SIZE] = {...};
 * q7_t out_vec[SIZE];
 *
 * riscv_nn_add_s8_sym(in_vec1, in_vec2, scale1, scale2, SIZE, pre_rshift,
 *     out_scale, post_rshift, out_vec);
 * @endcode
 */
void riscv_nn_add_s8_sym(const q7_t * in_vec1,
                        const q7_t * in_vec2,
                        const int16_t * scale1,
                        const int16_t * scale2,
                        const uint32_t size,
                        const uint16_t pre_rshift,
                        const uint16_t out_scale,
                        const uint16_t post_rshift,
                        q7_t * out_vec);

/**
 * @brief           This function performs elementwise addition on signed 8-bit
 *                  integer input vectors, applying symmetric quantization and
 *                  rounding to the outputs.
 * @param[in]       in_vec1     Pointer to the first input vector
 * @param[in]       in_vec2     Pointer to the second input vector
 * @param[in]       scale1      Scaling value for the first input vector. It
 *                              should be in the range of  0 to 2^23.
 * @param[in]       scale2      Scaling value for the second input vector. It
 *                              should be in the range of 0 to 2^23.
 * @param[in]       size        Number of elements in the input vectors
 * @param[in]       pre_rshift  Right shift amount for the accumulator before
 *                              the scaling
 * @param[in]       out_scale   Scaling value for the accumulator
 * @param[in]       post_rshift Right shift amount for the accumulator after the
 *                              scaling
 * @param[out]      out_vec     Pointer to element-wise addition results
 * @return          None
 *
 * @note
 * - The calculation of each element could be represented as Figure 1.
 * - The right shift operations for this function include rounding.
 */
void riscv_nn_add_s8_sym_round(const q7_t * in_vec1,
                            const q7_t * in_vec2,
                            const uint32_t scale1,
                            const uint32_t scale2,
                            const uint32_t size,
                            const uint16_t pre_rshift,
                            const uint32_t out_scale,
                            const uint16_t post_rshift,
                            q7_t * out_vec);

/**
 * @brief           This function performs elementwise addition on signed 8-bit
 *                  integer input vectors, applying asymmetric quantization to
 *                  either the inputs or the outputs.
 * @param[in]       in_vec1     Pointer to the first input vector
 * @param[in]       in_vec2     Pointer to the second input vector
 * @param[in]       in_offset1  Offset value for the first input vector. It
 *                              should be in the range of -127 to 128.
 * @param[in]       in_scale1   Scaling value for the quantization on the first
 *                              input vector
 * @param[in]       in_rshift1  Right shift amount for the quantization on the
 *                              first input vector
 * @param[in]       in_offset2  Offset value for the second input vector. It
 *                              should be in the range of -127 to 128.
 * @param[in]       in_scale2   Scaling value for the quantization on the
 *                              second input vector
 * @param[in]       in_rshift2  Right shift amount for the quantization on the
 *                              second input vector
 * @param[in]       lshift      Left shift amount for the first and second input
 *                              vectors
 * @param[out]      out_vec     Pointer to the element-wise addition results
 * @param[in]       out_offset  Offset value for the output. It should be in
 *                              the range of -128 to 127.
 * @param[in]       out_scale   Scaling value for the quantization on the
 *                              outputs
 * @param[in]       out_rshift  Right shift amount for the quantization on the
 *                              outputs
 * @param[in]       act_min     Minimum value that the outputs are limited to.
 *                              It should be in the range of -128 to 127.
 * @param[in]       act_max     Maximum value that the outputs are limited to.
 *                              It should be in the range of -128 to 127.
 * @param[in]       size        Number of elements in the input vectors
 * @return          This function only returns 0.
 *
 * @note
 * - The calculation of each element could be represented as Figure 2.
 *   @image html riscv_nn_ew_add_s8_asym_algorithm_flowchart.png "Figure 2. riscv_nn_ew_add_s8_asym Algorithm Flowchart" width=600px
 * - The multiplication for in_scale1, in_scale2 and out_scale could be roughly
 *   expressed as:
 *   32b = ((int64_t)32b * 32b) >> 31
 *
 * @b Example:
 * @code
 * #define SIZE 1024
 * int32_t in_offset1 = 16;        // Offset for in_vec1
 * int32_t in_scale1 = (1<<28);    // Scale down in_vec1 by 1/2^3
 * int32_t in_rshift1 = 3;         // Scale down in_vec1 by 1/2^3
 * int32_t in_offset2 = 17;        // Offset for in_vec2
 * int32_t in_scale2 = (1<<28);    // Scale down in_vec2 by 1/2^3
 * int32_t in_rshift2 = 3;         // Scale down in_vec2 by 1/2^3
 * int32_t lshift = 10;            // Scale up the input tensor by 2^10 times
 * int32_t out_offset = 18;        // Offset for the output tensor
 * int32_t out_scale = (1<<30);    // Scale down in_vec2 by 1/2
 * int32_t out_rshift = 4;         // Scale down in_vec2 by 1/2^4
 * int32_t act_min = 0xffffffa3;   // Limit the outputs to the range of
 *                                 // [0xffffffa3, 0x0000005d]
 * int32_t act_max = 0x0000005d;   // Limit the outputs to the range of
 *                                 // [0xffffffa3, 0x0000005d]
 *
 * int8_t in_vec1[SIZE] = {...};
 * int8_t in_vec2[SIZE] = {...};
 * int8_t out_vec[SIZE];
 *
 * riscv_nn_ew_add_s8_asym(in_vec1, in_vec2, in_offset1, in_scale1,
 *     in_rshift1, in_offset2, in_scale2, in_rshift2, lshift, out_vec, out_offset,
 *     out_scale, out_rshift, act_min, act_max, SIZE);
 * @endcode
 */
int riscv_nn_ew_add_s8_asym(const int8_t *in_vec1,
                            const int8_t *in_vec2,
                            const int32_t in_offset1,
                            const int32_t in_scale1,
                            const int32_t in_rshift1,
                            const int32_t in_offset2,
                            const int32_t in_scale2,
                            const int32_t in_rshift2,
                            const int32_t lshift,
                            int8_t *out_vec,
                            const int32_t out_offset,
                            const int32_t out_scale,
                            const int32_t out_rshift,
                            const int32_t act_min,
                            const int32_t act_max,
                            const uint32_t size);
/**
 * @brief           This function performs elementwise addition on signed 16-bit
 *                  integer input vectors, applying asymmetric quantization to
 *                  either the inputs or the outputs.
 * @param[in]       in_vec1     Pointer to the first input vector
 * @param[in]       in_vec2     Pointer to the second input vector
 * @param[in]       in_offset1  Dummy
 * @param[in]       in_scale1   Scaling value for the quantization on first
 *                              input vector
 * @param[in]       in_rshift1  Right shift amount for the quantization on the
 *                              first input vector
 * @param[in]       in_offset2  Dummy
 * @param[in]       in_scale2   Scaling value for the quantization on the
 *                              second input vector
 * @param[in]       in_rshift2  Right shift amount for the quantization on the
 *                              second input vector
 * @param[in]       lshift      Left shift amount for the first and second input
 *                              vectors
 * @param[out]      out_vec     Pointer to the elementwise addition results
 * @param[in]       out_offset  Dummy
 * @param[in]       out_scale   Scaling value for the quantization on the
 *                              outputs
 * @param[in]       out_rshift  Right shift amount for the quantization on the
 *                              outputs
 * @param[in]       act_min     Minimum value that the outputs are limited to.
 *                              It should be in the range of -32768 to 32767.
 * @param[in]       act_max     Maximum value that the outputs are limited to.
 *                              It should be in the range of -32768 to 32767.
 * @param[in]       size        Number of elements in the input vectors
 * @return          This function only returns 0.
 *
 * @note
 *  The calculation of each element could be represented as Figure 2 in which
 *  in_offset1, in_offset2 and out_offset are dummy for this function.
 */

int riscv_nn_ew_add_s16_asym(const int16_t *in_vec1,
                             const int16_t *in_vec2,
                             const int32_t in_offset1,
                             const int32_t in_scale1,
                             const int32_t in_rshift1,
                             const int32_t in_offset2,
                             const int32_t in_scale2,
                             const int32_t in_rshift2,
                             const int32_t lshift,
                             int16_t *out_vec,
                             const int32_t out_offset,
                             const int32_t out_scale,
                             const int32_t out_rshift,
                             const int32_t act_min,
                             const int32_t act_max,
                             const int32_t size);

#ifdef __riscv_zfh
/**
 * @brief           This function performs elementwise addition on
 *                  half-precision floating-point input vectors.
 * @param[in]       in_vec1     Pointer to the first input vector
 * @param[in]       in_vec2     Pointer to the second input vector
 * @param[out]      out_vec     Pointer to element-wise addition results
 * @param[in]       size        Number of elements in the input vectors
 * @return          This function only returns 0.
 */
int riscv_nn_ew_add_f16(const float16_t *in_vec1,
                        const float16_t *in_vec2,
                        float16_t *out_vec,
                        const uint32_t size);
#endif

/**
 * @brief           This function performs elementwise addition on a signed
 *                  8-bit integer input vector and a constant, applying
 *                  asymmetric quantization to either the inputs or the outputs.
 * @param[in]       in_vec      Pointer to the input vector
 * @param[in]       in_const    Constant value to be added. It should be in
 *                              the range of -255 to 255.
 * @param[in]       in_offset   Offset value for the input vector. It should be
 *                              in the range of -127 to 128.
 * @param[in]       in_scale    Scaling value for the quantization on the
 *                              input vector
 * @param[in]       in_rshift   Right shift amount for the quantization on the
 *                              input vector
 * @param[in]       lshift      Left shift amount for the input vector
 * @param[out]      out_vec     Pointer to the output vector
 * @param[in]       out_offset  Offset value for the outputs. It should be in
 *                              the range of -128 to 127.
 * @param[in]       out_scale   Scaling value for the quantization on the
 *                              output vector
 * @param[in]       out_rshift  Right shift amount for the quantization on the
 *                              output vector
 * @param[in]       act_min     Minimum value that the outputs are limited to.
 *                              It should be in the range of -128 to 127.
 * @param[in]       act_max     Maximum value that the outputs are limited to.
 *                              It should be in the range of -128 to 127.
 * @param[in]       size        Number of elements in the input vector
 * @return          This function only returns 0.
 */
int riscv_nn_ew_addc_s8_asym(const int8_t *in_vec,
                             const int32_t in_const,
                             const int32_t in_offset,
                             const int32_t in_scale,
                             const int32_t in_rshift,
                             const int32_t lshift,
                             int8_t *out_vec,
                             const int32_t out_offset,
                             const int32_t out_scale,
                             const int32_t out_rshift,
                             const int32_t act_min,
                             const int32_t act_max,
                             const uint32_t size);

/**
 * @brief           This function performs elementwise multiplication on signed
 *                  8-bit integer input vectors, applying asymmetric
 *                  quantization to the outputs.
 * @param[in]       in_vec1     Pointer to the first input vector
 * @param[in]       in_vec2     Pointer to the second input vector
 * @param[in]       in_offset1  Offset value for the first input vector. It
 *                              should be in the range of -127 to 128.
 * @param[in]       in_offset2  Offset value for the second input vector. It
 *                              should be in the range of -127 to 128.
 * @param[out]      out_vec     Pointer to the element-wise multiplication
 *                              results
 * @param[in]       out_offset  Offset value for the outputs. It should be in
 *                              the range of -128 to 127.
 * @param[in]       out_scale   Scaling value for the quantization on the
 *                              outputs
 * @param[in]       out_shift   Shift amount for the quantization on the
 *                              outputs
 * @param[in]       act_min     Minimum value that the outputs are limited to.
 *                              It should be in the range of -128 to 127.
 * @param[in]       act_max     Maximum value that the outputs are limited to.
 *                              It should be in the range of -128 to 127.
 * @param[in]       size        Number of elements in the input vectors
 * @return          This function returns 0.
 *
 * @note
 * - The calculation of each element could be represented as Figure 3.
 *   @image html riscv_nn_ew_mul_s8_asym_algorithm_flowchart.png "Figure 3. riscv_nn_ew_mul_s8_asym Algorithm Flowchart" width=600px
 * - The multiplication for out_scale could be roughly expressed as:
 *   32b = ((int64_t)32b * 32b) >> 31
 * - During the quantization process, a positive out_shift value is used to left
 *   shift calculation results whereas a negative one is used to right shift.
 *
 * @b Example:
 * @code
 * #define SIZE 1024
 * int32_t in_offset1 = 16;            // Offset for in_vec1
 * int32_t in_offset2 = 17;            // Offset for in_vec2
 * int32_t out_offset = 18;            // Offset for the output tensor
 * int32_t out_scale = (1<<30);        // Scale down the output tensor by 1/2
 * int32_t out_shift = -4;             // Scale down the output tensor by 1/2^4
 * int32_t act_min = 0xffffffa3;       // Limit the outputs to the range of
 *                                     // [0xffffffa3, 0x0000005d]
 * int32_t act_max = 0x0000005d;       // Limit the outputs to the range of
 *                                     // [0xffffffa3, 0x0000005d]
 *
 * int8_t in_vec1[SIZE] = {...};
 * int8_t in_vec2[SIZE] = {...};
 * int8_t out_vec[SIZE];
 *
 * riscv_nn_ew_mul_s8_asym(in_vec1, in_vec2, in_offset1, in_offset2, out_vec,
 *     out_offset, out_scale, out_shift, act_min, act_max, SIZE);
 * @endcode
 */
int riscv_nn_ew_mul_s8_asym(const int8_t *in_vec1,
                            const int8_t *in_vec2,
                            const int32_t in_offset1,
                            const int32_t in_offset2,
                            int8_t *out_vec,
                            const int32_t out_offset,
                            const int32_t out_scale,
                            const int32_t out_shift,
                            const int32_t act_min,
                            const int32_t act_max,
                            const uint32_t size);

/**
 * @brief           This function performs elementwise multiplication on signed
 *                  16-bit integer input/output vectors, applying asymmetric
 *                  quantization to the outputs.
 * @param[in]       in_vec1     Pointer to the first input vector
 * @param[in]       in_vec2     Pointer to the second input vector
 * @param[in]       in_offset1  Dummy
 * @param[in]       in_offset2  Dummy
 * @param[out]      out_vec     Pointer to the elementwise multiplication
 *                              results
 * @param[in]       out_offset  Dummy
 * @param[in]       out_scale   Scaling value for the quantization on the
 *                              outputs
 * @param[in]       out_shift   Shift amount for the quantization on the
 *                              outputs
 * @param[in]       act_min     Minimum value that the outputs are limited to.
 *                              It should be in the range of -32768 to 32767.
 * @param[in]       act_max     Maximum value that the outputs are limited to.
 *                              It should be in the range of -32768 to 32767.
 * @param[in]       size        Number of elements in the input vectors
 * @return          This function only returns 0.
 *
 * @note
 * - The calculation of each element could be represented as Figure 3 in which
 *   in_offset1, in_offset2 and out_offset are dummy for this function.
 * - During the quantization process, a positive out_shift value is used to left
 *   shift calculation results whereas a negative one is used to right shift.
 */
int riscv_nn_ew_mul_s16_asym(const int16_t *in_vec1,
                             const int16_t *in_vec2,
                             const int32_t in_offset1,
                             const int32_t in_offset2,
                             int16_t *out_vec,
                             const int32_t out_offset,
                             const int32_t out_scale,
                             const int32_t out_shift,
                             const int32_t act_min,
                             const int32_t act_max,
                             const int32_t size);
/**
 * @brief           This function performs elementwise multiplication on signed
 *                  16-bit integer input vectors and signed 8-bit integer output
 *                  vectors, applying asymmetric quantization to the outputs.
 * @param[in]       in_vec1     Pointer to the first input vector
 * @param[in]       in_vec2     Pointer to the second input vector
 * @param[out]      out_vec     Pointer to the output vector
 * @param[in]       out_offset  Offset value for the outputs. It should be in
 *                              the range of -128 to 127.
 * @param[in]       out_scale   Scaling value for the quantization on the
*                               outputs
 * @param[in]       out_shift   Shift amount for the quantization on the
 *                              outputs
 * @param[in]       size        Number of elements in the input vectors
 * @return          This function only returns 0.
 *
 * @note
 * During the quantization process, a positive out_shift value is used to left
 * shift calculation results whereas a negative one is used to right shift.
 */
int riscv_nn_ew_mul_s16_s8_asym(const int16_t *in_vec1,
                                const int16_t *in_vec2,
                                int8_t *out_vec,
                                const int32_t out_offset,
                                const int32_t out_scale,
                                const int32_t out_shift,
                                const int32_t size);

/**
 * @brief           This function performs elementwise multiplication on a
 *                  signed 8-bit integer input vector and a constant, applying
 *                  asymmetric quantization to the outputs.
 * @param[in]       in_vec      Pointer to the input vector
 * @param[in]       in_const    Constant value to be added
 * @param[in]       in_offset   Offset value for the input vector. It should be
 *                              in the range of -127 to 128.
 * @param[out]      out_vec     Pointer to the output vector
 * @param[in]       out_offset  Offset value for the outputs. It should be in
 *                              the range of -128 to 127.
 * @param[in]       out_scale   Scaling value for the quantization on the
 *                              output vector
 * @param[in]       out_shift   Shift amount for the quantization on the
 *                              outputs
 * @param[in]       act_min     Minimum value that the outputs are limited to.
 *                              It should be in the range of -128 to 127.
 * @param[in]       act_max     Maximum value that the outputs are limited to.
 *                              It should be in the range of -128 to 127.
 * @param[in]       size        Number of elements in the input vector
 * @return          This function only returns 0.
 *
 * @note
 *  During the quantization process, a positive out_shift value is used to left
 *  shift calculation results whereas a negative one is used to right shift.
 */
int riscv_nn_ew_mulc_s8_asym(const int8_t *in_vec,
                             const int32_t in_const,
                             const int32_t in_offset,
                             int8_t *out_vec,
                             const int32_t out_offset,
                             const int32_t out_scale,
                             const int32_t out_shift,
                             const int32_t act_min,
                             const int32_t act_max,
                             const uint32_t size);

#ifdef __riscv_zfh
/**
 * @brief           This function performs elementwise multiplication on
 *                  half-precision floating-point input vectors.
 * @param[in]       in_vec1     Pointer to the first input vector
 * @param[in]       in_vec2     Pointer to the second input vector
 * @param[out]      out_vec     Pointer to elementwise multiplication results
 * @param[in]       size        Number of elements in the input vectors
 * @return          This function only returns 0.
 */
int riscv_nn_ew_mul_f16(const float16_t *in_vec1,
                        const float16_t *in_vec2,
                        float16_t *out_vec,
                        const uint32_t size);
#endif // __riscv_zfh

/**
 * @brief           This function performs elementwise subtraction of a constant
 *                  from a signed 8-bit integer input vector, applying
 *                  asymmetric quantization to either the inputs or the outputs.
 * @param[in]       in_vec      Pointer to the input vector
 * @param[in]       in_const    Constant value for the subtraction
 * @param[in]       in_offset   Offset value for the input vector. It should be
 *                              in the range of -127 to 128.
 * @param[in]       in_scale    Scaling value for the quantization on the inputs
 * @param[in]       in_rshift   Right shift amount for the quantization on the
 *                              inputs
 * @param[in]       lshift      Left shift amount for the inputs
 * @param[out]      out_vec     Pointer to the output vector
 * @param[in]       out_offset  Offset value for the output. It should be in
 *                              the range of -128 to 127.
 * @param[in]       out_scale   Scaling value for the quantization on the
 *                              outputs
 * @param[in]       out_rshift  Right shift amount for the quantization on the
 *                              outputs
 * @param[in]       act_min     Minimum value that the outputs are limited to.
 *                              It should be in the range of -128 to 127.
 * @param[in]       act_max     Maximum value that the outputs are limited to.
 *                              It should be in the range of -128 to 127.
 * @param[in]       size        Number of elements in the input vector
 * @return          This function only returns 0.
 */
int riscv_nn_ew_rsubc_s8_asym(const int8_t *in_vec,
                              const int32_t in_const,
                              const int32_t in_offset,
                              const int32_t in_scale,
                              const int32_t in_rshift,
                              const int32_t lshift,
                              int8_t *out_vec,
                              const int32_t out_offset,
                              const int32_t out_scale,
                              const int32_t out_rshift,
                              const int32_t act_min,
                              const int32_t act_max,
                              const uint32_t size);

/**
 * @brief           This function performs elementwise subtraction on signed
 *                  8-bit integer input vectors, applying asymmetric
 *                  quantization to either the inputs or the outputs.
 * @param[in]       in_vec1     Pointer to the first input vector
 * @param[in]       in_vec2     Pointer to the second input vector
 * @param[in]       in_offset1  Offset value for the first input vector. It
 *                              should be in the range of -127 to 128.
 * @param[in]       in_scale1   Scaling value for the quantization on the first
 *                              input vector
 * @param[in]       in_rshift1  Right shift amount for the quantization on the
 *                              first input vector
 * @param[in]       in_offset2  Offset value for the second input vector. It
 *                              should be in the range of -127 to 128.
 * @param[in]       in_scale2   Scaling value for the quantization on the
 *                              second input vector
 * @param[in]       in_rshift2  Right shift amount for the quantization on the
 *                              second input vector
 * @param[in]       lshift      Left shift amount for the first and second input
 *                              vectors
 * @param[out]      out_vec     Pointer to the output vector
 * @param[in]       out_offset  Offset value for the output. It should be in
 *                              the range of -128 to 127.
 * @param[in]       out_scale   Scaling value for the quantization on the
 *                              outputs
 * @param[in]       out_rshift  Right shift amount for the quantization on the
 *                              outputs
 * @param[in]       act_min     Minimum value that the outputs are limited to.
 *                              It should be in the range of -128 to 127.
 * @param[in]       act_max     Maximum value that the outputs are limited to.
 *                              It should be in the range of -128 to 127.
 * @param[in]       size        Number of elements in the input vectors
 * @return          This function only returns 0.
 */
int riscv_nn_ew_sub_s8_asym(const int8_t *in_vec1,
                            const int8_t *in_vec2,
                            const int32_t in_offset1,
                            const int32_t in_scale1,
                            const int32_t in_rshift1,
                            const int32_t in_offset2,
                            const int32_t in_scale2,
                            const int32_t in_rshift2,
                            const int32_t lshift,
                            int8_t *out_vec,
                            const int32_t out_offset,
                            const int32_t out_scale,
                            const int32_t out_rshift,
                            const int32_t act_min,
                            const int32_t act_max,
                            const uint32_t size);

/**
 * @brief           This function performs elementwise subtraction of a constant
 *                  from a signed 8-bit integer input vector, applying
 *                  asymmetric quantization to either the inputs or the outputs.
 * @param[in]       in_vec      Pointer to the input vector
 * @param[in]       in_const    Constant value for the subtraction
 * @param[in]       in_offset   Offset value for the first input vector. It
 *                              should be in the range of -127 to 128.
 * @param[in]       in_scale    Scaling value for the quantization on the inputs
 * @param[in]       in_rshift   Right shift amount for the quantization on the
 *                              inputs
 * @param[in]       lshift      Left shift amount for the input vector
 * @param[out]      out_vec     Pointer to the output vector
 * @param[in]       out_offset  Offset value for the output. It should be in
 *                              the range of -128 to 127.
 * @param[in]       out_scale   Scaling value for the quantization on the
 *                              outputs
 * @param[in]       out_rshift  Right shift amount for the quantization on the
 *                              outputs
 * @param[in]       act_min     Minimum value that the outputs are limited to.
 *                              It should be in the range of -128 to 127.
 * @param[in]       act_max     Maximum value that the outputs are limited to.
 *                              It should be in the range of -128 to 127.
 * @param[in]       size        Number of elements in the input vector
 * @return          This function only returns 0.
 */
int riscv_nn_ew_subc_s8_asym(const int8_t *in_vec,
                             const int32_t in_const,
                             const int32_t in_offset,
                             const int32_t in_scale,
                             const int32_t in_rshift,
                             const int32_t lshift,
                             int8_t *out_vec,
                             const int32_t out_offset,
                             const int32_t out_scale,
                             const int32_t out_rshift,
                             const int32_t act_min,
                             const int32_t act_max,
                             const uint32_t size);

/**
 *   * @}
 */

#ifdef __cplusplus
}
#endif

#endif
