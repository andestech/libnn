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

#ifndef __RISCV_NN_CONCATENATION_H__
#define __RISCV_NN_CONCATENATION_H__

#ifdef __cplusplus
extern    "C"
{
#endif

#include "riscv_math_types.h"

/**
 * @defgroup Concatenation Concatenation Functions
 * @brief The concatenation functions are used to concatenate the tensor along
 *        the specified axis.
 *
 * @{
 */

/**
 * @brief       This function concatenates the signed 8-bit integer input tensor
 *              along the w-axis with the output tensor.
 * @param[in]   in_tensor       pointer of the input tensor
 * @param[in]   in_tensor_x     x dimension of the input tensor
 * @param[in]   in_tensor_y     y dimension of the input tensor
 * @param[in]   in_tensor_z     z dimension of the input tensor
 * @param[in]   in_tensor_w     w dimension of the input tensor
 * @param[out]  out_tensor      pointer of the output tensor
 * @param[in]   out_offset_w    offset value to be added to the w-axis of the
 *                              output tensor before the concatenation
 * @return      None
 *
 * @note
 * - The x, y and z dimension of the output tensor will be the same as those of
 *   the input tensor.
 * - The data layout of input and output tensors is assumed to be wzyx.
 */
void riscv_nn_concate_s8_w(const int8_t *in_tensor,
                        const uint16_t in_tensor_x,
                        const uint16_t in_tensor_y,
                        const uint16_t in_tensor_z,
                        const uint16_t in_tensor_w,
                        int8_t *out_tensor,
                        const uint32_t out_offset_w);

/**
 * @brief       This function concatenates the signed 8-bit integer input tensor
 *              along the x-axis with the output tensor.
 * @param[in]   in_tensor       pointer of the input tensor
 * @param[in]   in_tensor_x     x dimension of the input tensor
 * @param[in]   in_tensor_y     y dimension of the input tensor
 * @param[in]   in_tensor_z     z dimension of the input tensor
 * @param[in]   in_tensor_w     w dimension of the input tensor
 * @param[in]   out_tensor      pointer of the output tensor
 * @param[out]  out_tensor_x    x dimension of the output tensor
 * @param[in]   out_offset_x    offset value to be added to the x-axis of the
 *                              output tensor before the concatenation
 * @return      None
 *
 * @note
 * - The y, z and w dimensions of the output tensor will be the same as those of
 *   the input tensor.
 * - The data layout of input and output tensors is assumed to be wzyx.
 */
void riscv_nn_concate_s8_x(const int8_t *in_tensor,
                        const uint16_t in_tensor_x,
                        const uint16_t in_tensor_y,
                        const uint16_t in_tensor_z,
                        const uint16_t in_tensor_w,
                        int8_t *out_tensor,
                        const uint16_t out_tensor_x,
                        const uint32_t out_offset_x);

/**
 * @brief       This function concatenates the signed 8-bit integer input tensor
 *              along the y-axis with the output tensor.
 * @param[in]   in_tensor       pointer of the input tensor
 * @param[in]   in_tensor_x     x dimension of the input tensor
 * @param[in]   in_tensor_y     y dimension of the input tensor
 * @param[in]   in_tensor_z     z dimension of the input tensor
 * @param[in]   in_tensor_w     w dimension of the input tensor
 * @param[out]  out_tensor      pointer of the output tensor
 * @param[in]   out_tensor_y    y dimension of the output tensor
 * @param[in]   out_offset_y    offset value to be added to the y-axis of the
 *                              output tensor before the concatenation
 * @return      None
 *
 * @note
 * - The x, z and w dimensions of the output tensor will be the same as those of
 *   the input tensor.
 * - The data layout of input and output tensors is assumed to be wzyx.
 */
void riscv_nn_concate_s8_y(const int8_t *in_tensor,
                        const uint16_t in_tensor_x,
                        const uint16_t in_tensor_y,
                        const uint16_t in_tensor_z,
                        const uint16_t in_tensor_w,
                        int8_t *out_tensor,
                        const uint16_t out_tensor_y,
                        const uint32_t out_offset_y);

/**
 * @brief       This function concatenates the signed 8-bit integer input tensor
 *              along the z-axis with the output tensor.
 * @param[in]   in_tensor       pointer of the input tensor
 * @param[in]   in_tensor_x     x dimension of the input tensor
 * @param[in]   in_tensor_y     y dimension of the input tensor
 * @param[in]   in_tensor_z     z dimension of the input tensor
 * @param[in]   in_tensor_w     w dimension of the input tensor
 * @param[out]  out_tensor      pointer of the output tensor
 * @param[in]   out_tensor_z    z dimension of the output tensor
 * @param[in]   out_offset_z    offset value to be added to the z-axis of the
 *                              output tensor before the concatenation
 * @return      None
 *
 * @note
 * - The x, y and w dimensions of the output tensor will be the same as those of
 *   the input tensor.
 * - The data layout of input and output tensors is assumed to be wzyx.
 */
void riscv_nn_concate_s8_z(const int8_t *in_tensor,
                        const uint16_t in_tensor_x,
                        const uint16_t in_tensor_y,
                        const uint16_t in_tensor_z,
                        const uint16_t in_tensor_w,
                        int8_t *out_tensor,
                        const uint16_t out_tensor_z,
                        const uint32_t out_offset_z);

#ifdef __riscv_zfh
/**
 * @brief       This function concatenates the half-precision floating-point
 *              input tensor along the w-axis with the output tensor.
 * @param[in]   in_tensor       pointer of the input tensor
 * @param[in]   in_tensor_x     x dimension of the input tensor
 * @param[in]   in_tensor_y     y dimension of the input tensor
 * @param[in]   in_tensor_z     z dimension of the input tensor
 * @param[in]   in_tensor_w     w dimension of the input tensor
 * @param[out]  out_tensor      pointer of the output tensor
 * @param[in]   out_offset_w    offset value to be added to the w-axis of the
 *                              output tensor before the concatenation
 * @return      None
 *
 * @note
 * - The x, y and z dimension of the output tensor will be the same as those of
 *   the input tensor.
 * - The data layout of input and output tensors is assumed to be wzyx.
 */
void riscv_nn_concate_f16_w(const float16_t* in_tensor,
                            const uint16_t in_tensor_x,
                            const uint16_t in_tensor_y,
                            const uint16_t in_tensor_z,
                            const uint16_t in_tensor_w,
                            float16_t* out_tensor,
                            const uint32_t out_offset_w);

/**
 * @brief       This function concatenates the half-precision floating-point
 *              input tensor along the x-axis with the output tensor.
 * @param[in]   in_tensor       pointer of the input tensor
 * @param[in]   in_tensor_x     x dimension of the input tensor
 * @param[in]   in_tensor_y     y dimension of the input tensor
 * @param[in]   in_tensor_z     z dimension of the input tensor
 * @param[in]   in_tensor_w     w dimension of the input tensor
 * @param[out]  out_tensor      pointer of the output tensor
 * @param[in]   out_tensor_x    x dimension of the output tensor
 * @param[in]   out_offset_x    offset value to be added to the x-axis of the
 *                              output tensor before the concatenation
 * @return      None
 *
 * @note
 * - The y, z and w dimensions of the output tensor will be the same as those of
 *   the input tensor.
 * - The data layout of input and output tensors is assumed to be wzyx.
 */
void riscv_nn_concate_f16_x(const float16_t *in_tensor,
                            const uint16_t in_tensor_x,
                            const uint16_t in_tensor_y,
                            const uint16_t in_tensor_z,
                            const uint16_t in_tensor_w,
                            float16_t *out_tensor,
                            const uint16_t out_tensor_x,
                            const uint32_t out_offset_x);

/**
 * @brief       This function concatenates the half-precision floating-point
 *              input tensor along the y-axis with the output tensor.
 * @param[in]   in_tensor       pointer of the input tensor
 * @param[in]   in_tensor_x     x dimension of the input tensor
 * @param[in]   in_tensor_y     y dimension of the input tensor
 * @param[in]   in_tensor_z     z dimension of the input tensor
 * @param[in]   in_tensor_w     w dimension of the input tensor
 * @param[out]  out_tensor      pointer of the output tensor
 * @param[in]   out_tensor_y    y dimension of the output tensor
 * @param[in]   out_offset_y    offset value to be added to the y-axis of the
 *                              output tensor before the concatenation
 * @return      None
 *
 * @note
 * - The x, z and w dimensions of the output tensor will be the same as those of
 *   the input tensor.
 * - The data layout of input and output tensors is assumed to be wzyx.
 */
void riscv_nn_concate_f16_y(const float16_t *in_tensor,
                            const uint16_t in_tensor_x,
                            const uint16_t in_tensor_y,
                            const uint16_t in_tensor_z,
                            const uint16_t in_tensor_w,
                            float16_t *out_tensor,
                            const uint16_t out_tensor_y,
                            const uint32_t out_offset_y);

/**
 * @brief       This function concatenates the half-precision floating-point
 *              input tensor along the z-axis with the output tensor.
 * @param[in]   in_tensor       pointer of the input tensor
 * @param[in]   in_tensor_x     x dimension of the input tensor
 * @param[in]   in_tensor_y     y dimension of the input tensor
 * @param[in]   in_tensor_z     z dimension of the input tensor
 * @param[in]   in_tensor_w     w dimension of the input tensor
 * @param[out]  out_tensor      pointer of the output tensor
 * @param[in]   out_tensor_z    z dimension of the output tensor
 * @param[in]   out_offset_z    offset value to be added to the z-axis of the
 *                              output tensor before the concatenation
 * @return      None
 *
 * @note
 * - The x, y and w dimensions of the output tensor will be the same as those of
 *   the input tensor.
 * - The data layout of input and output tensors is assumed to be wzyx.
 */
void riscv_nn_concate_f16_z(const float16_t *in_tensor,
                            const uint16_t in_tensor_x,
                            const uint16_t in_tensor_y,
                            const uint16_t in_tensor_z,
                            const uint16_t in_tensor_w,
                            float16_t *out_tensor,
                            const uint16_t out_tensor_z,
                            const uint32_t out_offset_z);

#endif  //__riscv_zfh

/**
 *   * @}
 */

#ifdef __cplusplus
}
#endif

#endif
