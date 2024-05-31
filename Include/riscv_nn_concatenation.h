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

#ifndef __RISCV_NN_CONCATENATION_H__
#define __RISCV_NN_CONCATENATION_H__

#ifdef __cplusplus
extern    "C"
{
#endif

#include "riscv_math_types.h"

/**
 * @defgroup Concatenation Concatenation Functions
 * @brief The concatenation functions are used to concatenate or split the
 *        tensor along the specified axis.
 *
 * @{
 */

/**
 * @brief       This function concatenates a signed 8-bit integer input tensor
 *              along the w-axis with an output tensor.
 * @param[in]   in_tensor       Pointer to the input tensor
 * @param[in]   in_tensor_x     X dimension of the input tensor
 * @param[in]   in_tensor_y     Y dimension of the input tensor
 * @param[in]   in_tensor_z     Z dimension of the input tensor
 * @param[in]   in_tensor_w     W dimension of the input tensor
 * @param[out]  out_tensor      Pointer to the output tensor
 * @param[in]   out_offset_w    Offset value to be added to the w-axis of the
 *                              output tensor before the concatenation
 * @return      None
 *
 * @note
 * - The x, y and z dimension of the output tensor will be the same as those of
 *   the input tensor.
 * - The assumed data layout for both input and output tensors is wzyx.
 */
void riscv_nn_concate_s8_w(const int8_t *in_tensor,
                        const uint16_t in_tensor_x,
                        const uint16_t in_tensor_y,
                        const uint16_t in_tensor_z,
                        const uint16_t in_tensor_w,
                        int8_t *out_tensor,
                        const uint32_t out_offset_w);

/**
 * @brief       This function concatenates a signed 8-bit integer input tensor
 *              along the x-axis with an output tensor.
 * @param[in]   in_tensor       Pointer to the input tensor
 * @param[in]   in_tensor_x     X dimension of the input tensor
 * @param[in]   in_tensor_y     Y dimension of the input tensor
 * @param[in]   in_tensor_z     Z dimension of the input tensor
 * @param[in]   in_tensor_w     W dimension of the input tensor
 * @param[in]   out_tensor      Pointer to the output tensor
 * @param[out]  out_tensor_x    X dimension of the output tensor
 * @param[in]   out_offset_x    Offset value to be added to the x-axis of the
 *                              output tensor before the concatenation
 * @return      None
 *
 * @note
 * - The y, z and w dimensions of the output tensor will be the same as those of
 *   the input tensor.
 * - The assumed data layout for both input and output tensors is wzyx.
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
 * @brief       This function concatenates a signed 8-bit integer input tensor
 *              along the y-axis with an output tensor.
 * @param[in]   in_tensor       Pointer to the input tensor
 * @param[in]   in_tensor_x     X dimension of the input tensor
 * @param[in]   in_tensor_y     Y dimension of the input tensor
 * @param[in]   in_tensor_z     Z dimension of the input tensor
 * @param[in]   in_tensor_w     W dimension of the input tensor
 * @param[out]  out_tensor      Pointer to the output tensor
 * @param[in]   out_tensor_y    Y dimension of the output tensor
 * @param[in]   out_offset_y    Offset value to be added to the y-axis of the
 *                              output tensor before the concatenation
 * @return      None
 *
 * @note
 * - The x, z and w dimensions of the output tensor will be the same as those of
 *   the input tensor.
 * - The assumed data layout for both input and output tensors is wzyx.
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
 * @brief       This function concatenates a signed 8-bit integer input tensor
 *              along the z-axis with an output tensor.
 * @param[in]   in_tensor       Pointer to the input tensor
 * @param[in]   in_tensor_x     X dimension of the input tensor
 * @param[in]   in_tensor_y     Y dimension of the input tensor
 * @param[in]   in_tensor_z     Z dimension of the input tensor
 * @param[in]   in_tensor_w     W dimension of the input tensor
 * @param[out]  out_tensor      Pointer to the output tensor
 * @param[in]   out_tensor_z    Z dimension of the output tensor
 * @param[in]   out_offset_z    Offset value to be added to the z-axis of the
 *                              output tensor before the concatenation
 * @return      None
 *
 * @note
 * - The x, y and w dimensions of the output tensor will be the same as those of
 *   the input tensor.
 * - The assumed data layout for both input and output tensors is wzyx.
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
 * @brief       This function concatenates a half-precision floating-point input
 *              tensor along the w-axis with an output tensor.
 * @param[in]   in_tensor       Pointer to the input tensor
 * @param[in]   in_tensor_x     X dimension of the input tensor
 * @param[in]   in_tensor_y     Y dimension of the input tensor
 * @param[in]   in_tensor_z     Z dimension of the input tensor
 * @param[in]   in_tensor_w     W dimension of the input tensor
 * @param[out]  out_tensor      Pointer to the output tensor
 * @param[in]   out_offset_w    Offset value to be added to the w-axis of the
 *                              output tensor before the concatenation
 * @return      None
 *
 * @note
 * - The x, y and z dimension of the output tensor will be the same as those of
 *   the input tensor.
 * - The assumed data layout for both input and output tensors is wzyx.
 */
void riscv_nn_concate_f16_w(const float16_t* in_tensor,
                            const uint16_t in_tensor_x,
                            const uint16_t in_tensor_y,
                            const uint16_t in_tensor_z,
                            const uint16_t in_tensor_w,
                            float16_t* out_tensor,
                            const uint32_t out_offset_w);

/**
 * @brief       This function concatenates a half-precision floating-point input
 *              tensor along the x-axis with an output tensor.
 * @param[in]   in_tensor       Pointer to the input tensor
 * @param[in]   in_tensor_x     X dimension of the input tensor
 * @param[in]   in_tensor_y     Y dimension of the input tensor
 * @param[in]   in_tensor_z     Z dimension of the input tensor
 * @param[in]   in_tensor_w     W dimension of the input tensor
 * @param[out]  out_tensor      Pointer to the output tensor
 * @param[in]   out_tensor_x    X dimension of the output tensor
 * @param[in]   out_offset_x    Offset value to be added to the x-axis of the
 *                              output tensor before the concatenation
 * @return      None
 *
 * @note
 * - The y, z and w dimensions of the output tensor will be the same as those of
 *   the input tensor.
 * - The assumed data layout for both input and output tensors is wzyx.
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
 * @brief       This function concatenates a half-precision floating-point input
 *              tensor along the y-axis with an output tensor.
 * @param[in]   in_tensor       Pointer to the input tensor
 * @param[in]   in_tensor_x     X dimension of the input tensor
 * @param[in]   in_tensor_y     Y dimension of the input tensor
 * @param[in]   in_tensor_z     Z dimension of the input tensor
 * @param[in]   in_tensor_w     W dimension of the input tensor
 * @param[out]  out_tensor      Pointer to the output tensor
 * @param[in]   out_tensor_y    Y dimension of the output tensor
 * @param[in]   out_offset_y    Offset value to be added to the y-axis of the
 *                              output tensor before the concatenation
 * @return      None
 *
 * @note
 * - The x, z and w dimensions of the output tensor will be the same as those of
 *   the input tensor.
 * - The assumed data layout for both input and output tensors is wzyx.
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
 * @brief       This function concatenates a half-precision floating-point input
 *              tensor along the z-axis with an output tensor.
 * @param[in]   in_tensor       Pointer to the input tensor
 * @param[in]   in_tensor_x     X dimension of the input tensor
 * @param[in]   in_tensor_y     Y dimension of the input tensor
 * @param[in]   in_tensor_z     Z dimension of the input tensor
 * @param[in]   in_tensor_w     W dimension of the input tensor
 * @param[out]  out_tensor      Pointer to the output tensor
 * @param[in]   out_tensor_z    Z dimension of the output tensor
 * @param[in]   out_offset_z    Offset value to be added to the z-axis of the
 *                              output tensor before the concatenation
 * @return      None
 *
 * @note
 * - The x, y and w dimensions of the output tensor will be the same as those of
 *   the input tensor.
 * - The assumed data layout for both input and output tensors is wzyx.
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
 * @brief       This function splits a signed 8-bit integer input tensor along
 *              the w-axis with an input tensor.
 * @param[in]   in_tensor       Pointer to the input tensor
 * @param[in]   in_tensor_x     X dimension of the input tensor
 * @param[in]   in_tensor_y     Y dimension of the input tensor
 * @param[in]   in_tensor_z     Z dimension of the input tensor
 * @param[in]   in_tensor_w     W dimension of the input tensor
 * @param[out]  out_tensor      Pointer to the output tensor
 * @param[in]   split_len_w     The length to be splited in the w-axis
 * @param[in]   in_offset       The offset between the starting point of the
 *                              input tenosr and the position to be split
 * @return      None
 *
 * @note
 * - The x, y and z dimension of the output tensor will be the same as those of
 *   the input tensor.
 * - The assumed data layout for both input and output tensors is wzyx.
 */
void riscv_nn_split_s8_w(const int8_t *in_tensor,
                         const uint32_t in_tensor_x,
                         const uint32_t in_tensor_y,
                         const uint32_t in_tensor_z,
                         const uint32_t in_tensor_w,
                         int8_t *out_tensor,
                         const uint32_t split_len_w,
                         const uint32_t in_offset);

/**
 * @brief       This function splits a signed 8-bit integer input tensor along
 *              the x-axis with an input tensor.
 * @param[in]   in_tensor       Pointer to the input tensor
 * @param[in]   in_tensor_x     X dimension of the input tensor
 * @param[in]   in_tensor_y     Y dimension of the input tensor
 * @param[in]   in_tensor_z     Z dimension of the input tensor
 * @param[in]   in_tensor_w     W dimension of the input tensor
 * @param[out]  out_tensor      Pointer to the output tensor
 * @param[in]   split_len_x     The length to be splited in the x-axis
 * @param[in]   in_offset       The offset between the starting point of the
 *                              input tenosr and the position to be split
 * @return      None
 *
 * @note
 * - The y, z and w dimensions of the output tensor will be the same as those of
 *   the input tensor.
 * - The assumed data layout for both input and output tensors is wzyx.
 */
void riscv_nn_split_s8_x(const int8_t *in_tensor,
                         const uint32_t in_tensor_x,
                         const uint32_t in_tensor_y,
                         const uint32_t in_tensor_z,
                         const uint32_t in_tensor_w,
                         int8_t *out_tensor,
                         const uint32_t split_len_x,
                         const uint32_t in_offset);

/**
 * @brief       This function splits a signed 8-bit integer input tensor along
 *              the y-axis with an input tensor.
 * @param[in]   in_tensor       Pointer to the input tensor
 * @param[in]   in_tensor_x     X dimension of the input tensor
 * @param[in]   in_tensor_y     Y dimension of the input tensor
 * @param[in]   in_tensor_z     Z dimension of the input tensor
 * @param[in]   in_tensor_w     W dimension of the input tensor
 * @param[out]  out_tensor      Pointer to the output tensor
 * @param[in]   split_len_y     The length to be splited in the y-axis
 * @param[in]   in_offset       The offset between the starting point of the
 *                              input tenosr and the position to be split
 * @return      None
 *
 * @note
 * - The x, z and w dimensions of the output tensor will be the same as those of
 *   the input tensor.
 * - The assumed data layout for both input and output tensors is wzyx.
 */
void riscv_nn_split_s8_y(const int8_t *in_tensor,
                         const uint32_t in_tensor_x,
                         const uint32_t in_tensor_y,
                         const uint32_t in_tensor_z,
                         const uint32_t in_tensor_w,
                         int8_t *out_tensor,
                         const uint32_t split_len_y,
                         const uint32_t in_offset);

/**
 * @brief       This function splits a signed 8-bit integer input tensor along
 *              the z-axis with an input tensor.
 * @param[in]   in_tensor       Pointer to the input tensor
 * @param[in]   in_tensor_x     X dimension of the input tensor
 * @param[in]   in_tensor_y     Y dimension of the input tensor
 * @param[in]   in_tensor_z     Z dimension of the input tensor
 * @param[in]   in_tensor_w     W dimension of the input tensor
 * @param[out]  out_tensor      Pointer to the output tensor
 * @param[in]   split_len_z     The length to be splited in the z-axis
 * @param[in]   in_offset       The offset between the starting point of the
 *                              input tenosr and the position to be splited
 * @return      None
 *
 * @note
 * - The x, y and w dimensions of the output tensor will be the same as those of
 *   the input tensor.
 * - The assumed data layout for both input and output tensors is wzyx.
 */
void riscv_nn_split_s8_z(const int8_t *in_tensor,
                         const uint32_t in_tensor_x,
                         const uint32_t in_tensor_y,
                         const uint32_t in_tensor_z,
                         const uint32_t in_tensor_w,
                         int8_t *out_tensor,
                         const uint32_t split_len_z,
                         const uint32_t in_offset);

/**
 *   * @}
 */

#ifdef __cplusplus
}
#endif

#endif
