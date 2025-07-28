/******************************************************************************
 * Copyright (C) 2010-2025 Arm Limited or its affiliates. All rights reserved.*
 * Copyright (C) 2018-2025 Andes Technology Corporation. All rights reserved. *
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
void riscv_nn_concate_s8_w(const int8_t * in_tensor,
                           const uint16_t in_tensor_x,
                           const uint16_t in_tensor_y,
                           const uint16_t in_tensor_z,
                           const uint16_t in_tensor_w,
                           int8_t * out_tensor,
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
void riscv_nn_concate_s8_x(const int8_t * in_tensor,
                           const uint16_t in_tensor_x,
                           const uint16_t in_tensor_y,
                           const uint16_t in_tensor_z,
                           const uint16_t in_tensor_w,
                           int8_t * out_tensor,
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
void riscv_nn_concate_s8_y(const int8_t * in_tensor,
                          const uint16_t in_tensor_x,
                          const uint16_t in_tensor_y,
                          const uint16_t in_tensor_z,
                          const uint16_t in_tensor_w,
                          int8_t * out_tensor,
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
void riscv_nn_concate_s8_z(const int8_t * in_tensor,
                           const uint16_t in_tensor_x,
                           const uint16_t in_tensor_y,
                           const uint16_t in_tensor_z,
                           const uint16_t in_tensor_w,
                           int8_t * out_tensor,
                           const uint16_t out_tensor_z,
                           const uint32_t out_offset_z);

/**
 * @brief       This function concatenates a signed 16-bit integer input tensor
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
void riscv_nn_concate_s16_w(const int16_t * in_tensor,
                            const uint16_t in_tensor_x,
                            const uint16_t in_tensor_y,
                            const uint16_t in_tensor_z,
                            const uint16_t in_tensor_w,
                            int16_t * out_tensor,
                            const uint32_t out_offset_w);

/**
 * @brief       This function concatenates a signed 16-bit integer input tensor
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
void riscv_nn_concate_s16_x(const int16_t * in_tensor,
                            const uint16_t in_tensor_x,
                            const uint16_t in_tensor_y,
                            const uint16_t in_tensor_z,
                            const uint16_t in_tensor_w,
                            int16_t * out_tensor,
                            const uint16_t out_tensor_x,
                            const uint32_t out_offset_x);

/**
 * @brief       This function concatenates a signed 16-bit integer input tensor
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
void riscv_nn_concate_s16_y(const int16_t * in_tensor,
                            const uint16_t in_tensor_x,
                            const uint16_t in_tensor_y,
                            const uint16_t in_tensor_z,
                            const uint16_t in_tensor_w,
                            int16_t * out_tensor,
                            const uint16_t out_tensor_y,
                            const uint32_t out_offset_y);

/**
 * @brief       This function concatenates a signed 16-bit integer input tensor
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
void riscv_nn_concate_s16_z(const int16_t * in_tensor,
                            const uint16_t in_tensor_x,
                            const uint16_t in_tensor_y,
                            const uint16_t in_tensor_z,
                            const uint16_t in_tensor_w,
                            int16_t * out_tensor,
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
void riscv_nn_concate_f16_w(const float16_t * in_tensor,
                            const uint16_t in_tensor_x,
                            const uint16_t in_tensor_y,
                            const uint16_t in_tensor_z,
                            const uint16_t in_tensor_w,
                            float16_t * out_tensor,
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
void riscv_nn_concate_f16_x(const float16_t * in_tensor,
                            const uint16_t in_tensor_x,
                            const uint16_t in_tensor_y,
                            const uint16_t in_tensor_z,
                            const uint16_t in_tensor_w,
                            float16_t * out_tensor,
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
void riscv_nn_concate_f16_y(const float16_t * in_tensor,
                            const uint16_t in_tensor_x,
                            const uint16_t in_tensor_y,
                            const uint16_t in_tensor_z,
                            const uint16_t in_tensor_w,
                            float16_t * out_tensor,
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
void riscv_nn_concate_f16_z(const float16_t * in_tensor,
                            const uint16_t in_tensor_x,
                            const uint16_t in_tensor_y,
                            const uint16_t in_tensor_z,
                            const uint16_t in_tensor_w,
                            float16_t * out_tensor,
                            const uint16_t out_tensor_z,
                            const uint32_t out_offset_z);

#endif  //__riscv_zfh

/**
 * @brief       This function padds the speicified value to a signed 8-bit
 *              integer input tensor along the w, z, y and x dimension.
 * @param[in]   in_tensor       Pointer to the input tensor
 * @param[in]   in_tensor_w     W dimension of the input tensor
 * @param[in]   in_tensor_z     Z dimension of the input tensor
 * @param[in]   in_tensor_y     Y dimension of the input tensor
 * @param[in]   in_tensor_x     X dimension of the input tensor
 * @param[in]   pre_pad_w       Padding size for the head of W dimension
 * @param[in]   pre_pad_z       Padding size for the head of Z dimension
 * @param[in]   pre_pad_y       Padding size for the head of Y dimension
 * @param[in]   pre_pad_x       Padding size for the head of X dimension
 * @param[in]   post_pad_w      Padding size for the tail of W dimension
 * @param[in]   post_pad_z      Padding size for the tail of Z dimension
 * @param[in]   post_pad_y      Padding size for the tail of Y dimension
 * @param[in]   post_pad_x      Padding size for the tail of X dimension
 * @param[in]   pad_value       Value for padding
 * @param[out]  out_tensor      Pointer to the output tensor
 * @return      None
 *
 * @note
 * - The assumed data layout for both input and output tensors is wzyx.
 */
void riscv_nn_pad_s8(const int8_t * in_tensor,
                     const uint32_t in_tensor_w,
                     const uint32_t in_tensor_z,
                     const uint32_t in_tensor_y,
                     const uint32_t in_tensor_x,
                     const uint32_t pre_pad_w,
                     const uint32_t pre_pad_z,
                     const uint32_t pre_pad_y,
                     const uint32_t pre_pad_x,
                     const uint32_t post_pad_w,
                     const uint32_t post_pad_z,
                     const uint32_t post_pad_y,
                     const uint32_t post_pad_x,
                     const int8_t pad_value,
                     int8_t * out_tensor);

/**
 * @brief       This function padds the speicified value to a signed 16-bit
 *              integer input tensor along the w, z, y and x dimension.
 * @param[in]   in_tensor       Pointer to the input tensor
 * @param[in]   in_tensor_w     W dimension of the input tensor
 * @param[in]   in_tensor_z     Z dimension of the input tensor
 * @param[in]   in_tensor_y     Y dimension of the input tensor
 * @param[in]   in_tensor_x     X dimension of the input tensor
 * @param[in]   pre_pad_w       Padding size for the head of W dimension
 * @param[in]   pre_pad_z       Padding size for the head of Z dimension
 * @param[in]   pre_pad_y       Padding size for the head of Y dimension
 * @param[in]   pre_pad_x       Padding size for the head of X dimension
 * @param[in]   post_pad_w      Padding size for the tail of W dimension
 * @param[in]   post_pad_z      Padding size for the tail of Z dimension
 * @param[in]   post_pad_y      Padding size for the tail of Y dimension
 * @param[in]   post_pad_x      Padding size for the tail of X dimension
 * @param[in]   pad_value       Value for padding
 * @param[out]  out_tensor      Pointer to the output tensor
 * @return      None
 *
 * @note
 * - The assumed data layout for both input and output tensors is wzyx.
 */
void riscv_nn_pad_s16(const int16_t * in_tensor,
                      const uint32_t in_tensor_w,
                      const uint32_t in_tensor_z,
                      const uint32_t in_tensor_y,
                      const uint32_t in_tensor_x,
                      const uint32_t pre_pad_w,
                      const uint32_t pre_pad_z,
                      const uint32_t pre_pad_y,
                      const uint32_t pre_pad_x,
                      const uint32_t post_pad_w,
                      const uint32_t post_pad_z,
                      const uint32_t post_pad_y,
                      const uint32_t post_pad_x,
                      const int16_t pad_value,
                      int16_t * out_tensor);

/**
 * @brief       This function makes a copy of a portion of a signed 16-bit
 *              integer input tensor along the w-axis.
 * @param[in]   in_tensor       Pointer to the input tensor
 * @param[in]   in_tensor_w     W dimension of the input tensor
 * @param[in]   in_tensor_z     Z dimension of the input tensor
 * @param[in]   in_tensor_y     Y dimension of the input tensor
 * @param[in]   in_tensor_x     X dimension of the input tensor
 * @param[in]   begin_w         Position to start the slicing on w-axis
 * @param[in]   end_w           Position to end the slicing on w-axis
 * @param[out]  out_tensor      Pointer to the output tensor
 * @return      None
 *
 * @note
 * - The x, y and z dimension of the output tensor will be the same as those of
 *   the input tensor.
 * - The assumed data layout for both input and output tensors is wzyx.
 */
void riscv_nn_slice_s16_w(const int16_t * in_tensor,
                          const uint32_t in_tensor_w,
                          const uint32_t in_tensor_z,
                          const uint32_t in_tensor_y,
                          const uint32_t in_tensor_x,
                          const uint32_t begin_w,
                          const uint32_t end_w,
                          int16_t * out_tensor);

/**
 * @brief       This function makes a copy of a portion of a signed 16-bit
 *              integer input tensor along the x-axis.
 * @param[in]   in_tensor       Pointer to the input tensor
 * @param[in]   in_tensor_w     W dimension of the input tensor
 * @param[in]   in_tensor_z     Z dimension of the input tensor
 * @param[in]   in_tensor_y     Y dimension of the input tensor
 * @param[in]   in_tensor_x     X dimension of the input tensor
 * @param[in]   begin_x         Position to start the slicing on x-axis
 * @param[in]   end_x           Position to end the slicing on x-axis
 * @param[out]  out_tensor      Pointer to the output tensor
 * @return      None
 *
 * @note
 * - The y, z and w dimensions of the output tensor will be the same as those of
 *   the input tensor.
 * - The assumed data layout for both input and output tensors is wzyx.
 */
void riscv_nn_slice_s16_x(const int16_t * in_tensor,
                          const uint32_t in_tensor_w,
                          const uint32_t in_tensor_z,
                          const uint32_t in_tensor_y,
                          const uint32_t in_tensor_x,
                          const uint32_t begin_x,
                          const uint32_t end_x,
                          int16_t * out_tensor);

/**
 * @brief       This function makes a copy of a portion of a signed 16-bit
 *              integer input tensor along the y-axis.
 * @param[in]   in_tensor       Pointer to the input tensor
 * @param[in]   in_tensor_w     W dimension of the input tensor
 * @param[in]   in_tensor_z     Z dimension of the input tensor
 * @param[in]   in_tensor_y     Y dimension of the input tensor
 * @param[in]   in_tensor_x     X dimension of the input tensor
 * @param[in]   begin_y         Position to start the slicing on y-axis
 * @param[in]   end_y           Position to end the slicing on y-axis
 * @param[out]  out_tensor      Pointer to the output tensor
 * @return      None
 *
 * @note
 * - The x, z and w dimensions of the output tensor will be the same as those of
 *   the input tensor.
 * - The assumed data layout for both input and output tensors is wzyx.
 */
void riscv_nn_slice_s16_y(const int16_t * in_tensor,
                          const uint32_t in_tensor_w,
                          const uint32_t in_tensor_z,
                          const uint32_t in_tensor_y,
                          const uint32_t in_tensor_x,
                          const uint32_t begin_y,
                          const uint32_t end_y,
                          int16_t * out_tensor);

/**
 * @brief       This function makes a copy of a portion of a signed 16-bit
 *              integer input tensor along the z-axis.
 * @param[in]   in_tensor       Pointer to the input tensor
 * @param[in]   in_tensor_w     W dimension of the input tensor
 * @param[in]   in_tensor_z     Z dimension of the input tensor
 * @param[in]   in_tensor_y     Y dimension of the input tensor
 * @param[in]   in_tensor_x     X dimension of the input tensor
 * @param[in]   begin_z         Position to start the slicing on z-axis
 * @param[in]   end_z           Position to end the slicing on z-axis
 * @param[out]  out_tensor      Pointer to the output tensor
 * @return      None
 *
 * @note
 * - The x, y and w dimensions of the output tensor will be the same as those of
 *   the input tensor.
 * - The assumed data layout for both input and output tensors is wzyx.
 */
void riscv_nn_slice_s16_z(const int16_t * in_tensor,
                          const uint32_t in_tensor_w,
                          const uint32_t in_tensor_z,
                          const uint32_t in_tensor_y,
                          const uint32_t in_tensor_x,
                          const uint32_t begin_z,
                          const uint32_t end_z,
                          int16_t * out_tensor);

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
void riscv_nn_split_s8_w(const int8_t * in_tensor,
                         const uint32_t in_tensor_x,
                         const uint32_t in_tensor_y,
                         const uint32_t in_tensor_z,
                         const uint32_t in_tensor_w,
                         int8_t * out_tensor,
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
void riscv_nn_split_s8_x(const int8_t * in_tensor,
                         const uint32_t in_tensor_x,
                         const uint32_t in_tensor_y,
                         const uint32_t in_tensor_z,
                         const uint32_t in_tensor_w,
                         int8_t * out_tensor,
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
void riscv_nn_split_s8_y(const int8_t * in_tensor,
                         const uint32_t in_tensor_x,
                         const uint32_t in_tensor_y,
                         const uint32_t in_tensor_z,
                         const uint32_t in_tensor_w,
                         int8_t * out_tensor,
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
void riscv_nn_split_s8_z(const int8_t * in_tensor,
                         const uint32_t in_tensor_x,
                         const uint32_t in_tensor_y,
                         const uint32_t in_tensor_z,
                         const uint32_t in_tensor_w,
                         int8_t * out_tensor,
                         const uint32_t split_len_z,
                         const uint32_t in_offset);

/**
 * @brief       This function splits a signed 16-bit integer input tensor along
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
void riscv_nn_split_s16_w(const int16_t * in_tensor,
                          const uint32_t in_tensor_x,
                          const uint32_t in_tensor_y,
                          const uint32_t in_tensor_z,
                          const uint32_t in_tensor_w,
                          int16_t * out_tensor,
                          const uint32_t split_len_w,
                          const uint32_t in_offset);

/**
 * @brief       This function splits a signed 16-bit integer input tensor along
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
void riscv_nn_split_s16_x(const int16_t * in_tensor,
                          const uint32_t in_tensor_x,
                          const uint32_t in_tensor_y,
                          const uint32_t in_tensor_z,
                          const uint32_t in_tensor_w,
                          int16_t * out_tensor,
                          const uint32_t split_len_x,
                          const uint32_t in_offset);

/**
 * @brief       This function splits a signed 16-bit integer input tensor along
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
void riscv_nn_split_s16_y(const int16_t * in_tensor,
                          const uint32_t in_tensor_x,
                          const uint32_t in_tensor_y,
                          const uint32_t in_tensor_z,
                          const uint32_t in_tensor_w,
                          int16_t * out_tensor,
                          const uint32_t split_len_y,
                          const uint32_t in_offset);

/**
 * @brief       This function splits a signed 16-bit integer input tensor along
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
void riscv_nn_split_s16_z(const int16_t * in_tensor,
                          const uint32_t in_tensor_x,
                          const uint32_t in_tensor_y,
                          const uint32_t in_tensor_z,
                          const uint32_t in_tensor_w,
                          int16_t * out_tensor,
                          const uint32_t split_len_z,
                          const uint32_t in_offset);

/**
 * @brief       This function makes a copy of a portion of a signed 8-bit
 *              integer input tensor along the w, z, y and x axis.
 * @param[in]   in_tensor       Pointer to the input tensor
 * @param[in]   in_tensor_w     W dimension of the input tensor
 * @param[in]   in_tensor_z     Z dimension of the input tensor
 * @param[in]   in_tensor_y     Y dimension of the input tensor
 * @param[in]   in_tensor_x     X dimension of the input tensor
 * @param[in]   begin_w         Position to start the slicing in the W dimension
 * @param[in]   begin_z         Position to start the slicing in the Z dimension
 * @param[in]   begin_y         Position to start the slicing in the Y dimension
 * @param[in]   begin_x         Position to start the slicing in the X dimension
 * @param[in]   end_w           Position to end the slicing in the W dimension
 * @param[in]   end_z           Position to end the slicing in the Z dimension
 * @param[in]   end_y           Position to end the slicing in the Y dimension
 * @param[in]   end_z           Position to end the slicing in the X dimension
 * @param[in]   stride_w        Stride in the W dimension
 * @param[in]   stride_z        Stride in the Z dimension
 * @param[in]   stride_y        Stride in the Y dimension
 * @param[in]   stride_x        Stride in the X dimension
 * @param[out]  out_tensor      Pointer to the output tensor
 * @return      None
 *
 * @note
 * - The assumed data layout for both input and output tensors is wzyx.
 */
void riscv_nn_strided_slice_s8(const int8_t * in_tensor,
                               const uint32_t in_tensor_w,
                               const uint32_t in_tensor_z,
                               const uint32_t in_tensor_y,
                               const uint32_t in_tensor_x,
                               const uint32_t begin_w,
                               const uint32_t begin_z,
                               const uint32_t begin_y,
                               const uint32_t begin_x,
                               const uint32_t end_w,
                               const uint32_t end_z,
                               const uint32_t end_y,
                               const uint32_t end_x,
                               const uint32_t stride_w,
                               const uint32_t stride_z,
                               const uint32_t stride_y,
                               const uint32_t stride_x,
                               int8_t * out_tensor);

/**
 * @brief       This function makes a copy of a portion of a signed 16-bit
 *              integer input tensor along the w, z, y and x axis.
 * @param[in]   in_tensor       Pointer to the input tensor
 * @param[in]   in_tensor_w     W dimension of the input tensor
 * @param[in]   in_tensor_z     Z dimension of the input tensor
 * @param[in]   in_tensor_y     Y dimension of the input tensor
 * @param[in]   in_tensor_x     X dimension of the input tensor
 * @param[in]   begin_w         Position to start the slicing in the W dimension
 * @param[in]   begin_z         Position to start the slicing in the Z dimension
 * @param[in]   begin_y         Position to start the slicing in the Y dimension
 * @param[in]   begin_x         Position to start the slicing in the X dimension
 * @param[in]   end_w           Position to end the slicing in the W dimension
 * @param[in]   end_z           Position to end the slicing in the Z dimension
 * @param[in]   end_y           Position to end the slicing in the Y dimension
 * @param[in]   end_z           Position to end the slicing in the X dimension
 * @param[in]   stride_w        Stride in the W dimension
 * @param[in]   stride_z        Stride in the Z dimension
 * @param[in]   stride_y        Stride in the Y dimension
 * @param[in]   stride_x        Stride in the X dimension
 * @param[out]  out_tensor      Pointer to the output tensor
 * @return      None
 *
 * @note
 * - The assumed data layout for both input and output tensors is wzyx.
 */
void riscv_nn_strided_slice_s16(const int16_t * in_tensor,
                                const uint32_t in_tensor_w,
                                const uint32_t in_tensor_z,
                                const uint32_t in_tensor_y,
                                const uint32_t in_tensor_x,
                                const uint32_t begin_w,
                                const uint32_t begin_z,
                                const uint32_t begin_y,
                                const uint32_t begin_x,
                                const uint32_t end_w,
                                const uint32_t end_z,
                                const uint32_t end_y,
                                const uint32_t end_x,
                                const uint32_t stride_w,
                                const uint32_t stride_z,
                                const uint32_t stride_y,
                                const uint32_t stride_x,
                                int16_t * out_tensor);

/**
 *   * @}
 */

#ifdef __cplusplus
}
#endif

#endif
