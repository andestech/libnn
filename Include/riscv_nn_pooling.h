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

#ifndef __RISCV_NN_POOLING_H__
#define __RISCV_NN_POOLING_H__

#ifdef __cplusplus
extern    "C"
{
#endif

#include "riscv_math_types.h"

/**
 * @defgroup Pooling Pooling Functions
 * @brief The pooling functions are used to downsample input data. They include
 * max and average pooling functions.
 *
 * @{
 */

/**
 * @brief           This is an average pooling function for signed 8-bit integer
 *                  inputs.
 * @param[in]       in_tensor       pointer of the input tensor
 * @param[in]       in_tensor_dim   dimension of the input tensor
 * @param[in]       in_tensor_ch    number of input tensor channels
 * @param[in]       ker_dim         dimension of the pooling window
 * @param[in]       pad             padding size
 * @param[in]       stride          convolution stride
 * @param[in]       out_tensor_dim  dimension of the output tensor
 * @param[in]       in_tmp_buf      dummy
 * @param[out]      out_tensor      pointer of the output tensor.
 * @return          None
 *
 * @b Example:
 * @code
 * #define IN_DIM 32
 * #define IN_CH 32
 * #define KER_DIM 3
 * #define PAD 0
 * #define STRIDE 2
 * #define OUT_DIM 15
 *
 * q7_t in_data[IN_CH * IN_DIM * IN_DIM] = {...};
 * q7_t out_data[IN_CH * OUT_DIM * OUT_DIM] = {...};
 * q7_t *in_tmp_buf = NULL;
 *
 * riscv_nn_avepool_HWC_s8(in_data, IN_DIM, IN_CH, KER_DIM, PAD, STRIDE,
 *      OUT_DIM, in_tmp_buf, out_data);
 * @endcode
 */
void riscv_nn_avepool_HWC_s8(q7_t * in_tensor,
                           const uint16_t in_tensor_dim,
                           const uint16_t in_tensor_ch,
                           const uint16_t ker_dim,
                           const uint16_t pad,
                           const uint16_t stride,
                           const uint16_t out_tensor_dim,
                           q7_t * in_tmp_buf,
                           q7_t * out_tensor);

/**
 * @brief           This is an average pooling function for signed 8-bit integer
 *                  inputs in any x and y dimensions.
 * @param[in]       in_tensor           pointer of the input tensor
 * @param[in]       in_tensor_dim_x     x dimension of the input tensor
 * @param[in]       in_tensor_dim_y     y dimension of the input tensor
 * @param[in]       in_tensor_ch        number of input tensor channels
 * @param[in]       ker_dim_x           x dimension of the pooling window
 * @param[in]       ker_dim_y           y dimension of the pooling window
 * @param[in]       pad_x               padding size in the x dimension
 * @param[in]       pad_y               padding size in the y dimension
 * @param[in]       stride_x            stride of the pooling window in the x
 *                                      dimension
 * @param[in]       stride_y            stride of the pooling window in the y
 *                                      dimension
 * @param[in]       out_tensor_dim_x    x dimension of the output tensor
 * @param[in]       out_tensor_dim_y    y dimension of the output tensor
 * @param[in]       in_tmp_buf          dummy
 * @param[out]      out_tensor          pointer of the output tensor
 * @param[in]       out_lshift          left shift amount for the output
 * @return          None
 *
 * @b Example:
 * @code
 *  #define IN_X 15
 *  #define IN_Y 15
 *  #define IN_CH 10
 *  #define KER_DIM_X 2
 *  #define KER_DIM_Y 2
 *  #define PAD_X 0
 *  #define PAD_Y 0
 *  #define STRIDE_X 2
 *  #define STRIDE_Y 2
 *  #define OUT_LSHIFT 3
 *  #define OUT_X 7
 *  #define OUT_Y 7
 *
 * q7_t in_data[IN_CH * IN_X * IN_Y] = {...};
 * q7_t out_data[IN_CH * OUT_X * OUT_Y] = {...};
 * q7_t *in_tmp_buf = NULL;
 *
 * riscv_nn_avepool_HWC_s8_any(in_data, IN_X, IN_Y, IN_CH, KER_DIM_X, KER_DIM_Y,
 *      PAD_X, PAD_Y, STRIDE_X, STRIDE_Y, OUT_X, OUT_Y, in_tmp_buf, out_data,
 *      OUT_LSHIFT);
 * @endcode
 */
void riscv_nn_avepool_HWC_s8_any(q7_t * in_tensor,
                               const uint16_t in_tensor_dim_x,
                               const uint16_t in_tensor_dim_y,
                               const uint16_t in_tensor_ch,
                               const uint16_t ker_dim_x,
                               const uint16_t ker_dim_y,
                               const uint16_t pad_x,
                               const uint16_t pad_y,
                               const uint16_t stride_x,
                               const uint16_t stride_y,
                               const uint16_t out_tensor_dim_x,
                               const uint16_t out_tensor_dim_y,
                               q7_t * in_tmp_buf,
                               q7_t * out_tensor,
                               const uint16_t out_lshift);

/**
 * @brief           This is an average pooling function for signed 8-bit integer
 *                  inputs in any x and y dimensions with activation parameters
 *                  to limit the outputs.
 * @param[in]       in_tensor_dim_y     y dimension of the input tensor
 * @param[in]       in_tensor_dim_x     x dimension of the input tensor
 * @param[in]       out_tensor_dim_y    y dimension of the output tensor
 * @param[in]       out_tensor_dim_x    x dimension of the output tensor
 * @param[in]       stride_y            stride of the pooling window in the y
 *                                      dimension
 * @param[in]       stride_x            stride of the pooling window in the x
 *                                      dimension
 * @param[in]       ker_dim_y           y dimension of the pooling window
 * @param[in]       ker_dim_x           x dimension of the pooling window
 * @param[in]       pad_y               padding size in the y dimension
 * @param[in]       pad_x               padding size in the x dimension
 * @param[in]       act_min             minimum value that the output tensor is
 *                                      limited to. It should be in the range of
 *                                      -128 to 127.
 * @param[in]       act_max             maximum value that the output tensor is
 *                                      limited to. It should be in the range of
 *                                      -128 to 127.
 * @param[in]       in_tensor_ch        number of input tensor channels
 * @param[in]       in_tensor           pointer of the input tensor
 * @param[in]       in_tmp_buf          temporary buffer for the input tensor.
 *                                      Its needed size could be obtained by
 *                                      calling riscv_nn_avepool_HWC_s8_any_act_get_buffer_size.
 * @param[out]      out_tensor          pointer of the output tensor
 * @return          This function only returns 0.
 */
int32_t riscv_nn_avepool_HWC_s8_any_act(const int in_tensor_dim_y,
                                const int in_tensor_dim_x,
                                const int out_tensor_dim_y,
                                const int out_tensor_dim_x,
                                const int stride_y,
                                const int stride_x,
                                const int ker_dim_y,
                                const int ker_dim_x,
                                const int pad_y,
                                const int pad_x,
                                const int act_min,
                                const int act_max,
                                const int in_tensor_ch,
                                int8_t *in_tensor,
                                int16_t *in_tmp_buf,
                                int8_t *out_tensor);

/**
 * @brief           This function is used to obtain the required size, in bytes,
 *                  for the input temporary buffer of riscv_nn_avepool_HWC_s8_any_act.
 * @param[in]       out_tensor_dim_x    x dimension of the output tensor
 * @param[in]       in_tensor_ch        number of input tensor channels
 * @return          This function returns the size required by the temporary
 *                  buffer.
 */
int32_t riscv_nn_avepool_HWC_s8_any_act_get_buffer_size(const int out_tensor_dim_x,
                                                        const int in_tensor_ch);
/**
 * @brief           This is an average pooling function for signed 16-bit integer
 *                  inputs in any x and y dimensions with activation parameters
 *                  to limit the outputs.
 * @param[in]       in_tensor_dim_y     y dimension of the input tensor
 * @param[in]       in_tensor_dim_x     x dimension of the input tensor
 * @param[in]       out_tensor_dim_y    y dimension of the output tensor
 * @param[in]       out_tensor_dim_x    x dimension of the output tensor
 * @param[in]       stride_y            stride of the pooling window in the y
 *                                      dimension
 * @param[in]       stride_x            stride of the pooling window in the x
 *                                      dimension
 * @param[in]       ker_dim_y           y dimension of the pooling window
 * @param[in]       ker_dim_x           x dimension of the pooling window
 * @param[in]       pad_y               padding size in the y dimension
 * @param[in]       pad_x               padding size in the x dimension
 * @param[in]       act_min             minimum value that the output tensor is
 *                                      limited to. It should be in the range of
 *                                      -32768 to 32767.
 * @param[in]       act_max             maximum value that the output tensor is
 *                                      limited to. It should be in the range of
 *                                      -32768 to 32767.
 * @param[in]       in_tensor_ch        number of input tensor channels
 * @param[in]       in_tensor           pointer of the input tensor
 * @param[in]       in_tmp_buf          temporary buffer for the input tensor.
 *                                      Its needed size could be obtained by
 *                                      calling riscv_nn_avepool_HWC_s16_any_act_get_buffer_size.
 * @param[out]      out_tensor          pointer of the output tensor
 * @return          This function only returns 0.
 */
int32_t riscv_nn_avepool_HWC_s16_any_act(const int in_tensor_dim_y,
    const int in_tensor_dim_x,
    const int out_tensor_dim_y,
    const int out_tensor_dim_x,
    const int stride_y,
    const int stride_x,
    const int ker_dim_y,
    const int ker_dim_x,
    const int pad_y,
    const int pad_x,
    const int act_min,
    const int act_max,
    const int in_tensor_ch,
    int16_t *in_tensor,
    int16_t *in_tmp_buf,
    int16_t *out_tensor);

/**
 * @brief           This function is used to obtain the required size, in bytes,
 *                  for the input temporary buffer of riscv_nn_avepool_HWC_s16_any_act.
 * @param[in]       out_tensor_dim_x    x dimension of the output tensor
 * @param[in]       in_tensor_ch        number of input tensor channels
 * @return          This function returns the size required by the temporary
 *                  buffer.
 */
int32_t riscv_nn_avepool_HWC_s16_any_act_get_buffer_size(const int out_tensor_dim_x,
    const int in_tensor_ch);

#ifdef __riscv_zfh
/**
 * @brief           This is an average pooling function for half-precision
 *                  floating-point inputs in any x and y dimensions.
 * @param[in]       in_tensor           pointer of the input tensor
 * @param[in]       in_tensor_dim_x     x dimension of the input tensor
 * @param[in]       in_tensor_dim_y     y dimension of the input tensor
 * @param[in]       in_tensor_ch        number of input tensor channels
 * @param[in]       ker_dim_x           x dimension of the pooling window
 * @param[in]       ker_dim_y           y dimension of the pooling window
 * @param[in]       pad_x               padding size in the x dimension
 * @param[in]       pad_y               padding size in the y dimension
 * @param[in]       stride_x            stride of the pooling window in the x
 *                                      dimension
 * @param[in]       stride_y            stride of the pooling window in the y
 *                                      dimension
 * @param[in]       out_tensor_dim_x    x dimension of the output tensor
 * @param[in]       out_tensor_dim_y    y dimension of the output tensor
 * @param[in]       in_tmp_buf          dummy
 * @param[out]      out_tensor          pointer of the output tensor
 * @return          None
 */
void riscv_nn_avepool_HWC_f16_any(float16_t * in_tensor,
                                const uint16_t in_tensor_dim_x,
                                const uint16_t in_tensor_dim_y,
                                const uint16_t in_tensor_ch,
                                const uint16_t ker_dim_x,
                                const uint16_t ker_dim_y,
                                const uint16_t pad_x,
                                const uint16_t pad_y,
                                const uint16_t stride_x,
                                const uint16_t stride_y,
                                const uint16_t out_tensor_dim_x,
                                const uint16_t out_tensor_dim_y,
                                float16_t * in_tmp_buf,
                                float16_t * out_tensor);
#endif
/**
 * @brief           This is a max pooling function for signed 8-bit integer
 *                  inputs.
 * @param[in]       in_tensor       pointer of the input tensor
 * @param[in]       in_tensor_dim   dimension of the input tensor
 * @param[in]       in_tensor_ch    number of input tensor channels
 * @param[in]       ker_dim         dimension of the pooling window
 * @param[in]       pad             padding size
 * @param[in]       stride          stride of the pooling window
 * @param[in]       out_tensor_dim  dimension of the output tensor
 * @param[in]       in_tmp_buf      dummy
 * @param[out]      out_tensor      pointer of the output tensor
 * @return          None
 *
 * @b Example:
 * @code
 * #define IN_DIM 32
 * #define IN_CH 32
 * #define KER_DIM 3
 * #define PAD 0
 * #define STRIDE 2
 * #define OUT_DIM 15
 *
 * q7_t in_data[IN_CH * IN_DIM * IN_DIM] = {...};
 * q7_t out_data[IN_CH * OUT_DIM * OUT_DIM] = {...};
 *
 * riscv_nn_maxpool_HWC_s8(in_data, IN_DIM, IN_CH, KER_DIM, PAD, STRIDE,
 *      OUT_DIM, NULL, out_data);
 * @endcode
 */
 void riscv_nn_maxpool_HWC_s8(q7_t * in_tensor,
                            const uint16_t in_tensor_dim,
                            const uint16_t in_tensor_ch,
                            const uint16_t ker_dim,
                            const uint16_t pad,
                            const uint16_t stride,
                            const uint16_t out_tensor_dim,
                            q7_t * in_tmp_buf,
                            q7_t * out_tensor);

/**
 * @brief           This is a max pooling function for signed 8-bit integer
 *                  inputs in any x and y dimensions with the actvating
 *                  parameters to limit the outputs.
 * @param[in]       in_tensor_dim_y     y dimension of the input tensor
 * @param[in]       in_tensor_dim_x     x dimension of the input tensor
 * @param[in]       out_tensor_dim_y    y dimension of the output tensor
 * @param[in]       out_tensor_dim_x    x dimension of the output tensor
 * @param[in]       stride_y            stride of the pooling window in the y
 *                                      dimension
 * @param[in]       stride_x            stride of the pooling window in the x
 *                                      dimension
 * @param[in]       ker_dim_y           y dimension of the pooling window
 * @param[in]       ker_dim_x           x dimension of the pooling window
 * @param[in]       pad_y               padding size in the y dimension
 * @param[in]       pad_x               padding size in the x dimension
 * @param[in]       act_min             minimum value that the output tensor is
 *                                      limited to. It should be in the range of
 *                                      -128 to 127.
 * @param[in]       act_max             maximum value that the output tensor is
 *                                      limited to. It should be in the range of
 *                                      -128 to 127.
 * @param[in]       in_tensor_ch        number of input tensor channels
 * @param[in]       in_tensor           pointer of the input tensor
 * @param[in]       tmp_buffer          dummy
 * @param[out]      out_tensor          pointer of the output tensor
 * @return          This function only returns 0.
 */
int32_t riscv_nn_maxpool_HWC_s8_any_act(const uint16_t in_tensor_dim_y,
                                        const uint16_t in_tensor_dim_x,
                                        const uint16_t out_tensor_dim_y,
                                        const uint16_t out_tensor_dim_x,
                                        const uint16_t stride_y,
                                        const uint16_t stride_x,
                                        const uint16_t ker_dim_y,
                                        const uint16_t ker_dim_x,
                                        const uint16_t pad_y,
                                        const uint16_t pad_x,
                                        const int8_t act_min,
                                        const int8_t act_max,
                                        const uint16_t in_tensor_ch,
                                        int8_t *in_tensor,
                                        int16_t *tmp_buffer,
                                        int8_t *out_tensor);

/**
 * @brief           This is a max pooling function for signed 16-bit integer
 *                  inputs in any x and y dimensions with the actvating
 *                  parameters to limit the outputs.
 * @param[in]       in_tensor_dim_y     y dimension of the input tensor
 * @param[in]       in_tensor_dim_x     x dimension of the input tensor
 * @param[in]       out_tensor_dim_y    y dimension of the output tensor
 * @param[in]       out_tensor_dim_x    x dimension of the output tensor
 * @param[in]       stride_y            stride of the pooling window in the y
 *                                      dimension
 * @param[in]       stride_x            stride of the pooling window in the x
 *                                      dimension
 * @param[in]       ker_dim_y           y dimension of the pooling window
 * @param[in]       ker_dim_x           x dimension of the pooling window
 * @param[in]       pad_y               padding size in the y dimension
 * @param[in]       pad_x               padding size in the x dimension
 * @param[in]       act_min             minimum value that the output tensor is
 *                                      limited to. It should be in the range of
 *                                      -32768 to 32767.
 * @param[in]       act_max             maximum value that the output tensor is
 *                                      limited to. It should be in the range of
 *                                      -32768 to 32767.
 * @param[in]       in_tensor_ch        number of input tensor channels
 * @param[in]       in_tensor           pointer of the input tensor
 * @param[in]       tmp_buffer          dummy
 * @param[out]      out_tensor          pointer of the output tensor
 * @return          This function only returns 0.
 */
int32_t riscv_nn_maxpool_HWC_s16_any_act(const int32_t in_tensor_dim_y,
    const int32_t in_tensor_dim_x,
    const int32_t out_tensor_dim_y,
    const int32_t out_tensor_dim_x,
    const int32_t stride_y,
    const int32_t stride_x,
    const int32_t ker_dim_y,
    const int32_t ker_dim_x,
    const int32_t pad_y,
    const int32_t pad_x,
    const int32_t act_min,
    const int32_t act_max,
    const int32_t in_tensor_ch,
    int16_t *in_tensor,
    int16_t *tmp_buffer,
    int16_t *out_tensor);

#ifdef __riscv_zfh
/**
 * @brief           This is a max pooling function for half-precision
 *                  floating-point inputs in any x and y dimensions.
 * @param[in]       in_tensor_dim_y     y dimension of the input tensor
 * @param[in]       in_tensor_dim_x     x dimension of the input tensor
 * @param[in]       out_tensor_dim_y    y dimension of the output tensor
 * @param[in]       out_tensor_dim_x    x dimension of the output tensor
 * @param[in]       stride_y            stride of the pooling window in the y
 *                                      dimension
 * @param[in]       stride_x            stride of the pooling window in the x
 *                                      dimension
 * @param[in]       ker_dim_y           y dimension of the pooling window
 * @param[in]       ker_dim_x           x dimension of the pooling window
 * @param[in]       pad_y               padding size in the y dimension
 * @param[in]       pad_x               padding size in the x dimension
 * @param[in]       in_tensor_ch        number of input tensor channels
 * @param[in]       in_tensor           pointer of the input tensor
 * @param[in]       tmp_buffer          dummy
 * @param[out]      out_tensor          pointer of the output tensor
 * @return          This function only returns 0.
 */
int32_t riscv_nn_maxpool_HWC_f16_any(const uint16_t in_tensor_dim_y,
                                    const uint16_t in_tensor_dim_x,
                                    const uint16_t out_tensor_dim_y,
                                    const uint16_t out_tensor_dim_x,
                                    const uint16_t stride_y,
                                    const uint16_t stride_x,
                                    const uint16_t ker_dim_y,
                                    const uint16_t ker_dim_x,
                                    const uint16_t pad_y,
                                    const uint16_t pad_x,
                                    const uint16_t in_tensor_ch,
                                    float16_t *in_tensor,
                                    float16_t *tmp_buffer,
                                    float16_t *out_tensor);

#endif  //__riscv_zfh

/**
 *   * @}
 */
#ifdef __cplusplus
}
#endif

#endif
