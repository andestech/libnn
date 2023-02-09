/******************************************************************************
 * Copyright (C) 2010-2018 Arm Limited or its affiliates. All rights reserved.*
 * Copyright (C) 2018-2022 Andes Technology Corporation. All rights reserved. *
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

#ifndef __RISCV_NN_UTIL_H__
#define __RISCV_NN_UTIL_H__

#ifdef __cplusplus
extern    "C"
{
#endif

#include "riscv_math_types.h"

/**
 * @defgroup Utils Utils Functions
 * @brief Utils functions are miscellaneous auxiliary tools.
 *
 * @{
 */

typedef enum
{
    NN_UPSAMPLE_NEAREST = 0,
} riscv_nn_upsample_method;

#ifdef __riscv_zfh
/**
 * @brief           This function calculates the base-e exponential values of
 *                  half-precision floating-point inputs.
 * @param[in]       in_vec          pointer of the input vector
 * @param[in]       size            number of elements in the input vector
 * @param[out]      out_vec         pointer of the output vector
 * @return          This function only returns 0.
 */
int32_t riscv_nn_exp_f16(const float16_t * in_vec,
                        const uint32_t size,
                        float16_t * out_vec);
#endif

/**
 * @brief           This function calculates the base-e exponential values of
 *                  floating-point inputs.
 * @param[in]       in_vec          pointer of the input vector
 * @param[in]       size            number of elements in the input vector
 * @param[out]      out_vec         pointer of the output vector
 * @return          This function only returns 0.
 */
int32_t riscv_nn_exp_f32(const float32_t * in_vec,
                    uint32_t size,
                    float32_t * out_vec);

#ifdef __riscv_zfh
/**
 * @brief           This function performs layer normalization for
 *                  half-precision floating-point inputs.
 * @param[in]       in_tensor       pointer of the input tensor
 * @param[in]       epsilon         constant to be added to mini-batch variances
 * @param[in]       beta            pointer of the offset vector for each
 *                                  feature
 * @param[in]       gamma           pointer of the scaling vector for each
 *                                  feature
 * @param[in]       sentence_len    length of input sentences
 * @param[in]       features        length of features
 * @param[out]      out_tensor      pointer of the output tensor
 * @return          This function only returns 0.
 *
 * @note            The batch size is assumed to be 1.
 */
int32_t riscv_nn_layer_norm_f16(const float16_t *in_tensor,
                            const float16_t epsilon,
                            const float16_t *beta,
                            const float16_t *gamma,
                            const uint32_t sentence_len,
                            const uint32_t features,
                            float16_t *out_tensor);
#endif

/**
 * @brief           This function turns the 8-bit input tensor into another
 *                  tensor with the same data but in a different shape.
 * @param[in]       in_tensor       pointer of the input tensor
 * @param[out]      out_tensor      pointer of the output tensor
 * @param[in]       size            size, in bytes, of total input tensors
 * @return          None
 *
 * @b Example:
 * @code
 * #define SIZE 1024
 * int8_t in_tensor[SIZE] = {...};
 * int8_t out_tensor[SIZE];
 *
 * riscv_nn_reshape_s8(in_tensor, out_tensor, SIZE);
 * @endcode
 */
void riscv_nn_reshape_s8(const int8_t *in_tensor,
                        int8_t *out_tensor,
                        const uint32_t size);

#ifdef __riscv_zfh
/**
 * @brief           This function turns the half-precision floating-point input
 *                  tensor into another tensor with the same data but in a
 *                  different shape.
 * @param[in]       in_tensor       pointer of the input tensor
 * @param[out]      out_tensor      pointer of the output tensor
 * @param[in]       size            size, in bytes, of total input tensors
 * @return          None
 */
void riscv_nn_reshape_f16(const float16_t *in_tensor,
                        float16_t *out_tensor,
                        const uint32_t size);
#endif

/**
 * @brief           This function performs single value decomposition filter for
 *                  signed 8-bit integer inputs.
 * @param[in]       tmp_buf             temporary buffer for the input tensor
 * @param[in]       tmp_buf2            temporary buffer for the output tensor
 * @param[in]       rank                number of largest elements to be kept
 * @param[in]       in_offset           offset value for the input tensor It
 *                                      should be in the range of -127 to 128.
 * @param[in]       out_offset          offset value for the output tensor. It
 *                                      should be in the range of -128 to 127.
 * @param[in]       in_act_min          minimum value that the intput tensor is
 *                                      limited to. It should be in the range of
 *                                      -128 to 127.
 * @param[in]       in_act_max          maximum value that the intput tensor is
 *                                      limited to. It should be in the range of
 *                                      -128 to 127.
 * @param[in]       out_act_min         minimum value that the output tensor is
 *                                      limited to. It should be in the range of
 *                                      -128 to 127.
 * @param[in]       out_act_max         maximum value that the output tensor is
 *                                      limited to. It should be in the range of
 *                                      -128 to 127.
 * @param[in]       in_scale            pointer of the scaling vector for the
 *                                      intput tensor
 * @param[in]       in_shift            pointer of the shift vector for the
 *                                      input tensor
 * @param[in]       out_scale           pointer of the scaling vector for the
 *                                      output tensor
 * @param[in]       out_shift           pointer of the shift vector for the
 *                                      output tensor
 * @param[in]       in_batch            number of input tensors
 * @param[in]       in_height           height of the input tensor
 * @param[in]       in_tensor           pointer of the input tensor
 * @param[in]       state_tensor        pointer of the state tensor
 * @param[in]       wt_feature_batch    number of weight tensors for feature
 *                                      dimension
 * @param[in]       wt_feature_tensor   pointer of the weight tensor for feature
 *                                      dimension
 * @param[in]       wt_time_height      number of weight tensors for time
 *                                      dimension
 * @param[in]       wt_time_tensor      pointer of the weight tensor for time
 *                                      dimension
 * @param[in]       bias                pointer of the bias vector
 * @param[out]      out_tensor          pointer of the output tensor
 * @return          This function returns 0 on success; otherwise, it returns -1
 *                  if its inputs do not meet the constraints that in_height is
 *                  nonnegative and less than 0x7FFFFFF0 and wt_time_height is
 *                  also nonnegative.
 */
int32_t riscv_nn_svdf_s8(q31_t *tmp_buf,
                    q31_t *tmp_buf2,
                    const int32_t rank,
                    const int32_t in_offset,
                    const int32_t out_offset,
                    const int32_t in_act_min,
                    const int32_t in_act_max,
                    const int32_t out_act_min,
                    const int32_t out_act_max,
                    const int32_t in_scale,
                    const int32_t in_shift,
                    const int32_t out_scale,
                    const int32_t out_shift,
                    const int32_t in_batch,
                    const int32_t in_height,
                    const q7_t *in_tensor,
                    q15_t *state_tensor,
                    const int32_t wt_feature_batch,
                    const q7_t *wt_feature_tensor,
                    const int32_t wt_time_height,
                    const q15_t *wt_time_tensor,
                    const q31_t *bias,
                    q7_t *out_tensor);

/**
 * @brief           This function finds the k largest values and their indices
 *                  from the signed 8-bit integer input vector.
 * @param[in]       in_vec          pointer of the input vector
 * @param[in]       size            number of elements in the input vector
 * @param[in]       k               number of the largest values to be
 *                                  searched
 * @param[out]      val             the k largest values in the input vector
 * @param[out]      idx             indices of the k largest values in the
 *                                  input vector
 * @return          This function only returns 0.
 *
 * @note
 * The k largest values will be sorted from the largest to the smallest and
 * stored in the "val" output vector. If multiple elements share the same value,
 * those with smaller indices will have higher priority for the selection.
 */
int32_t riscv_nn_top_k_s8(q7_t *in_vec,
                        uint32_t size,
                        uint32_t k,
                        q7_t *val,
                        uint32_t *idx);

#ifdef __riscv_zfh
/**
 * @brief           This function finds the k largest values and their indices
 *                  from the half-precision floating point input vector.
 * @param[in]       in_vec          pointer of the input tensor
 * @param[in]       size            number of elements in the input vector
 * @param[in]       k               number of the largest values to be
 *                                  searched
 * @param[out]      val             the k largest values in the input vector
 * @param[out]      idx             indices of the k largest values in the
 *                                  input vector
 * @return          This function only returns 0.
 *
 * @note
 * The k largest values will be sorted from the largest to the smallest and
 * stored in the "val" output vector. If multiple elements share the same value,
 * those with smaller indices will have higher priority for the selection.
 */
int32_t riscv_nn_top_k_f16(float16_t *in_vec,
                        uint32_t size,
                        uint32_t k,
                        float16_t *val,
                        uint32_t *idx);
#endif

/**
 * @brief           This function performs upsampling for 2 dimension tensor
 *                  with signed 8-bit integer data.
 * @param[in]       in_tensor           pointer of the input tensor
 * @param[in]       in_tensor_dim_x     x dimension of the input tensor
 * @param[in]       in_tensor_dim_y     y dimension of the input tensor
 * @param[in]       in_tensor_ch        number of input tensor channels
 * @param[in]       scale_factor_x      factor to be scaled up for x dimension
 * @param[in]       scale_factor_y      factor to be scaled up for y dimension
 * @param[in]       upsample_method     algorithm to be applied for the
 *                                      upsampling
 * @param[out]      out_tensor          pointer of the output tensor
 * @return          This function only returns 0.
 *
 * @note
 * Now only NN_UPSAMPLE_NEAREST algorithm is supported for upsample_method.
 */
int32_t riscv_nn_upsampling2d_HWC_s8(const int8_t* in_tensor,
                                const uint32_t in_tensor_dim_x,
                                const uint32_t in_tensor_dim_y,
                                const uint32_t in_tensor_ch,
                                const uint32_t scale_factor_x,
                                const uint32_t scale_factor_y,
                                const riscv_nn_upsample_method upsample_method,
                                int8_t* out_tensor);

#ifdef __riscv_zfh
/**
 * @brief           This function performs upsampling for 2 dimension tensor
 *                  with half-precision floating-point data.
 * @param[in]       in_tensor           pointer of the input tensor
 * @param[in]       in_tensor_dim_x     x dimension of the input tensor
 * @param[in]       in_tensor_dim_y     y dimension of the input tensor
 * @param[in]       in_tensor_ch        number of input tensor channels
 * @param[in]       scale_factor_x      factor to be scaled up for x dimension
 * @param[in]       scale_factor_y      factor to be scaled up for y dimension
 * @param[in]       upsample_method     algorithm to be applied for the
 *                                      upsampling
 * @param[out]      out_tensor          pointer of the output tensor
 * @return          This function only returns 0.
 *
 * @note
 * Now only NN_UPSAMPLE_NEAREST algorithm is suuported for upsample_method.
 */
int32_t riscv_nn_upsampling2d_HWC_f16(const float16_t* in_tensor,
                                const uint32_t in_tensor_dim_x,
                                const uint32_t in_tensor_dim_y,
                                const uint32_t in_tensor_ch,
                                const uint32_t scale_factor_x,
                                const uint32_t scale_factor_y,
                                const riscv_nn_upsample_method upsample_method,
                                float16_t* out_tensor);
#endif

/**
 *   * @}
 */

#ifdef __cplusplus
}
#endif

#endif
