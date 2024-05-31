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

#ifndef __RISCV_NN_UTIL_H__
#define __RISCV_NN_UTIL_H__

#ifdef __cplusplus
extern    "C"
{
#endif

#include "riscv_math_types.h"
#include "riscv_nn_types.h"
#include "riscv_nn_activation.h"

/**
 * @defgroup Util Util Functions
 * @brief Utils functions are miscellaneous auxiliary tools.
 *
 * @{
 */

/**
 * @brief This is the enumeration type to select an unsampling algorithm.
 *
 */
typedef enum
{
    NN_UPSAMPLE_NEAREST = 0,
} riscv_nn_upsample_method;

#ifdef __riscv_zfh
/**
 * @brief           This function calculates the base-e exponential values for
 *                  half-precision floating-point inputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       size            Number of elements in the input vector
 * @param[out]      out_vec         Pointer to the output vector
 * @return          This function only returns 0.
 */
int32_t riscv_nn_exp_f16(const float16_t * in_vec,
                        const uint32_t size,
                        float16_t * out_vec);
#endif

/**
 * @brief           This function calculates the base-e exponential values for
 *                  floating-point inputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       size            Number of elements in the input vector
 * @param[out]      out_vec         Pointer to the output vector
 * @return          This function only returns 0.
 */
int32_t riscv_nn_exp_f32(const float32_t * in_vec,
                    uint32_t size,
                    float32_t * out_vec);

#ifdef __riscv_zfh
/**
 * @brief           This function performs layer normalization on
 *                  half-precision floating-point inputs.
 * @param[in]       in_tensor       Pointer to the input tensor
 * @param[in]       epsilon         Constant to be added to mini-batch variances
 * @param[in]       beta            Pointer to the offset vector for each
 *                                  feature
 * @param[in]       gamma           Pointer to the scaling vector for each
 *                                  feature
 * @param[in]       sentence_len    Length of input sentences
 * @param[in]       feature_len     Length of features
 * @param[out]      out_tensor      Pointer to the output tensor
 * @return          This function only returns 0.
 *
 * @note            The batch size is assumed to be 1.
 */
int32_t riscv_nn_layer_norm_f16(const float16_t *in_tensor,
                            const float16_t epsilon,
                            const float16_t *beta,
                            const float16_t *gamma,
                            const uint32_t sentence_len,
                            const uint32_t feature_len,
                            float16_t *out_tensor);
#endif

/**
 * @brief           This function performs a unidirectional long short-term
 *                  memory (LSTM) operation with signed 8-bit input and output,
 *                  and a signed 16-bit gate output.
 * @param[in]       scratch_buffers                 A structure containing the
 *                                                  scratch buffers. Each
 *                                                  scratch buffer is expected
 *                                                  to have a size of "lstm_dims->num_batches
 *                                                  * lstm_dims->num_outputs."
 * @param[in]       input_data                      Pointer to the input data
 * @param[in]       lstm_dims                       Dimension of the LSTM's inputs
 * @param[in]       in_to_in_weights                The input weights
 * @param[in]       in_to_forget_weights            The forget weights
 * @param[in]       in_to_cell_weights              The cell weights
 * @param[in]       in_to_out_weights               The output weights
 * @param[in]       recurrent_to_in_weights         Recurrent of the input weights
 * @param[in]       recurrent_to_forget_weights     Recurrent of the forget weights
 * @param[in]       recurrent_to_cell_weights       Recurrent of the cell weights
 * @param[in]       recurrent_to_out_weights        Recurrent of the output weights
 * @param[in]       cell_to_in_weights              Dummy
 * @param[in]       cell_to_forget_weights          Dummy
 * @param[in]       cell_to_out_weights             Dummy
 * @param[in]       projection_weights              Dummy
 * @param[in]       lstm                            LSTM parameters
 * @param[in]       output_state                    Pointer to the output state
 * @param[in]       cell_state                      Pointer to the cell state
 * @param[out]      output_data                     Pointer to the input data
 * @return          This function only returns 0.
 */
int32_t riscv_nn_lstm_unidirectional_s16_s8(riscv_nn_lstm_context *scratch_buffers,
                                        const int8_t *input_data,
                                        const riscv_nn_lstm_dims *lstm_dims,
                                        const int8_t *in_to_in_weights,
                                        const int8_t *in_to_forget_weights,
                                        const int8_t *in_to_cell_weights,
                                        const int8_t *in_to_out_weights,
                                        const int8_t *recurrent_to_in_weights,
                                        const int8_t *recurrent_to_forget_weights,
                                        const int8_t *recurrent_to_cell_weights,
                                        const int8_t *recurrent_to_out_weights,
                                        const int16_t *cell_to_in_weights,
                                        const int16_t *cell_to_forget_weights,
                                        const int16_t *cell_to_out_weights,
                                        const int8_t *projection_weights,
                                        const riscv_nn_lstm_params *lstm,
                                        int8_t *output_state,
                                        int16_t *cell_state,
                                        int8_t *output_data);

/**
 * @brief           This function turns an 8-bit input tensor into another
 *                  tensor with the same data but in a different shape.
 * @param[in]       in_tensor       Pointer to the input tensor
 * @param[out]      out_tensor      Pointer to the output tensor
 * @param[in]       size            Size, in bytes, of total input tensors
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
 * @brief           This function turns a half-precision floating-point input
 *                  tensor into another tensor with the same data but in a
 *                  different shape.
 * @param[in]       in_tensor       Pointer to the input tensor
 * @param[out]      out_tensor      Pointer to the output tensor
 * @param[in]       size            Size, in bytes, of total input tensors
 * @return          None
 */
void riscv_nn_reshape_f16(const float16_t *in_tensor,
                        float16_t *out_tensor,
                        const uint32_t size);
#endif

/**
 * @brief           This function performs singular value decomposition (SVD)
 *                  filtering for signed 8-bit integer inputs and a signed
 *                  16-bit integer state tensor.
 * @param[in]       tmp_buf             Temporary buffer for the input tensor
 * @param[in]       tmp_buf2            Temporary buffer for the output tensor
 * @param[in]       rank                Number of largest elements to be kept
 * @param[in]       in_offset           Offset value for the input tensor. It
 *                                      should be in the range of -127 to 128.
 * @param[in]       out_offset          Offset value for the output tensor. It
 *                                      should be in the range of -128 to 127.
 * @param[in]       in_act_min          Minimum value that the intput tensor is
 *                                      limited to. It should be in the range of
 *                                      -128 to 127.
 * @param[in]       in_act_max          Maximum value that the intput tensor is
 *                                      limited to. It should be in the range of
 *                                      -128 to 127.
 * @param[in]       out_act_min         Minimum value that the output tensor is
 *                                      limited to. It should be in the range of
 *                                      -128 to 127.
 * @param[in]       out_act_max         Maximum value that the output tensor is
 *                                      limited to. It should be in the range of
 *                                      -128 to 127.
 * @param[in]       in_scale            Scaling value for the quantization on
 *                                      the inputs
 * @param[in]       in_shift            Shift amount for the quantization on
 *                                      the inputs
 * @param[in]       out_scale           Scaling value for the quantization on
 *                                      the outputs
 * @param[in]       out_shift           Shift amount for the quantization on
 *                                      the outputs
 * @param[in]       in_batch            Size of input tensor batches
 * @param[in]       in_height           Height of the input tensor
 * @param[in]       in_tensor           Pointer to the input tensor
 * @param[in]       state_tensor        Pointer to the state tensor
 * @param[in]       wt_feature_batch    Size of the feature weight tensor
 *                                      batches
 * @param[in]       wt_feature_tensor   Pointer to the feature weight tensor
 * @param[in]       wt_time_height      Height of the time weight tensor
 * @param[in]       wt_time_tensor      Pointer to the time weight tensor
 * @param[in]       bias                Pointer to the bias vector
 * @param[out]      out_tensor          Pointer to the output tensor
 * @return          Returns 0 if successful; otherwise, returns -1 if the inputs
 *                  fail to meet the following constraints: in_height is
 *                  nonnegative and less than 0x7FFFFFF0, and wt_time_height is
 *                  also nonnegative.
 *
 * @note
 *  - bias could be a null pointer as the bias vector is optional for this
 *    function.
 *  - During the quantization process, positive in_shift and out_shift values
 *    are used to left shift calculation results whereas a negative ones are
 *    used to right shift.
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
 * @brief           This function performs singular value decomposition (SVD)
 *                  filtering for signed 8-bit integer inputs and a signed 8-bit
 *                  integer state tensor.
 * @param[in]       tmp_buf             Temporary buffer for the input tensor
 * @param[in]       tmp_buf2            Temporary buffer for the output tensor
 * @param[in]       rank                Number of largest elements to be kept
 * @param[in]       in_offset           Offset value for the input tensor It
 *                                      should be in the range of -127 to 128.
 * @param[in]       out_offset          Offset value for the output tensor. It
 *                                      should be in the range of -128 to 127.
 * @param[in]       in_act_min          Minimum value that the intput tensor is
 *                                      limited to. It should be in the range of
 *                                      -128 to 127.
 * @param[in]       in_act_max          Maximum value that the intput tensor is
 *                                      limited to. It should be in the range of
 *                                      -128 to 127.
 * @param[in]       out_act_min         Minimum value that the output tensor is
 *                                      limited to. It should be in the range of
 *                                      -128 to 127.
 * @param[in]       out_act_max         Maximum value that the output tensor is
 *                                      limited to. It should be in the range of
 *                                      -128 to 127.
 * @param[in]       in_scale            Scaling value for the quantization on
 *                                      the inputs
 * @param[in]       in_shift            Shift amount for the quantization on
 *                                      the inputs
 * @param[in]       out_scale           Scaling value for the quantization on
 *                                      the outputs
 * @param[in]       out_shift           Shift amount for the quantization on
 *                                      the outputs
 * @param[in]       in_batch            Size of input tensor batches
 * @param[in]       in_height           Height of the input tensor
 * @param[in]       in_tensor           Pointer to the input tensor
 * @param[in]       state_tensor        Pointer to the state tensor
 * @param[in]       wt_feature_batch    Size of the feature weight tensor
 *                                      batches
 * @param[in]       wt_feature_tensor   Pointer to the feature weight tensor
 * @param[in]       wt_time_height      Height of the time weight tensor
 * @param[in]       wt_time_tensor      Pointer to the time weight tensor
 * @param[in]       bias                Pointer to the bias vector
 * @param[out]      out_tensor          Pointer to the output tensor
 * @return          Returns 0 if successful; otherwise, returns -1 if the inputs
 *                  fail to meet the following constraints: in_height is
 *                  nonnegative and less than 0x7FFFFFF0, and wt_time_height is
 *                  also nonnegative.
 *
 * @note
 *  - bias could be a null pointer as the bias vector is optional for this
 *    function.
 *  - During the quantization process, positive in_shift and out_shift values
 *    are used to left shift calculation results whereas a negative ones are
 *    used to right shift.
 */
int riscv_nn_svdf_s8_state_s8(q31_t *tmp_buf,
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
                              q7_t *state_tensor,
                              const int32_t wt_feature_batch,
                              const q7_t *wt_feature_tensor,
                              const int32_t wt_time_height,
                              const q7_t *wt_time_tensor,
                              const q31_t *bias,
                              q7_t *out_tensor);
/**
 * @brief           This function identifies the k largest values and their
 *                  indices in a signed 8-bit integer input vector.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       k               Number of the largest values to be
 *                                  searched
 * @param[out]      val             The k largest values in the input vector
 * @param[out]      idx             Indices of the k largest values in the
 *                                  input vector
 * @return          This function only returns 0.
 *
 * @note
 * The k largest values are sorted from largest to smallest and stored in the
 * val output vector. If multiple elements share the same value, those with
 * smaller indices are given higher priority in the selection.
 */
int32_t riscv_nn_top_k_s8(q7_t *in_vec,
                        uint32_t size,
                        uint32_t k,
                        q7_t *val,
                        uint32_t *idx);

#ifdef __riscv_zfh
/**
 * @brief           This function identiﬁes the k largest values and their
 *                  indices in a half-precision floating-point input vector.
 * @param[in]       in_vec          Pointer to the input tensor
 * @param[in]       size            Number of elements in the input vector
 * @param[in]       k               Number of the largest values to be
 *                                  searched
 * @param[out]      val             The k largest values in the input vector
 * @param[out]      idx             Indices of the k largest values in the
 *                                  input vector
 * @return          This function only returns 0.
 *
 * @note
 * The k largest values are sorted from largest to smallest and stored in the
 * val output vector. If multiple elements share the same value, those with
 * smaller indices are given higher priority in the selection.
 */
int32_t riscv_nn_top_k_f16(float16_t *in_vec,
                        uint32_t size,
                        uint32_t k,
                        float16_t *val,
                        uint32_t *idx);
#endif

/**
 * @brief           This function performs upsampling on two-dimensional tensors
 *                  containing signed 8-bit integer data.
 * @param[in]       in_tensor           Pointer to the input tensor
 * @param[in]       in_tensor_dim_x     X dimension of the input tensor
 * @param[in]       in_tensor_dim_y     Y dimension of the input tensor
 * @param[in]       in_tensor_ch        Number of input tensor channels
 * @param[in]       scale_factor_x      Factor to be scaled up for X dimension
 * @param[in]       scale_factor_y      Factor to be scaled up for Y dimension
 * @param[in]       upsample_method     Algorithm used for upsampling
 * @param[out]      out_tensor          Pointer to the output tensor
 * @return          This function only returns 0.
 *
 * @note
 * Now only the algorithm NN_UPSAMPLE_NEAREST is supported for upsample_method.
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
 * @brief           This function performs upsampling on two-dimensional tensors
 *                  containing half-precision floating-point data.
 * @param[in]       in_tensor           Pointer to the input tensor
 * @param[in]       in_tensor_dim_x     X dimension of the input tensor
 * @param[in]       in_tensor_dim_y     Y dimension of the input tensor
 * @param[in]       in_tensor_ch        Number of input tensor channels
 * @param[in]       scale_factor_x      Factor to be scaled up for X dimension
 * @param[in]       scale_factor_y      Factor to be scaled up for Y dimension
 * @param[in]       upsample_method     Algorithm used for upsampling
 * @param[out]      out_tensor          Pointer to the output tensor
 * @return          This function only returns 0.
 *
 * @note
 * Now only the algorithm NN_UPSAMPLE_NEAREST is suuported for upsample_method.
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
