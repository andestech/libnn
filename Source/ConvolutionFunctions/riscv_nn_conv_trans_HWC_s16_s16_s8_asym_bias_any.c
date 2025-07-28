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

#include "internal_nn_math.h"
#include "riscv_nn_support.h"

//// Convolution Functions

int32_t riscv_nn_conv_trans_HWC_s16_s16_s8_asym_bias_any(const int16_t * in_tensor,
                                                         const int32_t in_tensor_dim_x,
                                                         const int32_t in_tensor_dim_y,
                                                         const int32_t in_tensor_ch,
                                                         const int32_t in_tensor_batch,
                                                         const int8_t * ker_weight,
                                                         const int32_t out_tensor_ch,
                                                         const int32_t ker_dim_x,
                                                         const int32_t ker_dim_y,
                                                         const int32_t pad_x,
                                                         const int32_t pad_y,
                                                         const int32_t pad_offset_x,
                                                         const int32_t pad_offset_y,
                                                         const int32_t stride_x,
                                                         const int32_t stride_y,
                                                         const int64_t * bias,
                                                         int16_t * out_tensor,
                                                         const int32_t * out_shift,
                                                         const int32_t * out_scale,
                                                         const int32_t out_offset,
                                                         const int32_t in_offset,
                                                         const int32_t act_min,
                                                         const int32_t act_max,
                                                         const int32_t out_tensor_dim_x,
                                                         const int32_t out_tensor_dim_y,
                                                         int8_t * in_tmp_buf)
{
    (void)pad_offset_x;
    (void)pad_offset_y;
    (void)in_offset;
    (void)out_offset;

    int64_t *out_tmp_buf = (int64_t*)in_tmp_buf;

    int i_batch;
    for (i_batch = 0; i_batch < in_tensor_batch; i_batch++)
    {
        riscv_nn_set_zero_s8((int8_t*)out_tmp_buf, sizeof(*out_tmp_buf) * out_tensor_dim_y * out_tensor_dim_x * out_tensor_ch);

        for (int i_in_y = 0; i_in_y < in_tensor_dim_y; ++i_in_y)
        {
            for (int i_in_x = 0; i_in_x < in_tensor_dim_x; ++i_in_x)
            {
                for (int i_in_channel = 0; i_in_channel < in_tensor_ch; ++i_in_channel)
                {
                    const int out_x_origin = (i_in_x * stride_x) - pad_x;
                    const int out_y_origin = (i_in_y * stride_y) - pad_y;

                    for (int i_ker_y = 0; i_ker_y < ker_dim_y; ++i_ker_y)
                    {
                        for (int i_ker_x = 0; i_ker_x < ker_dim_x; ++i_ker_x)
                        {
                            for (int i_out_channel = 0; i_out_channel < out_tensor_ch; ++i_out_channel)
                            {
                                const int i_out_x = out_x_origin + i_ker_x;
                                const int i_out_y = out_y_origin + i_ker_y;

                                // We cannot accumulate out of bounds.
                                // ker_weight: {I,H,W,O}
                                if ((i_out_x >= 0) && (i_out_x < out_tensor_dim_x) && (i_out_y >= 0) && (i_out_y < out_tensor_dim_y))
                                {
                                    int16_t input_value  = in_tensor[(i_in_y * in_tensor_dim_x + i_in_x) * in_tensor_ch + i_in_channel];
                                    int8_t filter_value = ker_weight[i_out_channel * in_tensor_ch * ker_dim_y * ker_dim_x + (i_ker_y * ker_dim_x + i_ker_x) * in_tensor_ch + i_in_channel];

                                    out_tmp_buf[(i_out_y * out_tensor_dim_x + i_out_x) * out_tensor_ch + i_out_channel] += (input_value) * filter_value;
                                }
                                else
                                {
                                    //skip point which is out of bound.
                                }
                            }
                        }
                    }
                }
            }
        }

        for (int i_out_y = 0; i_out_y < out_tensor_dim_y; ++i_out_y)
        {
            for (int i_out_x = 0; i_out_x < out_tensor_dim_x; ++i_out_x)
            {
                for (int i_out_channel = 0; i_out_channel < out_tensor_ch; ++i_out_channel)
                {
                    int64_t acc = out_tmp_buf[(i_out_y * out_tensor_dim_x + i_out_x) * out_tensor_ch + i_out_channel];
                    const int32_t reduced_scale = REDUCE_MULTIPLIER(out_scale[i_out_channel]);

                    if (bias)
                    {
                        acc += bias[i_out_channel];
                    }

                    int32_t conv_out = riscv_nn_requantize_s64(acc, reduced_scale, out_shift[i_out_channel]);
                    conv_out = MAX(conv_out, act_min);
                    conv_out = MIN(conv_out, act_max);
                    out_tensor[i_out_channel + (i_out_y * out_tensor_dim_x + i_out_x) * out_tensor_ch] = (int16_t)conv_out;
                }
            }
        }

        in_tensor += (in_tensor_dim_x * in_tensor_dim_y * in_tensor_ch);    //next batch.
        out_tensor += (out_tensor_dim_x * out_tensor_dim_y * out_tensor_ch);//next batch.
    }

    return 0;

}

uint32_t riscv_nn_conv_trans_HWC_s16_s16_s8_asym_bias_any_get_buffer_size(const int32_t in_tensor_dim_x,
                                                                          const int32_t in_tensor_dim_y,
                                                                          const int32_t in_tensor_ch,
                                                                          const int32_t in_tensor_batch,
                                                                          const int32_t out_tensor_ch,
                                                                          const int32_t ker_dim_x,
                                                                          const int32_t ker_dim_y,
                                                                          const int32_t pad_x,
                                                                          const int32_t pad_y,
                                                                          const int32_t stride_x,
                                                                          const int32_t stride_y,
                                                                          const int32_t out_tensor_dim_x,
                                                                          const int32_t out_tensor_dim_y)
{
    // return the required buffer size (in byte)
    (void)in_tensor_dim_x;
    (void)in_tensor_dim_y;
    (void)in_tensor_batch;
    (void)pad_x;
    (void)pad_y;
    (void)stride_x;
    (void)stride_y;

    uint32_t buf_size = 0;

    //Note. the element of partial results is 8 byte)
    //temporary buffer to keep the partial results
    buf_size = (out_tensor_ch * out_tensor_dim_y * out_tensor_dim_x) * sizeof(int64_t);

    return buf_size;
}
