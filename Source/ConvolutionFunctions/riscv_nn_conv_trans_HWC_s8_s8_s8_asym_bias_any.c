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

#include "internal_nn_math.h"
#include "riscv_nn_support.h"

//// Convolution Functions

int32_t riscv_nn_conv_trans_HWC_s8_s8_s8_asym_bias_any(const q7_t *in_tensor,
                                                       const uint16_t in_tensor_dim_x,
                                                       const uint16_t in_tensor_dim_y,
                                                       const uint16_t in_tensor_ch,
                                                       const uint16_t in_tensor_batch,
                                                       const q7_t *ker_weight,
                                                       const uint16_t out_tensor_ch,
                                                       const uint16_t ker_dim_x,
                                                       const uint16_t ker_dim_y,
                                                       const uint16_t pad_x,
                                                       const uint16_t pad_y,
                                                       const uint16_t pad_offset_x,
                                                       const uint16_t pad_offset_y,
                                                       const uint16_t stride_x,
                                                       const uint16_t stride_y,
                                                       const int32_t *bias,
                                                       q7_t *out_tensor,
                                                       const int32_t *out_shift,
                                                       const int32_t *out_scale,
                                                       const int32_t out_offset,
                                                       const int32_t in_offset,
                                                       const int32_t act_min,
                                                       const int32_t act_max,
                                                       const uint16_t out_tensor_dim_x,
                                                       const uint16_t out_tensor_dim_y,
                                                       int8_t *tmp_buf)
{
    (void)pad_offset_x;
    (void)pad_offset_y;

    int32_t *out_tmp_buf = (int32_t*)tmp_buf;

    for (int i_batch = 0; i_batch < in_tensor_batch; i_batch++)
    {
        riscv_nn_set_zero_s8((int8_t*)out_tmp_buf, sizeof(*out_tmp_buf) * out_tensor_dim_y * out_tensor_dim_x * out_tensor_ch);

        for (int i_in_y = 0; i_in_y < in_tensor_dim_y; i_in_y++)
        {
            for (int i_in_x = 0; i_in_x < in_tensor_dim_x; i_in_x++)
            {
                for (int i_in_ch = 0; i_in_ch < in_tensor_ch; i_in_ch++)
                {
                    const int out_x_origin = (i_in_x * stride_x) - pad_x;
                    const int out_y_origin = (i_in_y * stride_y) - pad_y;

                    for (int i_ker_y = 0; i_ker_y < ker_dim_y; i_ker_y++)
                    {
                        for (int i_ker_x = 0; i_ker_x < ker_dim_x; i_ker_x++)
                        {
                            for (int i_out_ch = 0; i_out_ch < out_tensor_ch; i_out_ch++)
                            {
                                const int i_out_x = out_x_origin + i_ker_x;
                                const int i_out_y = out_y_origin + i_ker_y;

                                if ((i_out_x >= 0) && (i_out_x < out_tensor_dim_x) && (i_out_y >= 0) && (i_out_y < out_tensor_dim_y))
                                {
                                    int8_t in_val  = in_tensor[(i_in_y * in_tensor_dim_x + i_in_x) * in_tensor_ch + i_in_ch];
                                    int8_t wt_val = ker_weight[i_out_ch * in_tensor_ch * ker_dim_y * ker_dim_x + (i_ker_y * ker_dim_x + i_ker_x) * in_tensor_ch + i_in_ch];

                                    out_tmp_buf[(i_out_y * out_tensor_dim_x + i_out_x) * out_tensor_ch + i_out_ch] += (in_val + in_offset) * wt_val;

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
                for (int i_out_ch = 0; i_out_ch < out_tensor_ch; ++i_out_ch)
                {
                    int32_t acc = out_tmp_buf[(i_out_y * out_tensor_dim_x + i_out_x) * out_tensor_ch + i_out_ch];
                    if (bias)
                    {
                        acc += bias[i_out_ch];
                    }

                    acc = riscv_nn_requantize(acc, out_scale[i_out_ch], out_shift[i_out_ch]);
                    acc += out_offset;
                    acc = MAX(acc, act_min);
                    acc = MIN(acc, act_max);
                    out_tensor[i_out_ch + (i_out_y * out_tensor_dim_x + i_out_x) * out_tensor_ch] = (int8_t)acc;
                }
            }
        }

        in_tensor += (in_tensor_dim_x * in_tensor_dim_y * in_tensor_ch);
        out_tensor += (out_tensor_dim_x * out_tensor_dim_y * out_tensor_ch);
    }

    return 0;
}

uint32_t riscv_nn_conv_trans_HWC_s8_s8_s8_asym_bias_any_get_buffer_size(const uint16_t in_tensor_dim_x,
                                                                        const uint16_t in_tensor_dim_y,
                                                                        const uint16_t in_tensor_ch,
                                                                        const uint16_t in_tensor_batch,
                                                                        const uint16_t out_tensor_ch,
                                                                        const uint16_t ker_dim_x,
                                                                        const uint16_t ker_dim_y,
                                                                        const uint16_t pad_x,
                                                                        const uint16_t pad_y,
                                                                        const uint16_t stride_x,
                                                                        const uint16_t stride_y,
                                                                        const uint16_t out_tensor_dim_x,
                                                                        const uint16_t out_tensor_dim_y)
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

    //Note. the element of partial results is 4 byte)
    //temporary buffer to keep the partial results
    buf_size = (out_tensor_ch * out_tensor_dim_y * out_tensor_dim_x) * sizeof(int32_t);

    return buf_size;
}
