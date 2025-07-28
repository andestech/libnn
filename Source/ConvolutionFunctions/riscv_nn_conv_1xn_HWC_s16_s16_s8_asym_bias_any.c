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

int32_t riscv_nn_conv_1xn_HWC_s16_s16_s8_asym_bias_any(const int16_t * in_tensor,
                                                       const int32_t in_tensor_dim_x,
                                                       const int32_t in_tensor_ch,
                                                       const int32_t in_tensor_batch,
                                                       const int8_t * ker_weight,
                                                       const int32_t ker_dim_x,
                                                       const int32_t pad_x,
                                                       const int32_t stride_x,
                                                       const int64_t * bias,
                                                       int16_t * out_tensor,
                                                       const int32_t * out_shift,
                                                       const int32_t * out_scale,
                                                       const int32_t out_offset,//no used.
                                                       const int32_t in_offset,//no used.
                                                       const int32_t act_min,
                                                       const int32_t act_max,
                                                       const int32_t out_tensor_ch,
                                                       const int32_t out_tensor_dim_x,
                                                       const int32_t dilation_x,
                                                       int16_t * in_tmp_buf)
{
    (void)in_offset;
    (void)out_offset;
    for (int i_batch = 0; i_batch < in_tensor_batch; i_batch++)
    {
        for (int32_t i_out_ch = 0; i_out_ch < out_tensor_ch; i_out_ch++)
        {
            const int32_t reduced_scale = REDUCE_MULTIPLIER(out_scale[i_out_ch]);

            for (int32_t base_idx_x = -pad_x, i_out_x = 0; i_out_x < out_tensor_dim_x; base_idx_x += stride_x, i_out_x++)
            {
                int64_t conv_out_acc = 0;
                const int32_t start_x_max = (-base_idx_x + dilation_x - 1) / dilation_x;
                const int32_t ker_x_start = MAX(0, start_x_max);
                const int32_t end_min_x = (in_tensor_dim_x - base_idx_x + dilation_x - 1) / dilation_x;
                const int32_t ker_x_end = MIN(ker_dim_x, end_min_x);
                // ker_weight: {O,W,I}
                for (int32_t i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                {
                    const int32_t in_col = base_idx_x + dilation_x * i_ker_x;
                    for (int32_t i_input_ch = 0; i_input_ch < in_tensor_ch; i_input_ch++)
                    {
                        // out = out + in * weight
                        conv_out_acc += in_tensor[in_col * in_tensor_ch + i_input_ch] *
                            ker_weight[i_out_ch * in_tensor_ch * ker_dim_x +
                                        i_ker_x * in_tensor_ch + i_input_ch];
                    }
                }
                if (bias)
                {
                    conv_out_acc += bias[i_out_ch];
                }

                int32_t conv_out = riscv_nn_requantize_s64(conv_out_acc, reduced_scale, out_shift[i_out_ch]);
                conv_out = MAX(conv_out, act_min);
                conv_out = MIN(conv_out, act_max);
                out_tensor[i_out_ch + i_out_x * out_tensor_ch] = (int16_t)conv_out;
            }

        }

        in_tensor += (in_tensor_dim_x * in_tensor_ch);
        out_tensor += (out_tensor_dim_x * out_tensor_ch);
    }
    return 0;
}
