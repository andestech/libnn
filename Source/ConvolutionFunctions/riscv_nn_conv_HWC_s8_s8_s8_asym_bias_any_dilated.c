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
int32_t riscv_nn_conv_HWC_s8_s8_s8_asym_bias_any_dilated(const int8_t * in_tensor,
                                                         const uint16_t in_tensor_dim_x,
                                                         const uint16_t in_tensor_dim_y,
                                                         const uint16_t in_tensor_ch,
                                                         const uint16_t in_tensor_batch,
                                                         const int8_t * ker_weight,
                                                         const uint16_t out_tensor_ch,
                                                         const uint16_t ker_dim_x,
                                                         const uint16_t ker_dim_y,
                                                         const uint16_t ker_ch,
                                                         const uint16_t pad_x,
                                                         const uint16_t pad_y,
                                                         const uint16_t stride_x,
                                                         const uint16_t stride_y,
                                                         const int32_t * bias,
                                                         int8_t * out_tensor,
                                                         const int32_t * out_shift,
                                                         const int32_t * out_scale,
                                                         const int32_t out_offset,    //value is in the range of [-128, 127]
                                                         const int32_t in_offset,     //value is in the range of [-127, 128]
                                                         const int32_t act_min,
                                                         const int32_t act_max,
                                                         const uint16_t out_tensor_dim_x,
                                                         const uint16_t out_tensor_dim_y,
                                                         const int32_t dilation_x,
                                                         const int32_t dilation_y,
                                                         int16_t * in_tmp_buf)
{
    (void)in_tmp_buf;
    const int32_t groups = in_tensor_ch / ker_ch;
    const int32_t out_ch_per_group = out_tensor_ch / groups;

    if (in_tensor_ch % groups != 0 || out_tensor_ch % groups != 0)
    {
        return -1;
    }

    for (int32_t i_batch = 0; i_batch < in_tensor_batch; i_batch++)
    {
        for (int32_t i_group = 0; i_group < groups; i_group++)
        {
            for (int32_t i_out_ch_in_group = 0; i_out_ch_in_group < out_ch_per_group; i_out_ch_in_group++)
            {
                for (int32_t i_out_y = 0; i_out_y < out_tensor_dim_y; i_out_y++)
                {
                    for (int32_t i_out_x = 0; i_out_x < out_tensor_dim_x; i_out_x++)
                    {
                        const int32_t base_idx_y = stride_y * i_out_y - pad_y;
                        const int32_t base_idx_x = stride_x * i_out_x - pad_x;
                        const int32_t start_y_max = (-base_idx_y + dilation_y - 1) / dilation_y;
                        const int32_t start_x_max = (-base_idx_x + dilation_x - 1) / dilation_x;
                        const int32_t ker_y_start = MAX(0, start_y_max);
                        const int32_t ker_x_start = MAX(0, start_x_max);
                        const int32_t end_min_y = (in_tensor_dim_y - base_idx_y + dilation_y - 1) / dilation_y;
                        const int32_t end_min_x = (in_tensor_dim_x - base_idx_x + dilation_x - 1) / dilation_x;
                        const int32_t ker_y_end = MIN(ker_dim_y, end_min_y);
                        const int32_t ker_x_end = MIN(ker_dim_x, end_min_x);
                        const int32_t i_out_ch = i_group * out_ch_per_group + i_out_ch_in_group;

                        int32_t conv_out = 0;
                        if (bias)
                        {
                            conv_out = bias[i_out_ch];
                        }

                        for (int32_t i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
                        {
                            for (int32_t i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                            {
                                const int32_t in_row = base_idx_y + dilation_y * i_ker_y;
                                const int32_t in_col = base_idx_x + dilation_x * i_ker_x;
                                for (int32_t i_ker_ch = 0; i_ker_ch < ker_ch; i_ker_ch++)
                                {
                                    int32_t i_input_ch = i_group * ker_ch + i_ker_ch;
                                    int32_t in_idx = (in_row * in_tensor_dim_x + in_col) * in_tensor_ch + i_input_ch;
                                    int32_t wt_idx = i_out_ch * ker_ch * ker_dim_y * ker_dim_x + (i_ker_y * ker_dim_x + i_ker_x) * ker_ch + i_ker_ch;
                                    int32_t in_val = in_tensor[in_idx] + in_offset;
                                    int32_t wt_val = ker_weight[wt_idx];
                                    conv_out += in_val * wt_val;
                                }
                            }
                        }
                        conv_out = riscv_nn_requantize(conv_out, out_scale[i_out_ch], out_shift[i_out_ch]);
                        conv_out += out_offset;
                        conv_out = MAX(conv_out, act_min);
                        conv_out = MIN(conv_out, act_max);
                        out_tensor[i_out_ch + (i_out_y * out_tensor_dim_x + i_out_x) * out_tensor_ch] = (int8_t)conv_out;
                    }
                }
            }
        }
        in_tensor += (in_tensor_dim_x * in_tensor_dim_y * in_tensor_ch);
        out_tensor += (out_tensor_dim_x * out_tensor_dim_y * out_tensor_ch);
    }

    /* Return to application */
    return 0;
}

int32_t riscv_nn_conv_HWC_s8_s8_s8_asym_bias_any_dilated_get_buffer_size(const uint16_t ker_ch,
                                                                         const uint16_t ker_dim_x,
                                                                         const uint16_t ker_dim_y,
                                                                         const uint16_t out_tensor_ch)
{
    (void)ker_ch;
    (void)ker_dim_x;
    (void)ker_dim_y;
    (void)out_tensor_ch;
    return 0;
}
