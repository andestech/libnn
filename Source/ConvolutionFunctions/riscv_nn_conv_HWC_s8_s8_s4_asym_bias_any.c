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

int32_t riscv_nn_conv_HWC_s8_s8_s4_asym_bias_any(const int8_t * in_tensor,
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
                                                 const int32_t stride_x,
                                                 const int32_t stride_y,
                                                 const int32_t * bias,
                                                 int8_t * out_tensor,
                                                 const int32_t * out_shift,
                                                 const int32_t * out_scale,
                                                 const int32_t out_offset,    //value is in the range of [-128, 127]
                                                 const int32_t in_offset,     //value is in the range of [-127, 128]
                                                 const int32_t act_min,
                                                 const int32_t act_max,
                                                 const int32_t out_tensor_dim_x,
                                                 const int32_t out_tensor_dim_y,
                                                 const int32_t dilation_x,
                                                 const int32_t dilation_y,
                                                 int8_t * in_tmp_buf)
{
    (void)in_tmp_buf;
    int32_t i_out_ch, i_out_y, i_out_x, i_input_ch, i_ker_y, i_ker_x;
    int32_t conv_out;

    int32_t i_batch;
    for (i_batch = 0; i_batch < in_tensor_batch; i_batch++)
    {
        for (i_out_ch = 0; i_out_ch < out_tensor_ch; i_out_ch++)
        {
            for (i_out_y = 0; i_out_y < out_tensor_dim_y; i_out_y++)
            {
                for (i_out_x = 0; i_out_x < out_tensor_dim_x; i_out_x++)
                {
                    conv_out = 0;
                    if (bias)
                    {
                        conv_out = bias[i_out_ch];
                    }

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

                    for (i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
                    {
                        for (i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                        {
                            const int32_t in_row = base_idx_y + dilation_y * i_ker_y;
                            const int32_t in_col = base_idx_x + dilation_x * i_ker_x;
                            int8_t *pIn = (int8_t*)&in_tensor[(in_row * in_tensor_dim_x + in_col) * in_tensor_ch];
                            long wt_idx_4b = i_out_ch * in_tensor_ch * ker_dim_y * ker_dim_x + (i_ker_y * ker_dim_x + i_ker_x) * in_tensor_ch;
                            long wt_idx_8b = wt_idx_4b >> 1;
                            int8_t *pWt = (int8_t*)&ker_weight[wt_idx_8b];

                            if (wt_idx_4b & 0x1)   // odd index
                            {
                                {
                                    int8_t in_val = *pIn;
                                    int8_t packed_wt_val = *pWt;
                                    int8_t wt_val = (int8_t)(packed_wt_val >> 4);
                                    pIn++;
                                    pWt++;

                                    conv_out += ((int32_t)in_val + in_offset) * wt_val;
                                }
                                for (i_input_ch = 1; (i_input_ch+2) <= in_tensor_ch; i_input_ch += 2)
                                {
                                    int8_t in_val_0 = *pIn;
                                    int8_t in_val_1 = *(pIn+1);
                                    int8_t packed_wt_val = *pWt;
                                    int8_t wt_val_0 = (int8_t)(packed_wt_val << 4) >> 4;
                                    int8_t wt_val_1 = (int8_t)(packed_wt_val >> 4);
                                    pIn += 2;
                                    pWt++;

                                    conv_out += ((int32_t)in_val_0 + in_offset) * wt_val_0;
                                    conv_out += ((int32_t)in_val_1 + in_offset) * wt_val_1;
                                }
                            }
                            else
                            {
                                for (i_input_ch = 0; (i_input_ch+2) <= in_tensor_ch; i_input_ch += 2)
                                {
                                    int8_t in_val_0 = *pIn;
                                    int8_t in_val_1 = *(pIn+1);
                                    int8_t packed_wt_val = *pWt;
                                    int8_t wt_val_0 = (int8_t)(packed_wt_val << 4) >> 4;
                                    int8_t wt_val_1 = (int8_t)(packed_wt_val >> 4);
                                    pIn += 2;
                                    pWt++;

                                    conv_out += ((int32_t)in_val_0 + in_offset) * wt_val_0;
                                    conv_out += ((int32_t)in_val_1 + in_offset) * wt_val_1;
                                }
                                if (in_tensor_ch & 0x1)
                                {
                                    int8_t in_val = *pIn;
                                    int8_t packed_wt_val = *pWt;
                                    int8_t wt_val = (int8_t)(packed_wt_val << 4) >> 4;

                                    conv_out += ((int32_t)in_val + in_offset) * wt_val;
                                }
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

        /* Advance to the next batch */
        in_tensor += (in_tensor_dim_x * in_tensor_dim_y * in_tensor_ch);
        out_tensor += (out_tensor_dim_x * out_tensor_dim_y * out_tensor_ch);
    }

    /* Return to application */
    return 0;
}

int32_t riscv_nn_conv_HWC_s8_s8_s4_asym_bias_any_get_buffer_size(const int32_t in_tensor_ch,
                                                                 const int32_t ker_dim_x,
                                                                 const int32_t ker_dim_y,
                                                                 const int32_t out_tensor_ch)
{
    // return the required buffer size in byte
    (void)in_tensor_ch;
    (void)ker_dim_x;
    (void)ker_dim_y;
    return 0;
}
