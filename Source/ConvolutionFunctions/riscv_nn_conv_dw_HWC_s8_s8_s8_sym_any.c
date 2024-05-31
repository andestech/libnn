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

void conv_dw_HWC_s8_s8_s8_sym_any_chmult(const q7_t * in_tensor,
        const uint16_t in_tensor_dim_x,
        const uint16_t in_tensor_dim_y,
        const uint16_t in_tensor_ch,
        const q7_t * ker_weight,
        const uint16_t out_tensor_ch,
        const uint16_t ker_dim_x,
        const uint16_t ker_dim_y,
        const uint16_t pad_x,
        const uint16_t pad_y,
        const uint16_t stride_x,
        const uint16_t stride_y,
        const uint16_t pre_rshift,
        const uint16_t out_scale,
        const uint16_t post_rshift,
        q7_t * out_tensor,
        const uint16_t out_tensor_dim_x,
        const uint16_t out_tensor_dim_y,
        q15_t * in_tmp_buf)
{
    int i_out_y, i_out_x, i_ch_in, i_ch_mult;
    int i_ker_y, i_ker_x;
    int ch_mult = out_tensor_ch / in_tensor_ch;

    for (i_out_y = 0; i_out_y < out_tensor_dim_y; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < out_tensor_dim_x; i_out_x++)
        {
            for (i_ch_in = 0; i_ch_in < in_tensor_ch; i_ch_in++)
            {
                for (i_ch_mult = 0; i_ch_mult < ch_mult; i_ch_mult++)
                {
                    const int idx_out_ch = i_ch_mult + i_ch_in * ch_mult;
                    int conv_out = 0;
                    for (i_ker_y = 0; i_ker_y < ker_dim_y; i_ker_y++)
                    {
                        for (i_ker_x = 0; i_ker_x < ker_dim_x; i_ker_x++)
                        {
                            int in_row = stride_y * i_out_y + i_ker_y - pad_y;
                            int in_col = stride_x * i_out_x + i_ker_x - pad_x;
                            if (in_row >= 0 && in_col >= 0 && in_row < in_tensor_dim_y && in_col < in_tensor_dim_x)
                            {
                                conv_out += in_tensor[(in_row * in_tensor_dim_x + in_col) * in_tensor_ch + i_ch_in] *
                                            ker_weight[(i_ker_y * ker_dim_x + i_ker_x) * (in_tensor_ch * ch_mult) + idx_out_ch];
                            }
                        }
                    }
                    conv_out = (conv_out >> pre_rshift) * out_scale + NN_ROUND(post_rshift);
                    out_tensor[(i_out_y * out_tensor_dim_x + i_out_x) * out_tensor_ch + idx_out_ch] = NDS_ISA_SATS((conv_out >> post_rshift), 8);
                }
            }
        }
    }
}

void conv_dw_HWC_s8_s8_s8_sym_any(const q7_t * in_tensor,
        const uint16_t in_tensor_dim_x,
        const uint16_t in_tensor_dim_y,
        const uint16_t in_tensor_ch,
        const q7_t * ker_weight,
        const uint16_t out_tensor_ch,
        const uint16_t ker_dim_x,
        const uint16_t ker_dim_y,
        const uint16_t pad_x,
        const uint16_t pad_y,
        const uint16_t stride_x,
        const uint16_t stride_y,
        const uint16_t pre_rshift,
        const uint16_t out_scale,
        const uint16_t post_rshift,
        q7_t * out_tensor,
        const uint16_t out_tensor_dim_x,
        const uint16_t out_tensor_dim_y,
        q15_t * in_tmp_buf)
{
    int i_out_y, i_out_x, i_ch_out;
    int i_ker_y, i_ker_x;

    for (i_out_y = 0; i_out_y < out_tensor_dim_y; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < out_tensor_dim_x; i_out_x++)
        {
            for (i_ch_out = 0; i_ch_out < out_tensor_ch; i_ch_out++)
            {
                int conv_out = 0;
                for (i_ker_y = 0; i_ker_y < ker_dim_y; i_ker_y++)
                {
                    for (i_ker_x = 0; i_ker_x < ker_dim_x; i_ker_x++)
                    {
                        int in_row = stride_y * i_out_y + i_ker_y - pad_y;
                        int in_col = stride_x * i_out_x + i_ker_x - pad_x;
                        if (in_row >= 0 && in_col >= 0 && in_row < in_tensor_dim_y && in_col < in_tensor_dim_x)
                        {
                            conv_out += in_tensor[(in_row * in_tensor_dim_x + in_col) * in_tensor_ch + i_ch_out] *
                                        ker_weight[(i_ker_y * ker_dim_x + i_ker_x) * out_tensor_ch + i_ch_out];
                        }
                    }
                }
                conv_out = (conv_out >> pre_rshift) * out_scale + NN_ROUND(post_rshift);
                out_tensor[(i_out_y * out_tensor_dim_x + i_out_x) * out_tensor_ch + i_ch_out] = NDS_ISA_SATS((conv_out >> post_rshift), 8);
            }
        }
    }
}

int32_t riscv_nn_conv_dw_HWC_s8_s8_s8_sym_any(const q7_t * in_tensor,
                                       const uint16_t in_tensor_dim_x,
                                       const uint16_t in_tensor_dim_y,
                                       const uint16_t in_tensor_ch,
                                       const q7_t * ker_weight,
                                       const uint16_t out_tensor_ch,
                                       const uint16_t ker_dim_x,
                                       const uint16_t ker_dim_y,
                                       const uint16_t pad_x,
                                       const uint16_t pad_y,
                                       const uint16_t stride_x,
                                       const uint16_t stride_y,
                                       const uint16_t pre_rshift,
                                       const uint16_t out_scale,
                                       const uint16_t post_rshift,
                                       q7_t * out_tensor,
                                       const uint16_t out_tensor_dim_x,
                                       const uint16_t out_tensor_dim_y,
                                       q15_t * in_tmp_buf)
{
    if (in_tensor_ch == out_tensor_ch)
    {
        conv_dw_HWC_s8_s8_s8_sym_any(in_tensor, in_tensor_dim_x,
            in_tensor_dim_y, in_tensor_ch, ker_weight, out_tensor_ch, ker_dim_x,
            ker_dim_y, pad_x, pad_y, stride_x, stride_y, pre_rshift, out_scale,
            post_rshift, out_tensor, out_tensor_dim_x, out_tensor_dim_y,
            in_tmp_buf);
    }
    else
    {
        conv_dw_HWC_s8_s8_s8_sym_any_chmult(in_tensor, in_tensor_dim_x,
            in_tensor_dim_y, in_tensor_ch, ker_weight, out_tensor_ch, ker_dim_x,
            ker_dim_y, pad_x, pad_y, stride_x, stride_y, pre_rshift, out_scale,
            post_rshift, out_tensor, out_tensor_dim_x, out_tensor_dim_y,
            in_tmp_buf);
    }
    return 0;
}
