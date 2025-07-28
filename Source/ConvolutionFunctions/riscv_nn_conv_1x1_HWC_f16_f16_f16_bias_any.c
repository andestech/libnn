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
int32_t riscv_nn_conv_1x1_HWC_f16_f16_f16_bias_any(const float16_t * in_tensor,
                                                const uint16_t in_tensor_dim_x,
                                                const uint16_t in_tensor_dim_y,
                                                const uint16_t in_tensor_ch,
                                                const float16_t * ker_weight,
                                                const uint16_t out_tensor_ch,
                                                const uint16_t ker_dim_x,
                                                const uint16_t ker_dim_y,
                                                const uint16_t pad_x,
                                                const uint16_t pad_y,
                                                const uint16_t stride_x,
                                                const uint16_t stride_y,
                                                const float16_t * bias,
                                                float16_t * out_tensor,
                                                const uint16_t out_tensor_dim_x,
                                                const uint16_t out_tensor_dim_y,
                                                float16_t * in_tmp_buf,
                                                float16_t * tmp_buf)
{
    if (in_tensor_ch % 4 != 0 || out_tensor_ch % 2 != 0 || ker_dim_x != 1 || ker_dim_y != 1
            || pad_x != 0 || pad_y != 0 || stride_x != 1 || stride_y != 1)
    {
        /* check if the input dimension meets the constraints */
        return -1;
    }

    //Note. the out_tensor dimension should be the same as in_tensor's.

    int         i, j, k, l, m, n, in_row, in_col;
    float16_t   conv_out;

    for (i = 0; i < out_tensor_ch; i++)
    {
        for (j = 0; j < out_tensor_dim_y; j++)
        {
            for (k = 0; k < out_tensor_dim_x; k++)
            {
                conv_out = bias[i];

                for (m = 0; m < ker_dim_y; m++)
                {
                    for (n = 0; n < ker_dim_x; n++)
                    {
                        in_row = stride_y * j + m - pad_y;
                        in_col = stride_x * k + n - pad_x;
                        if (in_row >= 0 && in_col >= 0 && in_row < in_tensor_dim_y && in_col < in_tensor_dim_x)
                        {
                            for (l = 0; l < in_tensor_ch; l++)
                            {
                                conv_out += in_tensor[(in_row * in_tensor_dim_x + in_col) * in_tensor_ch + l] *
                                            ker_weight[i * in_tensor_ch * ker_dim_y * ker_dim_x + (m * ker_dim_y + n) * in_tensor_ch + l];
                            }
                        }
                    }
                }
                out_tensor[i + (j * out_tensor_dim_x + k) * out_tensor_ch] = conv_out;
            }
        }
    }

    return 0;
}

uint32_t riscv_nn_conv_1x1_HWC_f16_f16_f16_bias_any_get_buffer_size(const uint16_t in_tensor_dim_x,
                                                                   const uint16_t in_tensor_dim_y,
                                                                   const uint16_t in_tensor_ch,
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
    (void) in_tensor_dim_x;
    (void) in_tensor_dim_y;
    (void) ker_dim_x;
    (void) ker_dim_y;
    (void) pad_x;
    (void) pad_y;
    (void) stride_x;
    (void) stride_y;
    (void) out_tensor_dim_x;
    (void) out_tensor_dim_y;
    uint32_t buf_size = 0;
    return buf_size;
}
