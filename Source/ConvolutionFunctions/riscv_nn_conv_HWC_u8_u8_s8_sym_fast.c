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

#include "internal_nn_math.h"
#include "riscv_nn_support.h"

//// Convolution Functions
int32_t riscv_nn_conv_HWC_u8_u8_s8_sym_fast(const u8_t * in_tensor,
                                     const uint16_t in_tensor_dim,
                                     const uint16_t in_tensor_ch,
                                     const q7_t * ker_weight,
                                     const uint16_t out_tensor_ch,
                                     const uint16_t ker_dim,
                                     const uint16_t pad,
                                     const uint16_t stride,
                                     const uint16_t pre_rshift,
                                     const uint16_t out_scale,
                                     const uint16_t post_rshift,
                                     u8_t * out_tensor,
                                     const uint16_t out_tensor_dim,
                                     q15_t * in_tmp_buf)
{
    if (in_tensor_ch % 4 != 0 || out_tensor_ch % 2 != 0)
    {
        /* check if the input dimension meets the constraints */
        return -1;
    }

    uint16_t  i, j, k, l, m, n;
    int       conv_out;
    long in_row, in_col;

    for (i = 0; i < out_tensor_ch; i++)
    {
        for (j = 0; j < out_tensor_dim; j++)
        {
            for (k = 0; k < out_tensor_dim; k++)
            {
                conv_out = 0;
                for (m = 0; m < ker_dim; m++)
                {
                    for (n = 0; n < ker_dim; n++)
                    {
                        // if-for implementation
                        in_row = stride * j + m - pad;
                        in_col = stride * k + n - pad;
                        if (in_row >= 0 && in_col >= 0 && in_row < in_tensor_dim && in_col < in_tensor_dim)
                        {
                            for (l = 0; l < in_tensor_ch; l++)
                            {
                                conv_out +=
                                    in_tensor[(in_row * in_tensor_dim + in_col) * in_tensor_ch + l]
                                    * ker_weight[i * in_tensor_ch * ker_dim * ker_dim + (m * ker_dim + n) * in_tensor_ch + l];
                            }
                        }
                    }
                }
                if (conv_out < 0)
                    out_tensor[i + (j * out_tensor_dim + k) * out_tensor_ch] = 0;
                else
                {
                    conv_out = (conv_out >> pre_rshift) * out_scale + NN_ROUND(post_rshift);
                    out_tensor[i + (j * out_tensor_dim + k) * out_tensor_ch] = NDS_ISA_SAT((conv_out >> post_rshift), 8);
                }
            }
        }
    }

    /* Return to application */
    return 0;
}
