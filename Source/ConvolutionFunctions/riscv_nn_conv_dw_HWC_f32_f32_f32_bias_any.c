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

//// Convolution Functions

int32_t riscv_nn_conv_dw_HWC_f32_f32_f32_bias_any(float32_t * in_tensor,
                                                  const uint16_t in_tensor_batch,
                                                  const uint16_t in_tensor_dim_x,
											      const uint16_t in_tensor_dim_y,
                                                  const uint16_t in_tensor_ch,
                                                  const float32_t * ker_weight,
                                                  const uint16_t out_tensor_ch,
                                                  const uint16_t ker_dim_x,
											      const uint16_t ker_dim_y,
                                                  const uint16_t pad_x,
											      const uint16_t pad_y,
                                                  const uint16_t stride_x,
											      const uint16_t stride_y,
                                                  const float32_t * bias,
                                                  float32_t * out_tensor,
                                                  const uint16_t out_tensor_dim_x,
											      const uint16_t out_tensor_dim_y,
                                                  float32_t * tmp_buf)
{
    if (in_tensor_ch != out_tensor_ch)
    {
        return -1;
    }

    int i_batch, i_out_y, i_out_x, i_ch_out, i_ker_x, i_ker_y;
    for (i_batch = 0; i_batch < in_tensor_batch; i_batch++)
    {
        float32_t conv_out = 0.f;
        for (i_out_y = 0; i_out_y < out_tensor_dim_y; i_out_y++)
        {
            for (i_out_x = 0; i_out_x < out_tensor_dim_x; i_out_x++)
            {
                for (i_ch_out = 0; i_ch_out < out_tensor_ch; i_ch_out++)
                {
                    conv_out = bias[i_ch_out];
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
                    out_tensor[(i_out_y * out_tensor_dim_x + i_out_x) * out_tensor_ch + i_ch_out] = conv_out;
                }
            }
        }
        in_tensor += (in_tensor_dim_x * in_tensor_dim_y * in_tensor_ch);
        out_tensor += (out_tensor_dim_x * out_tensor_dim_y * out_tensor_ch);
    }

    return 0;
}
