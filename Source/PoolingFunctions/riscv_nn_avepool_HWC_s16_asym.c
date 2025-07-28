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

//// Pooling Functions

int32_t riscv_nn_avepool_HWC_s16_asym(const int16_t * in_tensor,
                                      const uint16_t in_tensor_dim_x,
                                      const uint16_t in_tensor_dim_y,
                                      const uint16_t in_tensor_ch,
                                      const uint16_t ker_dim_x,
                                      const uint16_t ker_dim_y,
                                      const uint16_t pad_x,
                                      const uint16_t pad_y,
                                      const uint16_t stride_x,
                                      const uint16_t stride_y,
                                      int16_t * out_tensor,
                                      const int32_t out_shift,
                                      const int32_t out_scale,
                                      const int32_t out_round_pos,
                                      const int32_t out_round_neg,
                                      const uint16_t out_tensor_dim_x,
                                      const uint16_t out_tensor_dim_y,
                                      const int32_t out_offset,
                                      const int32_t in_offset,
                                      const int32_t act_min,
                                      const int32_t act_max)
{
    int16_t   i_ch_in, i_x, i_y;
    int16_t   k_x, k_y;

    for (i_ch_in = 0; i_ch_in < in_tensor_ch; i_ch_in++)
    {
        for (i_y = 0; i_y < out_tensor_dim_y; i_y++)
        {
            for (i_x = 0; i_x < out_tensor_dim_x; i_x++)
            {
                int       sum = 0;
                int       acc = 0;
                int64_t   new_val = 0;

                for (k_y = i_y * stride_y - pad_y; k_y < i_y * stride_y - pad_y + ker_dim_y; k_y++)
                {
                    for (k_x = i_x * stride_x - pad_x; k_x < i_x * stride_x - pad_x + ker_dim_x; k_x++)
                    {
                        if (k_y >= 0 && k_x >= 0 && k_y < in_tensor_dim_y && k_x < in_tensor_dim_x)
                        {
                            sum += (in_tensor[i_ch_in + in_tensor_ch * (k_x + k_y * in_tensor_dim_x)] + in_offset);
                        }
                    }
                }

                new_val = sum * (int64_t) out_scale;
                if (sum < 0)
                {
                    new_val += out_round_neg;
                }
                else
                {
                    new_val += out_round_pos;
                }

                acc = new_val >> out_shift;
                acc += out_offset;
                acc = MAX(acc, act_min);
                acc = MIN(acc, act_max);

                out_tensor[i_ch_in + (i_y * out_tensor_dim_x + i_x) * in_tensor_ch] = (int16_t)acc;
            }
        }
    }

    return 0;
}
