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

int32_t riscv_nn_maxpool_HWC_s16_any_act(const int32_t in_tensor_batch,
                                         const int32_t in_tensor_dim_y,
                                         const int32_t in_tensor_dim_x,
                                         const int32_t out_tensor_dim_y,
                                         const int32_t out_tensor_dim_x,
                                         const int32_t stride_y,
                                         const int32_t stride_x,
                                         const int32_t ker_dim_y,
                                         const int32_t ker_dim_x,
                                         const int32_t pad_y,
                                         const int32_t pad_x,
                                         const int32_t act_min,
                                         const int32_t act_max,
                                         const int32_t in_tensor_ch,
                                         int16_t * in_tensor,
                                         int16_t * tmp_buffer,
                                         int16_t * out_tensor)
{
    int32_t i_batch, i_ch_in, i_out_x, i_out_y;
    int32_t i_ker_x, i_ker_y;
    (void)tmp_buffer;
    const int32_t input_len = in_tensor_dim_y * in_tensor_dim_x * in_tensor_ch;
    const int32_t output_len = out_tensor_dim_y * out_tensor_dim_x * in_tensor_ch;

    for (i_batch = 0; i_batch < in_tensor_batch; i_batch++)
    {
        for (i_out_y = 0; i_out_y < out_tensor_dim_y; i_out_y++)
        {
            for (i_out_x = 0; i_out_x < out_tensor_dim_x; i_out_x++)
            {
                for (i_ch_in = 0; i_ch_in < in_tensor_ch; i_ch_in++)
                {
                    int32_t max_val = Q15_MIN;
                    const int32_t base_idx_y = (i_out_y * stride_y) - pad_y;
                    const int32_t base_idx_x = (i_out_x * stride_x) - pad_x;
                    const int32_t ker_y_start = MAX(0, -base_idx_y);
                    const int32_t ker_x_start = MAX(0, -base_idx_x);
                    const int32_t ker_y_end = MIN(ker_dim_y, in_tensor_dim_y - base_idx_y);
                    const int32_t ker_x_end = MIN(ker_dim_x, in_tensor_dim_x - base_idx_x);

                    for (i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
                    {
                        for (i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                        {
                            const int32_t col_idx = base_idx_x + i_ker_x;
                            const int32_t row_idx = base_idx_y + i_ker_y;

                            max_val = MAX(in_tensor[(row_idx * in_tensor_dim_x + col_idx) * in_tensor_ch + i_ch_in], max_val);
                        }
                    }

                    max_val = MAX(max_val, act_min);
                    max_val = MIN(max_val, act_max);

                    out_tensor[i_ch_in + in_tensor_ch * (i_out_x + i_out_y * out_tensor_dim_x)] = max_val;
                }
            }
        }
        in_tensor += input_len;
        out_tensor += output_len;
    }

    return 0;
}
