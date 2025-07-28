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

//// Util Functions

int32_t riscv_nn_gather_HWC_s16(const int16_t * in_tensor,
                                const uint32_t in_tensor_dim_x,
                                const uint32_t in_tensor_dim_y,
                                const uint32_t in_tensor_ch,
                                const uint32_t in_tensor_batch,
                                const uint32_t gather_idx,
                                const uint32_t axis, // 0-3
                                int16_t * out_tensor)
{
    int64_t start_n = 0L, end_n = (int64_t)in_tensor_batch - 1L;
    int64_t start_h = 0L, end_h = (int64_t)in_tensor_dim_y - 1L;
    int64_t start_w = 0L, end_w = (int64_t)in_tensor_dim_x - 1L;
    int64_t start_c = 0L, end_c = (int64_t)in_tensor_ch    - 1L;
    int64_t i_in_n = 0L, i_in_h = 0L, i_in_w = 0L, i_in_c = 0L, i_src = 0L, i_dst = 0L;
    int64_t tensor_w_step = (int64_t)in_tensor_ch;
    int64_t tensor_h_step = (int64_t)in_tensor_dim_x * tensor_w_step;
    int64_t tensor_n_step = (int64_t)in_tensor_dim_y * tensor_h_step;

    switch (axis)
    {
        case 0u: // N
            start_n = (int64_t)gather_idx;
            end_n = start_n;
            break;
        case 1u: // H
            start_h = (int64_t)gather_idx;
            end_h = start_h;
            break;
        case 2u: // W
            start_w = (int64_t)gather_idx;
            end_w = start_w;
            break;
        case 3u: // C
            start_c = (int64_t)gather_idx;
            end_c = start_c;
            break;
        default:
            // unsupported axis
            return -1;
            break;
    }

    for (i_in_n = start_n; i_in_n <= end_n; i_in_n++)
    {
        for (i_in_h = start_h; i_in_h <= end_h; i_in_h++)
        {
            for (i_in_w = start_w; i_in_w <= end_w; i_in_w++)
            {
                for (i_in_c = start_c; i_in_c <= end_c; i_in_c++)
                {
                    i_src = i_in_n * tensor_n_step + i_in_h * tensor_h_step +
                            i_in_w * tensor_w_step + i_in_c;
                    out_tensor[i_dst++] = in_tensor[i_src];
                }
            }
        }
    }

    return 0;
}
