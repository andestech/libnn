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
#include "internal_nn_math.h"

int32_t riscv_nn_reduce_sum_s16(const int16_t * in_tensor,
                                const uint32_t in_tensor_dim_x,
                                const uint32_t in_tensor_dim_y,
                                const uint32_t in_tensor_ch,
                                const uint32_t in_tensor_batch,
                                const uint8_t axis,
                                const int32_t in_offset,
                                const int32_t out_shift,
                                const int32_t out_scale,
                                const int32_t out_offset,
                                const int32_t act_min,
                                const int32_t act_max,
                                int16_t * out_tensor)
{
    (void)in_offset;
    (void)out_offset;

    if (axis >= 4)
    {
        return -1;
    }

    int32_t end_i0 = 1;
    int32_t end_i1 = 1;
    int32_t end_i2 = 1;
    int32_t reduced_dim = 0;

    int64_t input_stride = 1, input_stride0 = 1, input_stride1 = 1, input_stride2 = 1;

    switch (axis)
    {
    case 0: // N
        end_i0 = in_tensor_dim_y - 1L;
        end_i1 = in_tensor_dim_x - 1L;
        end_i2 = in_tensor_ch - 1L;
        reduced_dim = in_tensor_batch - 1L;

        input_stride0 = in_tensor_dim_x * in_tensor_ch;
        input_stride1 = in_tensor_ch;
        input_stride2 = 1L;
        input_stride = in_tensor_dim_y * in_tensor_dim_x * in_tensor_ch;
        break;
    case 1: // H
        end_i0 = in_tensor_batch - 1L;
        end_i1 = in_tensor_dim_x - 1L;
        end_i2 = in_tensor_ch - 1L;
        reduced_dim = in_tensor_dim_y - 1L;

        input_stride0 = in_tensor_dim_y * in_tensor_dim_x * in_tensor_ch;
        input_stride1 = in_tensor_ch;
        input_stride2 = 1L;
        input_stride = in_tensor_dim_x * in_tensor_ch;
        break;
    case 2: // W
        end_i0 = in_tensor_batch - 1L;
        end_i1 = in_tensor_dim_y - 1L;
        end_i2 = in_tensor_ch - 1L;
        reduced_dim = in_tensor_dim_x - 1L;
        input_stride0 = in_tensor_dim_y * in_tensor_dim_x * in_tensor_ch;
        input_stride1 = in_tensor_dim_x * in_tensor_ch;
        input_stride2 = 1L;
        input_stride = in_tensor_ch;
        break;
    case 3: // C
        end_i0 = in_tensor_batch - 1L;
        end_i1 = in_tensor_dim_y - 1L;
        end_i2 = in_tensor_dim_x - 1L;
        reduced_dim = in_tensor_ch - 1L;
        input_stride0 = in_tensor_dim_y * in_tensor_dim_x * in_tensor_ch;
        input_stride1 = in_tensor_dim_x * in_tensor_ch;
        input_stride2 = in_tensor_ch;
        input_stride = 1L;
        break;
    }

    const q31_t reduced_scale = REDUCE_MULTIPLIER(out_scale);
    int32_t i_dst = 0;

    for (int32_t i_in_i0 = 0; i_in_i0 <= end_i0; i_in_i0++)
    {
        for (int32_t i_in_i1 = 0; i_in_i1 <= end_i1; i_in_i1++)
        {
            for (int32_t i_in_i2 = 0; i_in_i2 <= end_i2; i_in_i2++)
            {
                int32_t i_src_base = i_in_i0 * input_stride0 + i_in_i1 * input_stride1 +
                                     i_in_i2 * input_stride2;
                int64_t sum = 0;
                for(int32_t j=0; j <= reduced_dim; j++)
                {
                    int32_t i_src = i_src_base + j * input_stride;
                    sum += in_tensor[i_src];
                }

                int32_t sum_out = riscv_nn_requantize_s64(sum, reduced_scale, out_shift);
                sum_out = MAX(sum_out, act_min);
                sum_out = MIN(sum_out, act_max);
                out_tensor[i_dst++] = (int16_t)sum_out;
            }
        }
    }
    return 0;
}
