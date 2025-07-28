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

#include "internal_nn_math.h"
#include "riscv_nn_support.h"

int32_t riscv_nn_reverseV2_s8(const int8_t * in_tensor,
                              const uint32_t in_tensor_dim_w,
                              const uint32_t in_tensor_dim_z,
                              const uint32_t in_tensor_dim_y,
                              const uint32_t in_tensor_dim_x,
                              const uint32_t axis, // 0, 1, 2, 3
                              int8_t * out_tensor)
{
    int64_t start_W = 0L, end_W = (int64_t)in_tensor_dim_w - 1L;
    int64_t start_Z = 0L, end_Z = (int64_t)in_tensor_dim_z - 1L;
    int64_t start_Y = 0L, end_Y = (int64_t)in_tensor_dim_y - 1L;
    int64_t start_X = 0L, end_X = (int64_t)in_tensor_dim_x - 1L;
    int64_t i_in_W = 0L, i_in_Z = 0L, i_in_Y = 0L, i_in_X = 0L, i_src = 0L, i_dst = 0L;
    int64_t tensor_Y_step = (int64_t)in_tensor_dim_x;
    int64_t tensor_Z_step = (int64_t)in_tensor_dim_y * tensor_Y_step;
    int64_t tensor_W_step = (int64_t)in_tensor_dim_z * tensor_Z_step;

    for (i_in_W = start_W; i_in_W <= end_W; i_in_W++)
    {
        for (i_in_Z = start_Z; i_in_Z <= end_Z; i_in_Z++)
        {
            for (i_in_Y = start_Y; i_in_Y <= end_Y; i_in_Y++)
            {
                for (i_in_X = start_X; i_in_X <= end_X; i_in_X++)
                {
                    switch (axis)
                    {
                    case 0u:  // N
                        i_src = (end_W - i_in_W) * tensor_W_step + i_in_Z * tensor_Z_step +
                                i_in_Y * tensor_Y_step + i_in_X;
                        break;
                    case 1u:  // H
                        i_src = i_in_W * tensor_W_step + (end_Z - i_in_Z) * tensor_Z_step +
                                i_in_Y * tensor_Y_step + i_in_X;
                        break;
                    case 2u:  // W
                        i_src = i_in_W * tensor_W_step + i_in_Z * tensor_Z_step +
                                (end_Y - i_in_Y) * tensor_Y_step + i_in_X;
                        break;
                    case 3u:  // C
                        i_src = i_in_W * tensor_W_step + i_in_Z * tensor_Z_step +
                                i_in_Y * tensor_Y_step + (end_X - i_in_X);
                        break;
                    }
                    out_tensor[i_dst++] = in_tensor[i_src];
                }
            }
        }
    }
    return 0;
}
