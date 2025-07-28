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

int32_t riscv_nn_pixel_shuffle_HWC_s8(const int8_t * in_tensor,
								      const uint32_t in_tensor_dim_x,
								      const uint32_t in_tensor_dim_y,
								      const uint32_t in_tensor_ch,
								      const uint32_t in_tensor_batch,
								      const uint32_t up_factor,
								      int8_t * out_tensor)
{

    const uint32_t up_factor_square = up_factor * up_factor;
    if (0u != in_tensor_ch % up_factor_square)
    {
        return -1;
    }

    uint32_t i_batch, i_out_y, i_out_x, i_out_c, i_in_c, i_src = 0u, i_dst = 0u;
    uint32_t i_in_y, offset_y, i_in_x, offset_x;

    uint32_t out_tensor_dim_x = in_tensor_dim_x * up_factor;
    uint32_t out_tensor_dim_y = in_tensor_dim_y * up_factor;
    uint32_t out_tensor_ch    = in_tensor_ch / up_factor_square;

    for (i_batch = 0u; i_batch < in_tensor_batch; i_batch++)  // N
    {
        for (i_out_y = 0u; i_out_y < out_tensor_dim_y; i_out_y++)  // H
        {
            i_in_y   = i_out_y / up_factor;
            offset_y = i_out_y % up_factor;

            for (i_out_x = 0u; i_out_x < out_tensor_dim_x; i_out_x++)  // W
            {
                i_in_x   = i_out_x / up_factor;
                offset_x = i_out_x % up_factor;

                for (i_out_c = 0u; i_out_c < out_tensor_ch; i_out_c++)  // C
                {
                    i_in_c = offset_y * up_factor + offset_x + i_out_c * up_factor * up_factor;

                    i_src = i_batch * in_tensor_dim_y * in_tensor_dim_x * in_tensor_ch +
                            i_in_y * in_tensor_dim_x * in_tensor_ch +
                            i_in_x * in_tensor_ch +
                            i_in_c;

                    out_tensor[i_dst++] = in_tensor[i_src];
                }
            }
        }
    }
    return 0;
}
