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

int32_t riscv_nn_channel_shuffle_HWC_s8(int8_t * in_tensor,
                                        const uint32_t in_tensor_dim_x,
                                        const uint32_t in_tensor_dim_y,
                                        const uint32_t in_tensor_ch,
                                        const uint32_t in_tensor_batch,
                                        const uint32_t group,
                                        int8_t * out_tensor)
{
    uint32_t b, y, x, c, n, set;
    uint32_t i_src = 0u, i_dst = 0u;

    if ((in_tensor_ch % group) != 0)
    {
        return -1;
    }

    set = in_tensor_ch / group;

    for (b = 0; b < in_tensor_batch; b++)
    {
        for (y = 0; y < in_tensor_dim_y; y++)
        {
            for (x = 0; x < in_tensor_dim_x; x++)
            {
                for (c = 0; c < set; c++)
                {
                    for (n = 0; n < group; n++)
                    {
                        i_src = b * in_tensor_dim_y * in_tensor_dim_x * in_tensor_ch +
                                y * in_tensor_dim_x * in_tensor_ch +
                                x * in_tensor_ch +
                                (set * n + c);
                        out_tensor[i_dst++] = in_tensor[i_src];
                    }
                }
            }
        }
    }
    return 0;
}
