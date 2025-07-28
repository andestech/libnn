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

#include <float.h>
#include "internal_nn_math.h"

//// Util Functions

int32_t riscv_nn_argmax_f32(const float32_t * in_tensor,
                            const uint32_t in_tensor_dim_y, //axis-0
                            const uint32_t in_tensor_dim_x, //axis-1
                            const uint8_t axis,
                            uint32_t * out_idx)
{
    if (axis > 1)
    {
        // unsupported axis
        return -1;
    }

    if (axis == 0)
    {
        for (uint32_t cur_x = 0; cur_x < in_tensor_dim_x; cur_x++)
        {
            float max_value = -FLT_MAX;
            uint32_t max_index = 0;
            for (uint32_t cur_y = 0; cur_y < in_tensor_dim_y; cur_y++)
            {
                float val = in_tensor[cur_y * in_tensor_dim_x + cur_x];
                if (val > max_value)
                {
                    max_value = val;
                    max_index = cur_y;
                }
            }
            out_idx[cur_x] = max_index;
        }
    }
    else if (axis == 1)
    {
        for (uint32_t cur_y = 0; cur_y < in_tensor_dim_y; cur_y++)
        {
            float max_value = -FLT_MAX;
            uint32_t max_index = 0;
            for (uint32_t cur_x = 0; cur_x < in_tensor_dim_x; cur_x++)
            {
                float val = in_tensor[cur_y * in_tensor_dim_x + cur_x];
                if (val > max_value)
                {
                    max_value = val;
                    max_index = cur_x;
                }
            }
            out_idx[cur_y] = max_index;
        }
    }

    return 0;
}
