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

void riscv_nn_pad_s16(const int16_t *in_tensor,
                      const uint32_t in_tensor_w,
                      const uint32_t in_tensor_z,
                      const uint32_t in_tensor_y,
                      const uint32_t in_tensor_x,
                      const uint32_t pre_pad_w,
                      const uint32_t pre_pad_z,
                      const uint32_t pre_pad_y,
                      const uint32_t pre_pad_x,
                      const uint32_t post_pad_w,
                      const uint32_t post_pad_z,
                      const uint32_t post_pad_y,
                      const uint32_t post_pad_x,
                      const int16_t pad_value,
                      int16_t *out_tensor)
{

    uint32_t out_w, out_z, out_y, out_x;
    uint32_t i_w, i_z, i_y, i_x, i_src = 0u, i_dst = 0u;

    out_w = in_tensor_w + pre_pad_w + post_pad_w;
    out_z = in_tensor_z + pre_pad_z + post_pad_z;
    out_y = in_tensor_y + pre_pad_y + post_pad_y;
    out_x = in_tensor_x + pre_pad_x + post_pad_x;

    for (i_w = 0; i_w < out_w; i_w++)
    {
        for (i_z = 0; i_z < out_z; i_z++)
        {
            for (i_y = 0; i_y < out_y; i_y++)
            {
                for (i_x = 0; i_x < out_x; i_x++)
                {
                    if( i_w < pre_pad_w || i_w >= (out_w - post_pad_w)
                     || i_z < pre_pad_z || i_z >= (out_z - post_pad_z)
                     || i_y < pre_pad_y || i_y >= (out_y - post_pad_y)
                     || i_x < pre_pad_x || i_x >= (out_x - post_pad_x) )
                    {
                        out_tensor[i_dst++] = pad_value;
                    }
                    else{
                        out_tensor[i_dst++] = in_tensor[i_src++];
                    }
                }
            }
        }
    }
}
