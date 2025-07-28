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

void riscv_nn_strided_slice_s8(const int8_t * in_tensor,
                               const uint32_t in_tensor_w,
                               const uint32_t in_tensor_z,
                               const uint32_t in_tensor_y,
                               const uint32_t in_tensor_x,
                               const uint32_t begin_w,
                               const uint32_t begin_z,
                               const uint32_t begin_y,
                               const uint32_t begin_x,
                               const uint32_t end_w,
                               const uint32_t end_z,
                               const uint32_t end_y,
                               const uint32_t end_x,
                               const uint32_t stride_w,
                               const uint32_t stride_z,
                               const uint32_t stride_y,
                               const uint32_t stride_x,
                               int8_t * out_tensor)
{
    uint32_t i_w, i_z, i_y, i_x, i_src = 0u, i_dst = 0u;

    for (i_w = begin_w; i_w < end_w; i_w += stride_w)
    {
        for (i_z = begin_z; i_z < end_z; i_z += stride_z)
        {
            for (i_y = begin_y; i_y < end_y; i_y += stride_y)
            {
                for (i_x = begin_x; i_x < end_x; i_x += stride_x)
                {
                    i_src = i_w * in_tensor_z * in_tensor_y * in_tensor_x +
                            i_z * in_tensor_y * in_tensor_x +
                            i_y * in_tensor_x + i_x;
                    out_tensor[i_dst++] = in_tensor[i_src];
                }
            }
        }
    }
}
