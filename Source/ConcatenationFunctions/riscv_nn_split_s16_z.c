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

//// Concatenation Functions

void riscv_nn_split_s16_z(const int16_t *in_tensor,
                        const uint32_t in_tensor_x,
                        const uint32_t in_tensor_y,
                        const uint32_t in_tensor_z,
                        const uint32_t in_tensor_w,
                        int16_t *out_tensor,
                        const uint32_t split_len_z,
                        const uint32_t in_offset)
{
    const uint32_t split_size = split_len_z * in_tensor_x * in_tensor_y;
    const uint32_t input_stride = in_tensor_z * in_tensor_y * in_tensor_x;

    in_tensor += in_offset;

    for (uint32_t i = 0; i < in_tensor_w; ++i)
    {
        memcpy(out_tensor, in_tensor, split_size * sizeof(*in_tensor));
        in_tensor += input_stride;
        out_tensor += split_size;
    }
}
