/******************************************************************************
 * Copyright (C) 2010-2018 Arm Limited or its affiliates. All rights reserved.*
 * Copyright (C) 2018-2023 Andes Technology Corporation. All rights reserved. *
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

void riscv_nn_concate_s8_x(const int8_t *in_tensor,
                        const uint16_t in_tensor_x,
                        const uint16_t in_tensor_y,
                        const uint16_t in_tensor_z,
                        const uint16_t in_tensor_w,
                        int8_t *out_tensor,
                        const uint16_t out_tensor_x,
                        const uint32_t out_offset_x)
{
    const uint32_t num_iterations = in_tensor_y * in_tensor_z * in_tensor_w;

    out_tensor += out_offset_x;

    uint32_t i;

    // Copy per row
    for (i = 0; i < num_iterations; ++i)
    {
        memcpy(out_tensor, in_tensor, in_tensor_x);
        in_tensor  += in_tensor_x;
        out_tensor += out_tensor_x;
    }
}