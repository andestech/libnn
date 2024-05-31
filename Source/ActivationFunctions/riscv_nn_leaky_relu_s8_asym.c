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

/** @file*/

#include "internal_nn_math.h"

//// Relu Functions

void riscv_nn_leaky_relu_s8_asym(q7_t * in_vec,
        q7_t * out_vec,
        const uint32_t size,
        const int32_t multi_identity,
        const int32_t shift_identity,
        const int32_t multi_alpha,
        const int32_t shift_alpha,
        const int32_t in_offset,
        const int32_t out_offset,
        const q7_t act_min,
        const q7_t act_max)
{
    for (int i = 0; i < size; ++i)
    {
        const int32_t val = in_vec[i] - in_offset;
        int32_t unclamped_output;
        if (val >= 0)
        {
            unclamped_output = out_offset + riscv_nn_requantize(val,
                                                multi_identity,
                                                shift_identity);
        }
        else
        {
            unclamped_output = out_offset + riscv_nn_requantize(val,
                                                multi_alpha,
                                                shift_alpha);
        }

        const int32_t clamped_output = MIN(act_max, MAX(act_min, unclamped_output));
        out_vec[i] = (q7_t)clamped_output;
    }
}
