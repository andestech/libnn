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

//// Basic Operation Functions

/************************************************************************
* Note. NDS' shift amount (in_rshift1, in_rshift2 and out_rshift)       *
* are expected to be >=0; however, CMSIS' are expected to be <=0.       *
*************************************************************************/

int32_t riscv_nn_ew_rsubc_s16_asym(const int16_t * in_vec,
                                   const int32_t in_const,
                                   const int32_t in_offset,
                                   const int32_t in_scale,
                                   const int32_t in_rshift,
                                   const int32_t lshift,
                                   int16_t * out_vec,
                                   const int32_t out_offset,
                                   const int32_t out_scale,
                                   const int32_t out_rshift,
                                   const int32_t act_min,
                                   const int32_t act_max,
                                   const uint32_t size)
{
    (void)in_offset;
    (void)out_offset;

    int32_t loop;

    loop = size;
    while (loop > 0)
    {
        int32_t input_1 = *in_vec++ << lshift;

        input_1 = riscv_nn_requantize_ns(input_1, in_scale, -in_rshift);

        int32_t sum = in_const - input_1;
        sum = riscv_nn_requantize_ns(sum, out_scale, -out_rshift);

        sum = MAX(sum, act_min);
        sum = MIN(sum, act_max);

        *out_vec++ = sum;
        loop--;
    }

    return 0;
}
