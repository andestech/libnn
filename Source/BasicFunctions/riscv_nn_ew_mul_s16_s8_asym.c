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

//// Basic Operation Functions

int riscv_nn_ew_mul_s16_s8_asym(const int16_t *in_vec1,
                                const int16_t *in_vec2,
                                int8_t *out_vec,
                                const int32_t out_offset,
                                const int32_t out_scale,
                                const int32_t out_shift,
                                const int32_t size)
{
    int32_t loop;
    loop = size;
    while (loop > 0)
    {
        /* C = A * B */
        int32_t mul_res = (*in_vec1) * (*in_vec2);
        mul_res = riscv_nn_requantize(mul_res, out_scale, out_shift) + out_offset;

        mul_res = MAX(mul_res, ((int8_t) 0x80));
        mul_res = MIN(mul_res, ((int8_t) 0x7f));

        *out_vec++ = (int8_t)mul_res;
        in_vec1++;
        in_vec2++;
        loop--;
    }
    return 0;
}
