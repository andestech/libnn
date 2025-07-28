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

int32_t riscv_nn_ew_mulc_s8_asym(const int8_t * in_vec,
                                 const int32_t in_const,    //rnage:[-255, 255]
                                 const int32_t in_offset,   //range:[-128, 127]
                                 int8_t * out_vec,
                                 const int32_t out_offset,  //range:[-128, 127]
                                 const int32_t out_scale,
                                 const int32_t out_shift,
                                 const int32_t act_min,
                                 const int32_t act_max,
                                 const uint32_t size)
{
    uint32_t loop = 0;
    int32_t output;
    loop = size;
    int32_t in1 = 0;
    while (loop > 0)
    {
        in1 = *in_vec++ + in_offset;
        output = in1 * in_const;// max:255*255
        output = riscv_nn_requantize(output, out_scale, out_shift) + out_offset;
        output = riscv_nn_clip_any(output, act_min, act_max);

        *out_vec++ = (int8_t)output;
        loop--;
    }

    return 0;
}
