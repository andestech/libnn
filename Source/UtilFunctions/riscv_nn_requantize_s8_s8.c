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

//// Util Functions

int32_t riscv_nn_requantize_s8_s8(int8_t * in_vec,
                                  int8_t * out_vec,
                                  const uint32_t size,
                                  const int32_t out_scale,
                                  const int32_t out_shift,
                                  const int32_t in_offset,    //-zp: [-127, 128]
                                  const int32_t out_offset,   //+zp: [-128, 127]
                                  const int32_t act_min,
                                  const int32_t act_max)
{
    int32_t output;
    for (int i = 0; i < size; ++i)
    {
        int32_t tmp = in_vec[i] + in_offset;
        output = riscv_nn_requantize(tmp, out_scale, out_shift);
        output += out_offset;
        output = MAX(output, act_min);
        output = MIN(output, act_max);
        out_vec[i] = (int8_t)output;
    }
    return 0;
}
