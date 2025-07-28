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

int32_t riscv_nn_dequantize_s8_f32(const int8_t * in_tensor,
                                   const uint32_t size,
                                   const float32_t in_scale,
                                   const int32_t in_zero_point,
                                   float32_t * out_tensor)
{
    uint32_t idx;

    for (idx = 0; idx < size; idx++)
    {
        out_tensor[idx] = in_scale * ((int32_t) in_tensor[idx] - in_zero_point);
    }
    return 0;
}
