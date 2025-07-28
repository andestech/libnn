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

static inline int32_t nn_fmagic_roundf_with_zp(float32_t x, int32_t zero_point)
{
    union Tmp {
        int32_t   asInt;
        float32_t asFloat;
    };
    union Tmp tmp;

    // fmagic algorithm. (roundToInteger)
    // float32_t fmagic = 12582912.0f;
    // z = (x + fmagic).asInt - (fmagic.asInt - zp)
    const float32_t fmagic = 12582912.0f;
    const int32_t imagic= (int32_t) 0x4B400000 - zero_point; // binary32 of 12582912.0f - zp

    tmp.asFloat = x + fmagic;
    return tmp.asInt - imagic;
}


int32_t riscv_nn_quantize_f32_s8(const float32_t * in_vec,
                                 const uint32_t size,
                                 int8_t * out_vec,
                                 const float32_t out_scale,
                                 const int32_t out_zero_point,// [-128, 127]
                                 const int32_t act_min,
                                 const int32_t act_max)
{
    int32_t result = 0;
    uint32_t idx;
    float32_t out_inv_scale = (float32_t) 1.f / out_scale;

    for (idx = 0; idx < size; idx++)
    {
        result = nn_fmagic_roundf_with_zp(out_inv_scale * (float32_t)in_vec[idx], out_zero_point);
        result = MAX(result, act_min);
        result = MIN(result, act_max);
        out_vec[idx] = (int8_t) result;
    }
    return 0;
}
