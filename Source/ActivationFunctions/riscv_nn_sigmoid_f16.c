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

//// Activations Functions

int32_t riscv_nn_sigmoid_f16(const float16_t * in_vec, uint32_t size, float16_t * out_vec)
{

    //--- const values for sigmoid ---
    const float16_t cst_1 = CST_1;
#ifdef ENA_KERNEL_FP32
    const float16_t sigmoid_min = SIGMOID_MIN;
    const float16_t sigmoid_max = SIGMOID_MAX;
#else
    const float16_t sigmoid_min = EXP_F16_MIN;
    const float16_t sigmoid_max = SIGMOID_MAX;
#endif

#if   defined(ENA_KERNEL_FP32)
    long i;
    for(i=0; i<size; i++)
    {
        float16_t x = in_vec[i];
        x = (x < sigmoid_min)? sigmoid_min : x;
        x = (x > sigmoid_max)? sigmoid_max : x;
        float16_t num = exp_f32(x);
        float16_t den = num + cst_1;
        out_vec[i] = num / den;
    }
#else
    long i;
    for(i=0; i<size; i++)
    {
        float16_t x = in_vec[i];
        x = (x < sigmoid_min)? sigmoid_min : x;
        x = (x > sigmoid_max)? sigmoid_max : x;
        float16_t num = exp_f16(x);
        float16_t den = num + cst_1;
        out_vec[i] = num / den;
    }
#endif

    return 0;
}
