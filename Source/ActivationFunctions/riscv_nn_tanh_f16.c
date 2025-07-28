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

//// Activation Functions

//--- const values for tanh_f32 ---
#ifdef ENA_KERNEL_FP32
static const float16_t tanh_max    = TANH_F32_MAX;
static const float16_t tanh_min    = TANH_F32_MIN;
static const float32_t cst_1       = CST_1;
static const float16_t cst_2       = CST_2;
#else
static const float16_t tanh_max    = EXP_F16_MAX / 2.f;
static const float16_t tanh_min    = EXP_F16_MIN / 2.f;
static const float16_t cst_1       = CST_1;
static const float16_t cst_2       = CST_2;
#endif



float16_t tanh_f16(float16_t x)
{
    x = (x < tanh_min)? tanh_min : x;
    x = (x > tanh_max)? tanh_max : x;
    float16_t exp2x = exp_f16(x * cst_2);
    float16_t num = exp2x - cst_1;
    float16_t den = exp2x + cst_1;
    float16_t ret = num / den;

    return ret;
}

int32_t riscv_nn_tanh_f16(const float16_t * in_vec, uint32_t size, float16_t * out_vec)
{
#if   defined(ENA_KERNEL_FP32)
    for(long i=0; i<size; i++)
    {
        float16_t x = in_vec[i];
        x = (x < tanh_min)? tanh_min : x;
        x = (x > tanh_max)? tanh_max : x;
        float32_t exp2x = exp_f32(x * cst_2);
        float32_t num = exp2x - cst_1;
        float32_t den = exp2x + cst_1;
        out_vec[i] = num / den;
    }
#else
    for(long i=0; i<size; i++)
    {
        out_vec[i] = tanh_f16(in_vec[i]);
    }
#endif

    return 0;
}
