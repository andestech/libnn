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
static const float32_t tanh_f32_max = TANH_F32_MAX;
static const float32_t tanh_f32_min = TANH_F32_MIN;
static const float32_t tanh_thr     = TANH_F32_THR;
static const float32_t cst_1        = CST_1;
static const float32_t cst_2        = CST_2;
static const float32_t cst_1_3      = CST_1_3;



extern float32_t exp_f32(float32_t x);
extern float32_t tanh_f32(float32_t x)
{
    float32_t ret;
    x = (x > tanh_f32_max)? tanh_f32_max : x;
    x = (x < tanh_f32_min)? tanh_f32_min : x;

    // x * (1 - x^2/3) if |x| < 5.e-3 or (exp2x - 1) / (exp2x + 1) otherwise
    if(fabs(x) < tanh_thr)
        ret = x * (1 - x*x*cst_1_3);
    else
        ret = (exp_f32(cst_2 * x) - cst_1) / (exp_f32(cst_2 * x) + cst_1);

    return ret;
}

int32_t riscv_nn_tanh_f32(const float32_t * in_vec, uint32_t size, float32_t * out_vec)
{

    long i;
    for(i=0; i<size; i++)
    {
        out_vec[i] = tanh_f32(in_vec[i]);
    }
    return 0;
}
