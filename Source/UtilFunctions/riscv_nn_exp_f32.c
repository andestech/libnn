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

typedef union
{
    float32_t f32;
    int32_t   i32;
} _union32_t;

//--- const values for taylor_poly_f32 ---
// Exponent polynomial coefficients
static const float32_t exp_coe0 = EXP_COE0;
static const float32_t exp_coe1 = EXP_COE1;
static const float32_t exp_coe2 = EXP_COE2;
static const float32_t exp_coe3 = EXP_COE3;
static const float32_t exp_coe4 = EXP_COE4;
static const float32_t exp_coe5 = EXP_COE5;
static const float32_t exp_coe6 = EXP_COE6;
static const float32_t exp_coe7 = EXP_COE7;

//--- const values for exp_f32 ---
static const float32_t ln2      = LN2;          // ln(2)
static const float32_t inv_ln2  = INV_LN2;      // 1/ln(2)
static const float32_t exp_max  = EXP_F32_MAX;
static const float32_t exp_min  = EXP_F32_MIN;

static inline float32_t taylor_poly_f32(float32_t x)
{
    float32_t A = exp_coe0 + exp_coe4 * x;
    float32_t B = exp_coe2 + exp_coe6 * x;
    float32_t C = exp_coe1 + exp_coe5 * x;
    float32_t D = exp_coe3 + exp_coe7 * x;
    float32_t x2 = x * x;
    float32_t x4 = x2 * x2;
    float32_t res = (A + B * x2) + (C + D * x2) * x4;
    return res;
}

float32_t exp_f32(float32_t x)
{
    // Clip the inputs
    x = (x > exp_min)? x : exp_min;
    x = (x < exp_max)? x : exp_max;

    // Perform range reduction [-log(2),log(2)]
    int m = x * inv_ln2;
    float32_t val = x - (float32_t)m * ln2;

    // Polynomial Approximation
    _union32_t poly;
    poly.f32 = taylor_poly_f32(val);

    // Reconstruct
    int m2 = m << 23;
    poly.i32 = poly.i32 + m2;

    return poly.f32;
}

int32_t riscv_nn_exp_f32(const float32_t * in_vec, uint32_t size, float32_t * out_vec)
{
    while(size-- > 0)
    {
        *out_vec = exp_f32(*in_vec);
        in_vec++;
        out_vec++;
    }

    return 0;
}
