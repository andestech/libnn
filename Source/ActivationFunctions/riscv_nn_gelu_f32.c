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
#include "riscv_nn_activation.h"

//// Relu Functions

extern float32_t tanh_f32(float32_t x);

void riscv_nn_gelu_f32(const float32_t * in_vec, uint32_t size, float32_t * out_vec)
{
    // approximation algorithm
    // 0.5 * x * ( 1 + tanh( sqrt( 2 / pi ) * ( x + 0.044715 * x^3 ) ) )

    const float32_t sqrt2dPi  = SQRT_2_D_PI;    // sqrt( 2 / pi )
    const float32_t gelu_coe0 = GELU_COE0;
    const float32_t gelu_coe1 = GELU_COE1;

    for (uint32_t i=0; i < size; i++)
    {
        out_vec[i] = gelu_coe0 * in_vec[i] * (1 + tanh_f32(sqrt2dPi * (in_vec[i] + gelu_coe1 * in_vec[i] * in_vec[i] * in_vec[i])));
    }
}
