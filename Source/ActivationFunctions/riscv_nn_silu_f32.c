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

int32_t riscv_nn_silu_f32(const float32_t * in_vec, uint32_t size, float32_t * out_vec)
{
    //--- const values for sigmoid ---
    const float32_t cst_1 = (float32_t) 1.f;
	uint32_t i;
    float32_t x, y;
    for(i = 0 ; i < size ; i++)
    {
		float32_t sigmoid = 0;
		x = in_vec[i];
        float32_t num = exp_f32(x);
        float32_t den = num + cst_1;
        sigmoid = num / den;
		y = in_vec[i] * sigmoid; // silu(x) = x*sigmoid(x).
        out_vec[i] = y;
    }
    return 0;
}
