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

//// Relu Functions

void riscv_nn_relu_any_f16(const float16_t * in_vec, uint32_t size, float16_t max_val, float16_t * out_vec)
{
    uint32_t  i;
    float16_t zero = (float16_t)0.0f;
    for (i = 0; i < size; i++)
    {
        if (in_vec[i] < zero) {
            out_vec[i] = zero;
        }
        else if (in_vec[i] > max_val) {
            out_vec[i] = max_val;
        }
        else {
            out_vec[i] = in_vec[i];
        }
    }
}
