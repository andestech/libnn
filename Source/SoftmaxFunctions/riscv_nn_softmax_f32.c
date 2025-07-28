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

//// Softmax Functions

int32_t riscv_nn_softmax_f32(const float32_t * in_vec, uint32_t size, float32_t * out_vec)
{

    float32_t sum = 0;
    long i;

    float32_t max = in_vec[0];
    for(i = 1; i < size; i++)
    {
        max = (in_vec[i] > max)? in_vec[i] : max;
    }

    for(i = 0; i < size; i++)
    {
        out_vec[i] = exp_f32(in_vec[i] - max);
        sum += out_vec[i];
    }

    for(i = 0; i < size; i++)
    {
        out_vec[i] = out_vec[i] / sum;
    }

    return 0;
}
