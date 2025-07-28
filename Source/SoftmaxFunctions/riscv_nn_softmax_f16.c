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

int32_t riscv_nn_softmax_f16(const float16_t * in_vec, uint32_t size, float16_t * out_vec)
{
#if defined(ENA_KERNEL_FP32)
    const float16_t exp_max = EXP_F16_MAX;
    const float16_t exp_min = EXP_F16_MIN;
#endif

#if   defined(ENA_KERNEL_FP32)
    float16_t sum = 0;
    long i;

    float16_t max = in_vec[0];
    for(i = 1; i < size; i++)
    {
        max = (in_vec[i] > max)? in_vec[i] : max;
    }

    for(i = 0; i < size; i++)
    {
        float16_t tmp = in_vec[i] - max;
        tmp = (tmp < exp_min)? exp_min : tmp;
        tmp = (tmp > exp_max)? exp_max : tmp;
        out_vec[i] = exp_f32(tmp);
        sum += out_vec[i];
    }

    for(i = 0; i < size; i++)
    {
        out_vec[i] = out_vec[i] / sum;
    }
#else
    float16_t sum = 0;
    long i;

    float16_t max = in_vec[0];
    for(i = 1; i < size; i++)
    {
        max = (in_vec[i] > max)? in_vec[i] : max;
    }

    for(i = 0; i < size; i++)
    {
        float16_t tmp = in_vec[i] - max;
        out_vec[i] = exp_f16(tmp);
        sum += out_vec[i];
    }

    for(i = 0; i < size; i++)
    {
        out_vec[i] = out_vec[i] / sum;
    }
#endif

    return 0;
}
