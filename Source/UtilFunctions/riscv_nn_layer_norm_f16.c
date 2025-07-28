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

int riscv_nn_layer_norm_f16(const float16_t *in_tensor,
                            const float16_t epsilon,
                            const float16_t *beta,
                            const float16_t *gamma,
                            const uint32_t sentence_len,
                            const uint32_t feature_len,
                            float16_t *out_tensor)
{
    int i, j;

    for (j = 0; j < sentence_len; j++)
    {
        float32_t mean = 0.f, var = 0.f, sigma = 0.f, tmp;

        for (i = 0; i < feature_len; i++)
        {
            mean += (float32_t)in_tensor[j * feature_len + i];
        }
        mean /= feature_len;

        for (i = 0; i < feature_len; i++)
        {
            tmp = (float32_t)in_tensor[j * feature_len + i] - mean;
            var += tmp * tmp;
        }
        var /= feature_len;
        sigma = sqrtf(var + (float32_t)epsilon);

        for (i = 0; i < feature_len; i++)
        {
            tmp = ((float32_t)in_tensor[j * feature_len + i] - mean);
            tmp = ((float32_t)gamma[i] * tmp);
            tmp = (tmp / sigma);
            out_tensor[j * feature_len + i] = tmp + (float32_t)beta[i];
        }
    }

    return 0;
}
