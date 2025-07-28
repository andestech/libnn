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

int32_t riscv_nn_rms_norm_f16(const float16_t * in_tensor,
                              const float16_t epsilon,
                              const float16_t * gamma,
                              const uint32_t sentence_len,
                              const uint32_t feature_len,
                              float16_t * out_tensor)
{
    int i, j;
    for (j = 0; j < sentence_len; j++)
    {
        float16_t var = 0.f, sigma = 0.f, tmp;
        for (i = 0; i < feature_len; i++)
        {
            tmp = in_tensor[j * feature_len + i];
            var += tmp * tmp;
        }

        var /= feature_len;
        sigma = sqrtf(var + epsilon);

        for (i = 0; i < feature_len; i++)
        {
            tmp = in_tensor[j * feature_len + i];
            tmp = (gamma[j] * tmp);
            tmp = (tmp / sigma);
            out_tensor[j * feature_len + i] = (float16_t)(tmp);
        }
    }

    return 0;
}
