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

int32_t riscv_nn_softmax2d_f16(const float16_t * in_buf,
                             uint32_t row,
                             uint32_t col,
                             float16_t * out_buf,
                             float16_t * tmp_buf)
{
#if defined(ENA_KERNEL_FP32)
    const float16_t exp_max  = EXP_F16_MAX;
    const float16_t exp_min  = EXP_F16_MIN;
#endif

#if   defined(ENA_KERNEL_FP32)
    for(long r=0; r < row; r++)
    {
        float16_t sum = 0;
        long c;

        float16_t max = in_buf[0];
        for(c = 1; c < col; c++)
        {
            max = (in_buf[c] > max)? in_buf[c] : max;
        }

        for(c = 0; c < col; c++)
        {
            float16_t tmp = in_buf[c] - max;
            tmp = (tmp < exp_min)? exp_min : tmp;
            tmp = (tmp > exp_max)? exp_max : tmp;
            out_buf[c] = exp_f32(tmp);
            sum += out_buf[c];
        }

        for(c = 0; c < col; c++)
        {
            out_buf[c] = out_buf[c] / sum;
        }

        in_buf += col;
        out_buf += col;
    }
#else
    for(long r=0; r < row; r++)
    {
        float16_t sum = 0;
        long c;

        float16_t max = in_buf[0];
        for(c = 1; c < col; c++)
        {
            max = (in_buf[c] > max)? in_buf[c] : max;
        }

        for(c = 0; c < col; c++)
        {
            out_buf[c] = exp_f16(in_buf[c] - max);
            sum += out_buf[c];
        }

        for(c = 0; c < col; c++)
        {
            out_buf[c] = out_buf[c] / sum;
        }

        in_buf += col;
        out_buf += col;
    }
#endif

    return 0;
}
