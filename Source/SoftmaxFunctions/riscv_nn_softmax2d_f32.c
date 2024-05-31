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

//// Softmax Functions

int32_t riscv_nn_softmax2d_f32(const float32_t * in_buf,
                             uint32_t row,
                             uint32_t col,
                             float32_t * out_buf,
                             float32_t * tmp_buf)
{

    for(long r=0; r < row; r++)
    {
        float32_t sum = 0;
        long c;

        float32_t max = in_buf[0];
        for(c = 1; c < col; c++)
        {
            max = (in_buf[c] > max)? in_buf[c] : max;
        }

        for(c = 0; c < col; c++)
        {
            out_buf[c] = exp_f32(in_buf[c] - max);
            sum += out_buf[c];
        }

        for(c = 0; c < col; c++)
        {
            out_buf[c] = out_buf[c] / sum;
        }

        in_buf += col;
        out_buf += col;
    }

    return 0;
}
