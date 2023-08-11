/******************************************************************************
 * Copyright (C) 2010-2018 Arm Limited or its affiliates. All rights reserved.*
 * Copyright (C) 2018-2023 Andes Technology Corporation. All rights reserved. *
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
#include "riscv_nn_support.h"

//// FullyConnected Functions
void riscv_nn_fc_mat_vec_s8_wt_converter(const q7_t *wt_mat,
                                        const uint32_t size,
                                        const uint32_t wt_row_num,
                                        q7_t *wt_mat_out)
{
    long row, col;
    int8_t *ptr1 = (int8_t*)wt_mat,
           *ptr2 = ptr1 + size,
           *ptr3 = ptr2 + size,
           *ptr4 = ptr3 + size,
           *out = wt_mat_out;

    row = wt_row_num >> 2;
    while(row-- > 0)
    {
        col = size >> 1;
        while(col-- > 0)
        {
            *out++ = *ptr1++;
            *out++ = *ptr2++;
            *out++ = *ptr1++;
            *out++ = *ptr2++;

            *out++ = *ptr3++;
            *out++ = *ptr4++;
            *out++ = *ptr3++;
            *out++ = *ptr4++;
        }

        if(size & 0x1)
        {
           *out++ = *ptr1++;
           *out++ = *ptr2++;
           *out++ = *ptr3++;
           *out++ = *ptr4++;
        }

        ptr1 += size * 3;
        ptr2 += size * 3;
        ptr3 += size * 3;
        ptr4 += size * 3;
    }

    row = wt_row_num & 3;
    long reset_length = row * size;
    memcpy(out, ptr1, reset_length * sizeof(*wt_mat));
}
