/******************************************************************************
 * Copyright (C) 2010-2018 Arm Limited or its affiliates. All rights reserved.*
 * Copyright (C) 2018-2022 Andes Technology Corporation. All rights reserved. *
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

int32_t riscv_nn_top_k_s8(q7_t *data_in,
                        uint32_t size,
                        uint32_t k_val,
                        q7_t *out_val,
                        uint32_t *out_idx)
{
    q7_t max_val = 127, tmp_min = max_val, tmp_val = 0;
    uint32_t tmp_i=0, tmp_j=0, tmp_min_idx=0, tmp_idx=0, prev_idx=0, curr_idx=0;

    // choose the first k data and get the min_val/idx info
    for(tmp_i=0; tmp_i < k_val; tmp_i++)
    {
        out_val[tmp_i] = data_in[tmp_i];
        out_idx[tmp_i] = tmp_i;
        if(tmp_min > out_val[tmp_i])
        {
            tmp_min = out_val[tmp_i];
            tmp_min_idx = tmp_i;
        }
    }

    // checking the remainder data
    tmp_i = k_val;
    do
    {
        if(data_in[tmp_i] > tmp_min)
        {
            out_val[tmp_min_idx] = data_in[tmp_i];
            out_idx[tmp_min_idx] = tmp_i;

            tmp_min = max_val;
            prev_idx = 0;
            for(tmp_j = 0; tmp_j < k_val; tmp_j++)
            {
                if(tmp_min >= out_val[tmp_j])
                {
                    if(tmp_min == out_val[tmp_j])
                    {
                        curr_idx = out_idx[tmp_j];
                        if(prev_idx < curr_idx)
                        {
                            prev_idx = curr_idx;
                            tmp_min_idx = tmp_j;
                        }
                    }
                    else
                    {
                        tmp_min = out_val[tmp_j];
                        tmp_min_idx = tmp_j;
                        prev_idx = out_idx[tmp_j];
                    }
                }
            }
        }

        tmp_i++;

    } while(tmp_i < size);

    /* after scanning done, "out_val" will be the top-k of input,
       sorting the array*/
    for(tmp_i =0 ; tmp_i < k_val-1 ; tmp_i++)
    {
        for(tmp_j = 0 ; tmp_j < k_val-1 ; tmp_j++)
        {
            if(out_val[tmp_j] <= out_val[tmp_j + 1])
            {
                tmp_val = out_val[tmp_j];
                out_val[tmp_j] = out_val[tmp_j + 1];
                out_val[tmp_j + 1] = tmp_val;
                if(out_val[tmp_j] == out_val[tmp_j + 1])
                {
                    if(out_idx[tmp_j] > out_idx[tmp_j + 1])
                    {
                        tmp_idx = out_idx[tmp_j];
                        out_idx[tmp_j] = out_idx[tmp_j + 1];
                        out_idx[tmp_j + 1] = tmp_idx;
                    }
                }
                else
                {
                    tmp_idx = out_idx[tmp_j];
                    out_idx[tmp_j] = out_idx[tmp_j + 1];
                    out_idx[tmp_j + 1] = tmp_idx;
                }
            }
        }
    }

    return 0;
}
