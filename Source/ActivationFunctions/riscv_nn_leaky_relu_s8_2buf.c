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

void riscv_nn_leaky_relu_s8_2buf(q7_t * FUNC_RESTRICT in_vec,
                                 uint32_t size,
                                 q15_t slope,
                                 q7_t * FUNC_RESTRICT out_vec)
{
    uint32_t  i;
    for (i = 0; i < size; i++)
    {
        if (in_vec[i] < 0)
        {
            int32_t tmp = (((int32_t)in_vec[i] * slope) + NN_ROUND(15)) >> 15;
            out_vec[i] = NDS_ISA_SATS(tmp, 8);
        }
        else{
            out_vec[i] = in_vec[i];
        }
    }
}
