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

#include "riscv_nn_activation.h"
#include "internal_nn_table.h"
#include "internal_nn_math.h"

//// Activations Functions

void riscv_nn_activate_s16_hp(const q15_t * FUNC_RESTRICT in_vec,
                              q15_t * FUNC_RESTRICT out_vec,
                              const uint32_t size,
                              const uint32_t left_shift,
                              const riscv_nn_activation_fun act_fun)
{
    uint32_t abs_input_shift, max_saturation;
    switch (act_fun)
    {
        case NN_SIGMOID:
            abs_input_shift = 9;
            max_saturation = 0x7FFF << 10;
            break;
        case NN_TANH:
        default:
            abs_input_shift = 8;
            max_saturation = 0xFFFF << 8;
            break;
    }

    // Use the LUT for sigmoid and take into account, that
    // tanh(x) = 2*sigmoid(2*x) - 1
    int32_t in_scale = ((int32_t)3) << left_shift;

    for (int i = 0; i < size; ++i, in_vec++, out_vec++)
    {
        int32_t in_data = ((*in_vec) * in_scale);
        uint32_t abs_in_data = in_data > 0 ? in_data : -in_data;
        uint32_t uh = abs_in_data >> abs_input_shift;
        uint32_t result;

        if (uh >= 255)
        {
            result = max_saturation;
        }
        else
        {
            uint32_t ua = sigmoid_table_uint16[uh];
            uint32_t ub = sigmoid_table_uint16[uh + 1];
            uint32_t ut;
            if (act_fun == NN_SIGMOID)
            {
                ut = abs_in_data & 0x1ff;
            }
            else
            {
                ut = abs_in_data & 0x0ff;
            }
            result = (ua << abs_input_shift) + ut * (ub - ua);
        }
        if (act_fun == NN_SIGMOID)
        {
            result = (in_data >= 0) ? (result + (1 << 9)) : ((1 << 25) - result + (1 << 9) - 1);
            result >>= 10;
        }
        else
        {
            result = (in_data >= 0) ? (result - (1 << 23)) + (1 << 7) : ((-result + (1 << 23)) + (1 << 7) - 1);
            result >>= 8;
        }
        *out_vec = (int16_t)result;
    }
}
