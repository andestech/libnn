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

#include "internal_nn_math.h"

void lstm_update_cell_state_s16(const int32_t n_block,
                                       const int32_t cell_state_scale,
                                       int16_t *cell_state,
                                       const int16_t *input_gate,
                                       const int16_t *forget_gate,
                                       const int16_t *cell_gate)
{
    const int32_t cell_scale = 30 + cell_state_scale;
    int32_t loop_count = n_block;

    loop_count = n_block;
    while (loop_count > 0)
    {
        int32_t value = (*cell_state) * (*forget_gate);
        int32_t value_1 = (*input_gate) * (*cell_gate);

        value = riscv_nn_divide_by_power_of_two(value, 15);
        value_1 = riscv_nn_divide_by_power_of_two(value_1, cell_scale);

        *cell_state++ = riscv_nn_clip_any(value + value_1, ((int16_t)(0x8000)), ((int16_t)(0x7FFF)));

        forget_gate++;
        input_gate++;
        cell_gate++;

        loop_count--;
    }
}
