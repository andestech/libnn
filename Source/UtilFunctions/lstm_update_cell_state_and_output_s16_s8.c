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
#include "riscv_nn_types.h"
#include "riscv_nn_activation.h"
#include "riscv_nn_basic.h"
#include "riscv_nn_support.h"
#include "internal_nn_table.h"

void lstm_update_cell_state_and_output_s16_s8(
    const int32_t cell_state_scale,
    int16_t *cell_state,
    riscv_nn_lstm_context *scratch_buffers,
    const riscv_nn_scaling hidden_scaling,
    const int32_t hidden_offset,
    int8_t *output_state,
    const int n_batch,
    const int n_cell,
    const int n_output,
    int8_t* output
)
{
    const int32_t size = n_batch * n_cell;
    const int32_t cell_scale = 30 + cell_state_scale;
    int32_t loop_count = size;

    int16_t* tmp_cell_state = cell_state;
    const int16_t *input_gate = scratch_buffers->input_gate;
    const int16_t *forget_gate = scratch_buffers->forget_gate;
    const int16_t *cell_gate = scratch_buffers->cell_gate;
    const int16_t *output_gate = scratch_buffers->output_gate;

    int32_t tanh_input_left_shift = (15 + cell_state_scale) - 3;
    int32_t tmp_shift = 0;

    if (tanh_input_left_shift < 0)
    {
        tmp_shift = -tanh_input_left_shift;
    }

    loop_count = size;
    while (loop_count > 0)
    {
        int32_t value = (*tmp_cell_state) * (*forget_gate);
        int32_t value_1 = (*input_gate) * (*cell_gate);

        value = riscv_nn_divide_by_power_of_two(value, 15);
        value_1 = riscv_nn_divide_by_power_of_two(value_1, cell_scale);
        *tmp_cell_state = riscv_nn_clip_any(value + value_1, ((int16_t)(0x8000)), ((int16_t)(0x7FFF)));
        *tmp_cell_state = *tmp_cell_state >> tmp_shift;

        tmp_cell_state++;
        forget_gate++;
        input_gate++;
        cell_gate++;

        loop_count--;
    }
    if (tanh_input_left_shift < 0)
    {
        tanh_input_left_shift = 0;
    }

    riscv_nn_activate_s16_hp(cell_state, scratch_buffers->input_gate, size, tanh_input_left_shift, NN_TANH);

    riscv_nn_ew_mul_s16_s8_asym(output_gate,
                                scratch_buffers->input_gate,
                                output_state,
                                hidden_offset,
                                hidden_scaling.multiplier,
                                hidden_scaling.shift,
                                size);

    riscv_nn_dup_s8(output_state, output, n_batch * n_output * sizeof(int8_t));
}
