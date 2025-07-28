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

void lstm_update_output_s16_s8(const int n_batch,
                                      const int n_cell,
                                      int16_t *cell_state,
                                      const int32_t cell_state_scale,
                                      const int16_t *output_gate,
                                      const riscv_nn_scaling hidden_scaling,
                                      const int32_t hidden_offset,
                                      int8_t *output_state,
                                      int16_t *cell_gate_scratch)
{
    const int32_t size = n_batch * n_cell;

    int32_t tanh_input_left_shift = (15 + cell_state_scale) - 3;
    if (tanh_input_left_shift < 0)
    {
        tanh_input_left_shift = -tanh_input_left_shift;
        int32_t loop_count = size;
        int16_t* tmp_cell_state = cell_state;
        while (loop_count > 0)
        {
            *tmp_cell_state = ((*tmp_cell_state) >> tanh_input_left_shift);
            tmp_cell_state++;
            loop_count--;
        }
        tanh_input_left_shift = 0;
    }

    riscv_nn_activate_s16_hp(cell_state, cell_gate_scratch, size, tanh_input_left_shift, NN_TANH);

    riscv_nn_ew_mul_s16_s8_asym(output_gate,
                                cell_gate_scratch,
                                output_state,
                                hidden_offset,
                                hidden_scaling.multiplier,
                                hidden_scaling.shift,
                                size);
}
