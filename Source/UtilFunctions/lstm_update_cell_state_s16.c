#include "internal_nn_math.h"    
#include "stdio.h"

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