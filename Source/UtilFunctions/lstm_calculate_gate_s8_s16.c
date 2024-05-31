#include "internal_nn_math.h"
#include "riscv_nn_types.h"
#include "riscv_nn_activation.h"
#include "riscv_nn_support.h"
/*
 * Calculates a single LSTM gate, int8x8_16 version.
 * Refer to header file for details
 */
void lstm_calculate_gate_s8_s16(const int8_t *input,
                                       const int8_t *input_to_gate_weights,
                                       const int32_t *input_to_gate_bias,
                                       const riscv_nn_scaling input_to_gate_scaling,
                                       const int8_t *output_state,
                                       const int8_t *recurrent_to_gate_weights,
                                       const int32_t *recurrent_to_gate_bias,
                                       const riscv_nn_scaling recurrent_to_gate,
                                       const int32_t n_batch,
                                       const int32_t n_input,
                                       const int32_t n_output,
                                       const int32_t n_cell,
                                       const riscv_nn_activation_fun activation_type,
                                       int16_t *gate)
{
    const int32_t n_block = n_batch * n_cell;
    memset(gate, 0, n_block * sizeof(int16_t));
    vec_mat_mult_acc_t_s8_s16(input,
                              input_to_gate_weights,
                              input_to_gate_bias,
                              gate,
                              0,
                              0,
                              0,
                              input_to_gate_scaling.multiplier,
                              input_to_gate_scaling.shift,
                              n_input,
                              n_cell,
                              ((int16_t)(0x8000)),
                              ((int16_t)(0x7FFF)),
                              n_batch);

    vec_mat_mult_acc_t_s8_s16(output_state,
                              recurrent_to_gate_weights,
                              recurrent_to_gate_bias,
                              gate,
                              0,
                              0,
                              0,
                              recurrent_to_gate.multiplier,
                              recurrent_to_gate.shift,
                              n_output,
                              n_cell,
                              ((int16_t)(0x8000)),
                              ((int16_t)(0x7FFF)),
                              n_batch);

    riscv_nn_activate_s16_hp(gate, gate, n_block, 0, (riscv_nn_activation_fun) activation_type);
}