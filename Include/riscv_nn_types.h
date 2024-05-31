#include "riscv_math_types.h"


#ifndef __RISCV_NN_TYPES_H__
#define __RISCV_NN_TYPES_H__

/** RISCV-NN object for the quantized Relu activation */
typedef struct
{
    int32_t min; /**< Min value used to clamp the result */
    int32_t max; /**< Max value used to clamp the result */
} riscv_nn_activation;

/** RISCV-NN object to contain LSTM specific input parameters related to dimensions */
typedef struct
{
    int32_t max_time;
    int32_t num_inputs;
    int32_t num_batches;
    int32_t num_outputs;
} riscv_nn_lstm_dims;

/** LSTM guard parameters */
typedef struct
{
    int32_t input_variance;
    int32_t forget_variance;
    int32_t cell_variance;
    int32_t output_variance;
} riscv_nn_lstm_guard_params;

/** LSTM scratch buffer container */
typedef struct
{
    int16_t *input_gate;
    int16_t *forget_gate;
    int16_t *cell_gate;
    int16_t *output_gate;
} riscv_nn_lstm_context;

/** Quantized clip value for cell and projection of LSTM input. Zero value means no clipping. */
typedef struct
{
    int16_t cell;
    int8_t projection;
} riscv_nn_lstm_clip_params;

/** RISCV-NN object for quantization parameters */
typedef struct
{
    int32_t multiplier; /**< Multiplier value */
    int32_t shift;      /**< Shift value */
} riscv_nn_scaling;

/** RISCV-NN norm layer coefficients */
typedef struct
{
    int16_t *input_weight;
    int16_t *forget_weight;
    int16_t *cell_weight;
    int16_t *output_weight;
} riscv_nn_layer_norm;

/** Parameters for integer LSTM, as defined in TFLM */
typedef struct
{
    int32_t time_major; /**< Nonzero (true) if first row of data is timestamps for input */
    riscv_nn_scaling input_to_input_scaling;
    riscv_nn_scaling input_to_forget_scaling;
    riscv_nn_scaling input_to_cell_scaling;
    riscv_nn_scaling input_to_output_scaling;
    riscv_nn_scaling recurrent_to_input_scaling;
    riscv_nn_scaling recurrent_to_forget_scaling;
    riscv_nn_scaling recurrent_to_cell_scaling;
    riscv_nn_scaling recurrent_to_output_scaling;
    riscv_nn_scaling cell_to_input_scaling;
    riscv_nn_scaling cell_to_forget_scaling;
    riscv_nn_scaling cell_to_output_scaling;
    riscv_nn_scaling projection_scaling;
    riscv_nn_scaling hidden_scaling;
    riscv_nn_scaling layer_norm_input_scaling;  /**< layer normalization for input layer */
    riscv_nn_scaling layer_norm_forget_scaling; /**< layer normalization for forget gate */
    riscv_nn_scaling layer_norm_cell_scaling;   /**< layer normalization for cell */
    riscv_nn_scaling layer_norm_output_scaling; /**< layer normalization for outpus layer */

    int32_t cell_state_shift;
    int32_t hidden_offset;
    int32_t output_state_offset;

    riscv_nn_lstm_clip_params clip;
    riscv_nn_lstm_guard_params guard;
    riscv_nn_layer_norm layer_norm;

    /* Effective bias is precalculated as bias + zero_point * weight.
    Only applicable to when input/output are s8 and weights are s16 */
    const int32_t *i2i_effective_bias; /**< input to input effective bias */
    const int32_t *i2f_effective_bias; /**< input to forget gate effective bias */
    const int32_t *i2c_effective_bias; /**< input to cell effective bias */
    const int32_t *i2o_effective_bias; /**< input to output effective bias */

    const int32_t *r2i_effective_bias; /**< recurrent gate to input effective bias */
    const int32_t *r2f_effective_bias; /**< recurrent gate to forget gate effective bias */
    const int32_t *r2c_effective_bias; /**< recurrent gate to cell effective bias */
    const int32_t *r2o_effective_bias; /**< recurrent gate to output effective bias */

    const int32_t *projection_effective_bias;

    /* Not precalculated bias */
    const int32_t *input_gate_bias;
    const int32_t *forget_gate_bias;
    const int32_t *cell_gate_bias;
    const int32_t *output_gate_bias;

    /* Activation min and max */
    riscv_nn_activation activation;

} riscv_nn_lstm_params;

#endif // RISCV_NN_TYPES_H