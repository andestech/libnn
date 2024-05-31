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

#ifndef __RISCV_NN_ACTIVATION_H__
#define __RISCV_NN_ACTIVATION_H__

#ifdef __cplusplus
extern    "C"
{
#endif

#include "riscv_math_types.h"

/**
 * @brief This is the struct to select an activation function.
 *
 */
typedef enum
{
    NN_SIGMOID = 0,    /**< Use sigmoid activation function */
    NN_TANH = 1,       /**< Use tanh activation function */
} riscv_nn_activation_fun;

/**
 * @defgroup Activation Activation Functions
 * @brief Activation functions are used to introduce nonlinearity to the neural
 *        network by filtering the input data. These include Gaussian Error
 *        Linear Unit (GELU), Hyperbolic tangent (Tanh), Rectified Linear Unit
 *        (ReLU), and Sigmoid functions.
 *
 * @{
 */

/**
 * @brief           This function performs activation on signed 8-bit integer
 *                  input vectors using either the Sigmoid or Tanh function,
 *                  along with in-place and look-up table (LUT) algorithms.
 * @param[in,out]   in_out      Pointer to the input/output vector
 * @param[in]       size        Number of elements in the input/output vector
 * @param[in]       int_bits    Number of the bits in the integer part should
 *                              be less than 4
 * @param[in]       act_fun     Selection of activation functions. See Note
 *                              below for details.
 * @return          None
 *
 * @note
 * The available activation functions for selection include:
 *  - NN_SIGMOID: Use the Sigmoid activation function
 *  - NN_TANH: Use the Tanh activation function
 *
 * @b Example:
 * @code
 * #define SIZE 32
 * q7_t in_out[SIZE] = {...};
 * riscv_nn_activate_s8(in_out, SIZE, 0, NN_SIGMOID);
 * @endcode
 */
void riscv_nn_activate_s8(q7_t * in_out,
                        uint32_t size,
                        uint16_t int_bits,
                        riscv_nn_activation_fun act_fun);

/**
 * @brief           This function performs activation on signed 8-bit integer
 *                  input vectors using either the Sigmoid or Tanh function,
 *                  along with out-of-place and LUT algorithms.
 * @param[in]       in_vec      Pointer to the input vector
 * @param[in]       size        Number of elements in the input/output vector
 * @param[in]       int_bits    Number of the bits in the integer part should
 *                              be less than 4
 * @param[in]       act_fun     Selection of activation functions. See Note
 *                              below for details.
 * @param[out]      out_vec     Pointer to the output vector
 * @return          None
 *
 * @note
 * The available activation functions for selection include:
 *  - NN_SIGMOID: Use the Sigmoid activation function
 *  - NN_TANH: Use the Tanh activation function
 *
 * @b Example:
 * @code
 * #define SIZE 32
 * q7_t in_vec[SIZE] = {...};
 * q7_t out_vec[SIZE];
 * riscv_nn_activate_s8_2buf(in_vec, SIZE, 0, NN_SIGMOID, out_vec);
 * @endcode
 */
void riscv_nn_activate_s8_2buf(q7_t * in_vec,
                               uint32_t size,
                               uint16_t int_bits,
                               riscv_nn_activation_fun act_fun,
                               q7_t * out_vec);

/**
 * @brief           This function performs activation on signed 16-bit integer
 *                  input vectors using either the Sigmoid or Tanh function,
 *                  along with in-place and LUT algorithms.
 * @param[in,out]   in_out      Pointer to the input/output vector
 * @param[in]       size        Number of elements in the input/output vector
 * @param[in]       int_bits    Number of the bits in the integer part should
 *                              be less than 4
 * @param[in]       act_fun     Selection of activation functions. See Note
 *                              below for details.
 * @return          None
 *
 * @note
 * The availbale activation functions for selection include:
 *  - NN_SIGMOID: Use the Sigmoid activation function
 *  - NN_TANH: Use the Tanh activation function
 */
void riscv_nn_activate_s16(q15_t * in_out,
                           uint32_t size,
                           uint16_t int_bits,
                           riscv_nn_activation_fun act_fun);

/**
 * @brief           This function performs activation on signed 16-bit integer
 *                  input vectors using either the Sigmoid or Tanh function,
 *                  along with out-of-place and LUT algorithms.
 * @param[in]       in_vec      Pointer to the input vector
 * @param[in]       size        Number of elements in the input/output vector
 * @param[in]       int_bits    Number of the bits in the integer part should
 *                              be less than 4
 * @param[in]       act_fun     Selection of activation functions. See Note
 *                              below for details.
 * @param[out]      out_vec     Pointer to the output vector
 * @return          None
 *
 * @note
 * The availbale activation functions for selection include:
 *  - NN_SIGMOID: Use the Sigmoid activation function
 *  - NN_TANH: Use the Tanh activation function
 */
void riscv_nn_activate_s16_2buf(q15_t * in_vec,
                                uint32_t size,
                                uint16_t int_bits,
                                riscv_nn_activation_fun act_fun,
                                q15_t * out_vec);
/**
 * @brief           This function performs activation on signed 16-bit integer
 *                  input vectors using either the Sigmoid or Tanh function,
 *                  along with a high-precision algorithm.
 * @param[in]       in_vec      Pointer to the input vector
 * @param[out]      out_vec     Pointer to the output vector
 * @param[in]       size        Number of elements in the input/output vector
 * @param[in]       left_shift  The scaling for the inputs. See Note below for
 *                              details.
 * @param[in]       act_fun     Selection of activation functions. See Note
 *                              below for details.
 * @return          None
 *
 * @note
 * - Let INT_BITS represent the number of the bits in the integer part of the
 *   inputs. Set left_shift to INT_BITS -3 if INT_BITS is 3 or greater.
 *   Otherwise, set left_shift to 0.
 * - The inputs should be scaled down by the following if INT_BITS is less than
 *   3:
 *   1/(2^(3 - INT_BITS))
 * - The availbale activation functions for selection include:
 *   - NN_SIGMOID: Use the Sigmoid activation function
 *   - NN_TANH: Use the tanh Activation function
 */
void riscv_nn_activate_s16_hp(const q15_t * in_vec,
                              q15_t * out_vec,
                              const uint32_t size,
                              const uint32_t left_shift,
                              const riscv_nn_activation_fun act_fun);

#ifdef __riscv_zfh
/**
 * @brief           This function performs activation on half-precision
 *                  floating-point input vectors using the GELU function.
 * @param[in]       in_vec      Pointer to the input vector
 * @param[in]       size        Number of elements in the input/output vector
 * @param[out]      out_vec     Pointer to the output vector
 * @return          None
 */
void riscv_nn_gelu_f16(const float16_t * in_vec,
                       uint32_t size,
                       float16_t * out_vec);
#endif

/**
 * @brief           This function performs activation on single-precision
 *                  floating-point input vectors using the GELU function.
 * @param[in]       in_vec      Pointer to the input vector
 * @param[in]       size        Number of elements in the input/output vector
 * @param[out]      out_vec     Pointer to the output vector
 * @return          None
 */
void riscv_nn_gelu_f32(const float32_t * in_vec,
                       uint32_t size,
                       float32_t * out_vec);

/**
 * @brief           This function performs activation on signed 8-bit integer
 *                  input vectors using the Leaky ReLU function, along with an
 *                  in-place algorithm.
 * @param[in,out]   in_out      Pointer to the input/output vector
 * @param[in]       size        Number of elements in the input/output vector
 * @param[in]       slope       Slope value to be multiplied with the negative
 *                              inputs. The result will be right shifted 15 bits
 *                              to scale back to signed 8-bit integer.
 * @return          None
 *
 * @b Example:
 * @code
 * #define SIZE 1024
 * q15_t slope = 16384;
 * q7_t in_out[SIZE] = {...};
 * riscv_nn_leaky_relu_s8(in_out, SIZE, slope);
 * @endcode
 */
void riscv_nn_leaky_relu_s8(q7_t * in_out,
                            uint32_t size,
                            q15_t slope);

/**
 * @brief           This function performs activation on signed 8-bit integer
 *                  input vectors using the Leaky ReLU function, along with an
 *                  out-of-place algorithm.
 * @param[in]       in_vec      Pointer to the input vector
 * @param[in]       size        Number of elements in the input/output vector
 * @param[in]       slope       Slope value to be multiplied with the negative
 *                              inputs. The result will be right shifted 15 bits
 *                              to scale back to signed 8-bit integer.
 * @param[out]      out_vec     Pointer to the output vector
 * @return          None
 *
 * @b Example:
 * @code
 * #define SIZE 1024
 * q15_t slope = 16384;
 * q7_t in_vec[SIZE] = {...};
 * q7_t out_vec[SIZE];
 * riscv_nn_leaky_relu_s8_2buf(in_vec, SIZE, slope, out_vec);
 * @endcode
 */
void riscv_nn_leaky_relu_s8_2buf(q7_t * in_vec,
                                 uint32_t size,
                                 q15_t slope,
                                 q7_t * out_vec);

/**
 * @brief           This function performs activation on signed 8-bit integer
 *                  input vectors using the Leaky ReLU function and applies
 *                  asymmetric quantization to the outputs.
 * @param[in]       in_vec          Pointer to the input vector
 * @param[out]      out_vec         Pointer to the output vector
 * @param[in]       size            Number of elements in the input/output
 *                                  vector
 * @param[in]       multi_identity  Scaling value for the quantization on the
 *                                  nonnegative inputs
 * @param[in]       shift_identity  Shift amount for the quantization on the
 *                                  nonnegative inputs
 * @param[in]       multi_alpha     Scaling value for the quantization on the
 *                                  negative inputs
 * @param[in]       shift_alpha     Shift amount for the quantization on the
 *                                  negative inputs
 * @param[in]       in_offset       Offset value for the input vector. It should
 *                                  be in the range of -128 to 127.
 * @param[in]       out_offset      Offset value for the outputs. It should be
 *                                  in the range of -128 to 127.
 * @param[in]       act_min         Minimum value that the outputs are limited
 *                                  to. It should be in the range of -128 to 127.
 * @param[in]       act_max         Maximum value that the outputs are limited
 *                                  to. It should be in the range of -128 to 127.
 * @return          None
 *
 * @note
 *  During the quantization process, a positive shift_identity/shift_alpha value
 *  is used to left shift calculation results whereas a negative one is used to
 *  right shift.
 */
void riscv_nn_leaky_relu_s8_asym(int8_t * in_vec,
                                 int8_t * out_vec,
                                 const uint32_t size,
                                 const int32_t multi_identity,
                                 const int32_t shift_identity,
                                 const int32_t multi_alpha,
                                 const int32_t shift_alpha,
                                 const int32_t in_offset,
                                 const int32_t out_offset,
                                 const int8_t act_min,
                                 const int8_t act_max);

#ifdef __riscv_zfh
/**
 * @brief           This function performs activation on half-precision
 *                  floating-point input vectors using the Leaky ReLU function.
 * @param[in]       in_vec      Pointer to the input vector
 * @param[in]       size        Number of elements in the input/output vector
 * @param[in]       slope       Slope value to be multiplied with the negative
 *                              inputs.
 * @param[out]      out_vec     Pointer to the output vector
 * @return          None
 */
void riscv_nn_leaky_relu_f16(const float16_t* in_vec,
                        uint32_t size,
                        float16_t slope,
                        float16_t* out_vec);
#endif

/**
 * @brief           This function performs activation on signed 8-bit integer
 *                  input tensors using the PReLU function and applies
 *                  asymmetric quantization to the outputs.
 * @param[in]       in_vec              Pointer to the input tensor
 * @param[out]      out_vec             Pointer to the output tensor
 * @param[in]       alpha_data          Pointer to the scaling vector for the
 *                                      negative inputs
 * @param[in]       in_tensor_dim_x     X dimension of the input tensor
 * @param[in]       in_tensor_dim_y     Y dimension of the input tensor
 * @param[in]       in_tensor_ch        Number of input tensor channels
 * @param[in]       multi_identity      Scaling value for the quantization on
 *                                      the nonnegative inputs
 * @param[in]       shift_identity      Shift amount for the quantization on the
 *                                      nonnegative inputs
 * @param[in]       multi_alpha         Scaling value for the quantization on
 *                                      the negative inputs
 * @param[in]       shift_alpha         Shift amount for the quantization on the
 *                                      negative inputs
 * @param[in]       in_offset           Offset value for the input tensor. It
 *                                      should be in the range of -128 to 127.
 * @param[in]       alpha_offset        Offset value for the scaling vector. It
 *                                      should be in the range of -128 to 127.
 * @param[in]       out_offset          Offset value for the outputs. It should
 *                                      be in the range of -128 to 127.
 * @param[in]       act_min             Minimum value that the outputs are
 *                                      limited to. It should be in the range of
 *                                      -128 to 127.
 * @param[in]       act_max             Maximum value that the outputs are
 *                                      limited to. It should be in the range of
 *                                      -128 to 127.
 * @return          None
 *
 * @note
 *  During the quantization process, a positive shift_identity/shift_alpha value
 *  is used to left shift calculation results whereas a negative one is used to
 *  right shift.
 */
void riscv_nn_prelu_s8_asym(q7_t * in_vec,
                          q7_t * out_vec,
                          const q7_t * alpha_data,
                          const uint16_t in_tensor_dim_x,
                          const uint16_t in_tensor_dim_y,
                          const uint16_t in_tensor_ch,
                          const int32_t multi_identity,
                          const int32_t shift_identity,
                          const int32_t multi_alpha,
                          const int32_t shift_alpha,
                          const int32_t in_offset,
                          const int32_t alpha_offset,
                          const int32_t out_offset,
                          const q7_t act_min,
                          const q7_t act_max);

/**
 * @brief           This function performs activation on signed 8-bit integer
 *                  input vectors using the ReLU function, along with an
 *                  in-place algorithm. The maximum output from the ReLU
 *                  function is user-specified, for example:
 *                  f(x) = min(max(0,x),max_val)
 * @param[in,out]   in_vec      Pointer to the input/output vector
 * @param[in]       size        Number of elements in the input/output vector
 * @param[in]       max_val     Maximum value to limit the output vector
 * @return          None
 */
void riscv_nn_relu_any_s8(q7_t * in_vec,
                          uint32_t size,
                          q7_t max_val);

/**
 * @brief           This function performs activation on signed 8-bit integer
 *                  input vectors using the ReLU function, along with an
 *                  out-of-place algorithm. The maximum output from the ReLU
 *                  function is user-specified, for example:
 *                  f(x) = min(max(0,x),max_val)
 * @param[in]       in_vec      Pointer to the input vector
 * @param[in]       size        Number of elements in the input/output vector
 * @param[in]       max_val     Maximum value to limit the output vector
 * @param[out]      out_vec     Pointer to the output vector
 * @return          None
 */
void riscv_nn_relu_any_s8_2buf(q7_t * in_vec,
                               uint32_t size,
                               q7_t max_val,
                               q7_t * out_vec);

#ifdef __riscv_zfh
/**
 * @brief           This function performs activation on half-precision
 *                  floating-point input vectors using the ReLU function. The
 *                  maximum output from the ReLU function is user-specified, for
 *                  example:
 *                  f(x) = min(max(0,x),max_val)
 * @param[in]       in_vec      Pointer to the input vector
 * @param[in]       size        Number of elements in the input/output vector
 * @param[in]       max_val     Maximum value to limit the output vector
 * @param[out]      out_vec     Pointer to the output vector
 * @return          None
 */
void riscv_nn_relu_any_f16(const float16_t * in_vec,
                        uint32_t size,
                        float16_t max_val,
                        float16_t * out_vec);
#endif

/**
 * @brief           This function performs activation on signed 8-bit integer
 *                  input vectors using the ReLU function, along with an
 *                  in-place algorithm.
 * @param[in,out]   in_out      Pointer to the input/output vector
 * @param[in]       size        Number of elements in the input/output vector
 * @return          None
 *
 * @b Example:
 * @code
 * #define H 16
 * #define W 16
 * #define CH 5
 * #define NUM (H * W * CH)
 * q7_t in_out[NUM] = {...};
 * riscv_nn_relu_s8(in_out, NUM);
 * @endcode
 */
void riscv_nn_relu_s8(q7_t * in_out,
                      uint32_t size);

/**
 * @brief           This function performs activation on signed 8-bit integer
 *                  input vectors using the ReLU function, along with an
 *                  out-of-place algorithm.
 * @param[in]       in_vec      Pointer to the input vector
 * @param[in]       size        Number of elements in the input/output vector
 * @param[out]      out_vec     Pointer to the output vector
 * @return          None
 *
 * @b Example:
 * @code
 * #define H 16
 * #define W 16
 * #define CH 5
 * #define NUM (H * W * CH)
 * q7_t in_vec[NUM] = {...};
 * q7_t out_vec;
 * riscv_nn_relu_s8_2buf(in_vec, NUM, out_vec);
 * @endcode
 */
void riscv_nn_relu_s8_2buf(q7_t * in_vec,
                           uint32_t size,
                           q7_t * out_vec);

/**
 * @brief           This function performs activation on signed 16-bit integer
 *                  input vectors using the ReLU function, along with an
 *                  in-place algorithm.
 * @param[in,out]   in_out      Pointer to the input/output vector
 * @param[in]       size        Number of elements in the input/output vector
 * @return          None
 */
void riscv_nn_relu_s16(q15_t * in_out,
                       uint32_t size);

/**
 * @brief           This function performs activation on signed 16-bit integer
 *                  input vectors using the ReLU function, along with an
 *                  out-of-place algorithm.
 * @param[in]       in_vec      Pointer to the input vector
 * @param[in]       size        Number of elements in the input/output vector
 * @param[out]      out_vec     Pointer to the output vector
 * @return          None
 */
void riscv_nn_relu_s16_2buf(q15_t * in_vec,
                            uint32_t size,
                            q15_t * out_vec);

#ifdef __riscv_zfh
/**
 * @brief           This function performs activation on half-precision
 *                  floating-point input vectors using the ReLU function.
 * @param[in]       in_vec      Pointer to the input vector
 * @param[in]       size        Number of elements in the input/output vector
 * @param[out]      out_vec     Pointer to the output vector
 * @return          None
 */
void riscv_nn_relu_f16(const float16_t * in_vec,
                        uint32_t size,
                        float16_t * out_vec);
#endif

/**
 * @brief           This function performs activation on signed 8-bit integer
 *                  input vectors using the Sigmoid function.
 * @param[in]       in_offset       Offset value for the input vector. It should
 *                                  be in the range of -127 to 128.
 * @param[in]       in_range_radius The maximum or minimum value for the inputs.
 *                                  If the input is less than or equal to
 *                                  in_range_radius, the output will be limited
 *                                  to -128. Conversely, if the input is greater
 *                                  than in_range_radius, the output will be
 *                                  limited to 127.
 * @param[in]       in_mult         Scaling value for rescaling the inputs
 * @param[in]       in_lshift       Shift amount for resacling the inputs
 * @param[in]       size            Number of elements in the input/output
 *                                  vector
 * @param[in]       in_vec          Pointer to the input vector
 * @param[out]      out_vec         Pointer to the output vector
 * @return          None
 */
void riscv_nn_sigmoid_s8(const int32_t in_offset,
                         const int32_t in_range_radius,
                         const int16_t in_mult,
                         const int16_t in_lshift,
                         const uint32_t size,
                         const int8_t* in_vec,
                         int8_t* out_vec);

#ifdef __riscv_zfh
/**
 * @brief           This function performs activation on half-precision
 *                  floating-point input vectors using the Sigmoid function.
 * @param[in]       in_vec      Pointer to the input vector
 * @param[in]       size        Number of elements in the input/output vector
 * @param[out]      out_vec     Pointer to the output vector
 * @return          This function returns 0.
 *
 * @note
 * The inputs are restricted to the range [-10, 10].
 */
int32_t riscv_nn_sigmoid_f16(const float16_t * in_vec,
                            uint32_t size,
                            float16_t * out_vec);
#endif

/**
 * @brief           This function performs activation on signed 8-bit integer
 *                  input vectors using the Tanh function.
 * @param[in]       in_offset       Offset value for the input vector. It should
 *                                  be in the range of -127 to 128.
 * @param[in]       in_range_radius The maximum or minimum value for the inputs.
 *                                  If the input is less than or equal to
 *                                  in_range_radius, the output will be limited
 *                                  to -128. Conversely, if the input is greater
 *                                  than in_range_radius, the output will be
 *                                  limited to 127.
 * @param[in]       in_mult         Scaling value for rescaling the inputs
 * @param[in]       in_lshift       Shift amount for resacling the inputs
 * @param[in]       size            Number of elements in the input/output
 *                                  vector
 * @param[in]       in_vec          Pointer to the input vector
 * @param[out]      out_vec         Pointer to the output vector
 * @return          None
 */
void riscv_nn_tanh_s8(const int32_t in_offset,
                      const int32_t in_range_radius,
                      const int16_t in_mult,
                      const int16_t in_lshift,
                      const uint32_t size,
                      const int8_t* in_vec,
                      int8_t* out_vec);

#ifdef __riscv_zfh
/**
 * @brief           This function performs activation on half-precision
 *                  floating-point input vectors using the Tanh function.
 * @param[in]       in_vec      Pointer to the input vector
 * @param[in]       size        Number of elements in the input/output vector
 * @param[out]      out_vec     Pointer to the output vector
 * @return          This function returns 0.
 *
 * @note
 * The inputs are restricted to the range [-10, 10].
 */
int32_t riscv_nn_tanh_f16(const float16_t * in_vec,
                        uint32_t size,
                        float16_t * out_vec);
#endif

/**
 * @brief           This function performs activation on single-precision
 *                  floating-point input vectors using the Tanh function.
 * @param[in]       in_vec      Pointer to the input vector
 * @param[in]       size        Number of elements in the input/output vector
 * @param[out]      out_vec     Pointer to the output vector
 * @return          This function returns 0.
 */
int32_t riscv_nn_tanh_f32(const float32_t * in_vec,
                        uint32_t size,
                        float32_t * out_vec);


/**
 *   * @}
 */


#ifdef __cplusplus
}
#endif

#endif
