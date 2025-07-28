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
#include "riscv_nn_support.h"

//// Convolution Functions

int32_t riscv_nn_conv_1x1_HWC_s8_s8_s4_asym_bias_any(const int8_t * in_tensor,
                                                     const int32_t in_tensor_dim_x,
                                                     const int32_t in_tensor_dim_y,
                                                     const int32_t in_tensor_ch,
                                                     const int32_t in_tensor_batch,
                                                     const int8_t * ker_weight,
                                                     const int32_t out_tensor_ch,
                                                     const int32_t pad_x,
                                                     const int32_t pad_y,
                                                     const int32_t stride_x,
                                                     const int32_t stride_y,
                                                     const int32_t * bias,
                                                     int8_t * out_tensor,
                                                     const int32_t * out_shift,
                                                     const int32_t * out_scale,
                                                     const int32_t out_offset,   //value is in the range of [-128, 127]
                                                     const int32_t in_offset,    //value is in the range of [-127, 128]
                                                     const int32_t act_min,
                                                     const int32_t act_max,
                                                     const int32_t out_tensor_dim_x,
                                                     const int32_t out_tensor_dim_y,
                                                     int8_t * tmp_buf)
{
    if ((pad_x != 0) ||
        (pad_y != 0))
    {
        return -1;
    }

    (void)tmp_buf;

    if ((stride_x == 1) && (stride_y == 1))
    {
        const int32_t lhs_rows = in_tensor_dim_x * in_tensor_dim_y * in_tensor_batch;
        const int32_t rhs_rows = out_tensor_ch;
        const int32_t rhs_cols = in_tensor_ch;
        const int32_t lhs_cols_offset = rhs_cols;

        riscv_nn_mat_mult_nt_t_s4(in_tensor,
                                  ker_weight,
                                  bias,
                                  out_tensor,
                                  out_scale,
                                  out_shift,
                                  lhs_rows,
                                  rhs_rows,
                                  rhs_cols,
                                  in_offset,
                                  out_offset,
                                  act_min,
                                  act_max,
                                  lhs_cols_offset);
    }
    else
    {
        const int32_t lhs_rows = out_tensor_dim_x;
        const int32_t rhs_rows = out_tensor_ch;
        const int32_t rhs_cols = in_tensor_ch;
        const int32_t input_inc = in_tensor_dim_x * stride_y * rhs_cols;
        const int32_t output_inc = out_tensor_dim_x * rhs_rows;
        const int32_t lhs_cols_offset = rhs_cols * stride_x;

        for (int i_batch = 0; i_batch < in_tensor_batch; i_batch++)
        {
            const int8_t *in_tensor2 = in_tensor + (i_batch * rhs_cols * in_tensor_dim_x * in_tensor_dim_y);
            for (int i_output_h = 0; i_output_h < out_tensor_dim_y; i_output_h++)
            {
                riscv_nn_mat_mult_nt_t_s4(in_tensor2,
                                          ker_weight,
                                          bias,
                                          out_tensor,
                                          out_scale,
                                          out_shift,
                                          lhs_rows,
                                          rhs_rows,
                                          rhs_cols,
                                          in_offset,
                                          out_offset,
                                          act_min,
                                          act_max,
                                          lhs_cols_offset);

                in_tensor2 += input_inc;
                out_tensor += output_inc;
            }
        }
    }

    return 0;
}

int32_t riscv_nn_conv_1x1_HWC_s8_s8_s4_asym_bias_any_get_buffer_size(const int32_t in_tensor_ch,
                                                                     const int32_t out_tensor_ch)
{
    (void)in_tensor_ch;
    (void)out_tensor_ch;
    return 0;
}
