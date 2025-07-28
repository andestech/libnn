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
#include "riscv_nn_convolution.h"

//// Convolution Functions

int32_t riscv_nn_conv_1xn_HWC_s8_s8_s4_asym_bias_any(const int8_t * in_tensor,
                                                     const int32_t in_tensor_dim_x,
                                                     const int32_t in_tensor_ch,
                                                     const int32_t in_tensor_batch,
                                                     const int8_t * ker_weight,
                                                     const int32_t out_tensor_ch,
                                                     const int32_t ker_dim_x,
                                                     const int32_t pad_x,
                                                     const int32_t stride_x,
                                                     const int32_t * bias,
                                                     int8_t * out_tensor,
                                                     const int32_t * out_shift,
                                                     const int32_t * out_scale,
                                                     const int32_t out_offset,   //value is in the range of [-127, 128]
                                                     const int32_t in_offset,    //value is in the range of [-128, 127]
                                                     const int32_t act_min,
                                                     const int32_t act_max,
                                                     const int32_t out_tensor_dim_x,
                                                     const int32_t dilation_x,
                                                     int8_t * in_tmp_buf)
{
    int status = 0;
    status = riscv_nn_conv_HWC_s8_s8_s4_asym_bias_any(in_tensor,
                in_tensor_dim_x,
                1,
                in_tensor_ch,
                in_tensor_batch,
                ker_weight,
                out_tensor_ch,
                ker_dim_x,
                1,
                pad_x,
                0,
                stride_x,
                1,
                bias,
                out_tensor,
                out_shift,
                out_scale,
                out_offset,
                in_offset,
                act_min,
                act_max,
                out_tensor_dim_x,
                1,
                dilation_x,
                1,
                in_tmp_buf);

    return status;
}

int32_t riscv_nn_conv_1xn_HWC_s8_s8_s4_asym_bias_any_get_buffer_size(const int32_t in_tensor_dim_x,
                                                                     const int32_t in_tensor_ch,
                                                                     const int32_t in_tensor_batch,
                                                                     const int32_t out_tensor_ch,
                                                                     const int32_t ker_dim_x,
                                                                     const int32_t pad_x,
                                                                     const int32_t stride_x,
                                                                     const int32_t out_tensor_dim_x,
                                                                     const int32_t dilation_x)
{

    (void)in_tensor_dim_x;
    (void)in_tensor_ch;
    (void)in_tensor_batch;
    (void)out_tensor_ch;
    (void)ker_dim_x;
    (void)pad_x;
    (void)stride_x;
    (void)out_tensor_dim_x;
    (void)dilation_x;
    return 0;
}
