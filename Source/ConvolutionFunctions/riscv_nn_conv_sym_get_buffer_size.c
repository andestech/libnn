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

uint32_t riscv_nn_conv_sym_get_buffer_size(const uint16_t in_tensor_dim_x,
                                            const uint16_t in_tensor_dim_y,
                                            const uint16_t in_tensor_ch,
                                            const uint16_t out_tensor_ch,
                                            const uint16_t ker_dim_x,
                                            const uint16_t ker_dim_y,
                                            const uint16_t pad_x,
                                            const uint16_t pad_y,
                                            const uint16_t stride_x,
                                            const uint16_t stride_y,
                                            const uint16_t out_tensor_dim_x,
                                            const uint16_t out_tensor_dim_y)
{
    (void) in_tensor_dim_x;
    (void) in_tensor_dim_y;
    (void) pad_x;
    (void) pad_y;
    (void) stride_x;
    (void) stride_y;
    (void) out_tensor_dim_x;
    (void) out_tensor_dim_y;
    uint32_t buf_size = 0;
    return buf_size;
}