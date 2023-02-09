/******************************************************************************
 * Copyright (C) 2010-2018 Arm Limited or its affiliates. All rights reserved.*
 * Copyright (C) 2018-2022 Andes Technology Corporation. All rights reserved. *
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

//// Convolution Functions

u8_t *riscv_nn_mat_mul_kernel_q7_u16_u8_reordered_bias_2sft(const q7_t * src1,
                                                    const u16_t * src2,
                                                    const uint16_t out_tensor_ch,
                                                    const uint16_t col_src1,
                                                    const uint16_t pre_rshift,
                                                    const uint16_t out_scale,
                                                    const uint16_t post_rshift,
                                                    const q31_t * bias,
                                                    u8_t * out)
{
    /* To be completed */
    return NULL;
}
