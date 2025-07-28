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

int32_t nn_vec_mat_mult_t_s16_s16_s16(const int16_t *lhs,
                                      const int16_t *rhs,
                                      const int16_t lhs_offset,
                                      const int16_t rhs_offset,
                                      const int64_t *bias,
                                      int16_t *dst,
                                      const int16_t dst_offset,
                                      const int32_t dst_multiplier,
                                      const int32_t dst_shift,
                                      const int32_t rhs_cols,
                                      const int32_t rhs_rows,
                                      const int32_t activation_min,
                                      const int32_t activation_max)
{
    (void)lhs_offset;
    (void)rhs_offset;
    (void)dst_offset;

    for (int i_row_loop_cnt = 0; i_row_loop_cnt < rhs_rows; i_row_loop_cnt++)
    {
        const int16_t *lhs_ptr = lhs;
        const int16_t *rhs_ptr_0 = &rhs[0];

        int64_t result = 0;

        for (int32_t rhs_cols_idx = 0; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
        {
            const int64_t rhs_value0 = *rhs_ptr_0;
            const int64_t lhs_value  = *lhs_ptr;

            result += lhs_value * rhs_value0;

            ++rhs_ptr_0;
            ++lhs_ptr;
        }

        if (bias)
        {
            result += *bias++;
        }

        // Quantize down
        result = riscv_nn_requantize_s64(result, dst_multiplier, dst_shift);

        // Clamp the result
        result = MAX(result, activation_min);
        result = MIN(result, activation_max);

        *dst++ = (int16_t)result;
        rhs += rhs_cols;
    }
    return 0;

}
