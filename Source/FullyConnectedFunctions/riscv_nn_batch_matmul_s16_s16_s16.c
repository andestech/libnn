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

int32_t riscv_nn_batch_matmul_s16_s16_s16(const int16_t * in_lhs,
                                          const int16_t * in_rhs,
                                          const int16_t lhs_offset,
                                          const int16_t rhs_offset,
                                          const int64_t * bias,
                                          int16_t * dst,
                                          const int16_t out_offset,
                                          const int32_t out_scale,
                                          const int32_t out_shift,
                                          const int32_t lhs_dim_n,
                                          const int32_t lhs_dim_h,
                                          const int32_t lhs_dim_w,
                                          const int32_t rhs_dim_n,
                                          const int32_t rhs_dim_h,
                                          const int32_t rhs_dim_w,
                                          const int32_t rhs_dim_c,
                                          const int32_t dst_dim_n,
                                          const int32_t dst_dim_h,
                                          const int32_t act_min,
                                          const int32_t act_max)
{
    const int32_t output_batch = dst_dim_n;
    const int32_t output_height = dst_dim_h;
    const int32_t lhs_rows = lhs_dim_w;
    const int32_t rhs_rows = rhs_dim_w;
    const int32_t rhs_cols = rhs_dim_c;

    const int32_t inner_lhs_diff = lhs_dim_h >= rhs_dim_h ? 0 : lhs_rows * rhs_cols;
    const int32_t inner_rhs_diff = rhs_dim_h >= lhs_dim_h ? rhs_rows * rhs_cols : 0;
    const int32_t outer_lhs_diff = lhs_dim_n >= rhs_dim_n
        ? inner_lhs_diff
        : -((lhs_rows * rhs_cols) - inner_lhs_diff) * lhs_dim_h;
    const int32_t outer_rhs_diff = rhs_dim_n >= lhs_dim_n ? (rhs_rows * rhs_cols) - inner_rhs_diff
                                                                          : -inner_rhs_diff * rhs_dim_h;

    const int32_t reduced_multiplier = REDUCE_MULTIPLIER(out_scale);

    for (int i_out_batch = 0; i_out_batch < output_batch; i_out_batch++)
    {
        for (int i_out_height = 0; i_out_height < output_height; i_out_height++)
        {
            for (int j = 0; j < lhs_rows; j++)
            {
                nn_vec_mat_mult_t_s16_s16_s16(in_lhs,
                                              in_rhs,
                                              lhs_offset,
                                              rhs_offset,
                                              bias,
                                              dst,
                                              out_offset,
                                              reduced_multiplier,
                                              out_shift,
                                              rhs_cols,
                                              rhs_rows,
                                              act_min,
                                              act_max);
                in_lhs += rhs_cols;
                dst += rhs_rows;
            }
            in_lhs -= inner_lhs_diff;
            in_rhs += inner_rhs_diff;
        }
        in_lhs += outer_lhs_diff;
        in_rhs += outer_rhs_diff;
    }

    return 0;
}
