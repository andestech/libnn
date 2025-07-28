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

//// Relu Functions
void riscv_nn_prelu_s8_asym(int8_t * in_tensor,
                          int8_t * out_tensor,
                          const int8_t * alpha_data,
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
                          const int8_t act_min,
                          const int8_t act_max)
{
    for (int32_t i_out_y = 0; i_out_y < in_tensor_dim_y; ++i_out_y)
    {
        for (int32_t i_out_x = 0; i_out_x < in_tensor_dim_x; ++i_out_x)
        {
            int32_t base_idx = (i_out_y * in_tensor_dim_x + i_out_x) * in_tensor_ch;
            int32_t i_out_ch = 0;
            uint32_t loop = 0;

            loop = in_tensor_ch;
            while (loop > 0)
            {
                int32_t idx = base_idx + i_out_ch;
                const int32_t val = in_tensor[idx] - in_offset;
                int32_t unclamped_output;

                if (val >= 0)
                {
                    unclamped_output = out_offset + riscv_nn_requantize(val,
                                                        multi_identity,
                                                        shift_identity);
                }
                else
                {
                    const int32_t a = alpha_data[i_out_ch] - alpha_offset;
                    unclamped_output = out_offset + riscv_nn_requantize(val*a,
                                                        multi_alpha,
                                                        shift_alpha);
                }

                const int32_t clamped_output = MIN(act_max, MAX(act_min, unclamped_output));
                out_tensor[idx] = (int8_t)clamped_output;

                loop--;
                i_out_ch++;
            }
        }
    }
}
