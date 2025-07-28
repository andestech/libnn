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

//// Basic Operation Functions
int32_t riscv_nn_broadcast_mul_asym_s16(const int16_t * in_tensor,
                                        const int16_t * in_alpha,    // vector length should equal to in_tensor_ch
                                        int16_t * out_tensor,
                                        const uint32_t in_tensor_dim_x,
                                        const uint32_t in_tensor_dim_y,
                                        const uint32_t in_tensor_ch,
                                        const int32_t out_scale,
                                        const int32_t out_shift,  //TFL rule
                                        const int32_t in_offset,    //no used.
                                        const int32_t alpha_offset, //no used.
                                        const int32_t out_offset,   //no used.
                                        const int16_t act_min,
                                        const int16_t act_max)
{
    (void)in_offset;
	(void)alpha_offset;
	(void)out_offset;
    for (int i_out_y = 0; i_out_y < in_tensor_dim_y; ++i_out_y)
    {
        for (int i_out_x = 0; i_out_x < in_tensor_dim_x; ++i_out_x)
        {
            int base_idx = (i_out_y * in_tensor_dim_x + i_out_x) * in_tensor_ch;

            for (int i_out_ch = 0; i_out_ch < in_tensor_ch; ++i_out_ch)
            {
                int idx = base_idx + i_out_ch;

                const int32_t val = in_tensor[idx];
                const int32_t a = in_alpha[i_out_ch];

                int32_t unclamped_output = riscv_nn_requantize(val*a, out_scale, out_shift);

                const int32_t clamped_output = MIN(act_max, MAX(act_min, unclamped_output));
                out_tensor[idx] = (int16_t) clamped_output;
            }
        }
    }

    return 0;
}
