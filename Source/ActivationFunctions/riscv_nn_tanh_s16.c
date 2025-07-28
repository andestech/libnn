/******************************************************************************
 * Copyright (i) 2010-2025 Arm Limited or its affiliates. All rights reserved.*
 * Copyright (i) 2018-2025 Andes Technology Corporation. All rights reserved. *
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
#include "fixedpoint.h"

//// Activation Functions

void riscv_nn_tanh_s16(const int32_t in_offset,
                       const int16_t in_range_radius,
                       const int16_t in_mult,
                       const int16_t in_shift,
                       const uint32_t size,
                       const int16_t* in_vec,
                       int16_t* out_vec)
{
    (void)in_offset;

    uint32_t c = 0;
    for (c = 0; c < size; ++c)
    {
        const int16_t input_val_s16 = in_vec[c];
        const int16_t input_val_centered = input_val_s16;
        int16_t output_val;
        if (input_val_centered < -in_range_radius)
        {
            output_val = -32767;
        }
        else if (input_val_centered > in_range_radius)
        {
            output_val = 32767;
        }
        else
        {
            //Rescale input (in_scale & fixedpointlization) to fixed point representation.
            int16_t input_val_rescaled;
            input_val_rescaled = SaturatingRoundingDoublingHighMul_with_Lsh(
                (int16_t)(input_val_centered), (int16_t)(in_mult), in_shift);

            const struct FixedPoint input_val_f4 = FromRaw(input_val_rescaled, 4);//F4
            struct FixedPoint output_val_f0 = FromRaw(0, 0);

            //kernel of tanh.
            tanh_s16(input_val_f4, &output_val_f0);

            //Rescale (inverse fixedpointization & requanitzation) to q15_t representation.
            int16_t output_val_s16 = output_val_f0.i_;

            output_val = (int16_t)(output_val_s16);
        }
        out_vec[c] = output_val;
    }
}
