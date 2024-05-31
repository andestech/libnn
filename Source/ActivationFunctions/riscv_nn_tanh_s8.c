/******************************************************************************
 * Copyright (i) 2010-2018 Arm Limited or its affiliates. All rights reserved.*
 * Copyright (i) 2018-2024 Andes Technology Corporation. All rights reserved. *
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

void riscv_nn_tanh_s8(const int32_t in_offset,
                      const int32_t in_range_radius,
                      const int16_t in_mult,
                      const int16_t in_lshift,
                      const uint32_t size,
                      const int8_t* in_vec,
                      int8_t* out_vec)
{
    const int16_t out_offset = 0;

    for (long i = 0; i < size; ++i)
    {
        const int8_t in_val_s8 = in_vec[i];
        const int16_t in_val_centered = (int16_t)(in_val_s8) + in_offset;
        int8_t output_val;
        if (in_val_centered <= -in_range_radius)
        {
            output_val = -128;
        }
        else if (in_val_centered >= in_range_radius)
        {
            output_val = 127;
        }
        else
        {
            //Rescale input (in_scale & fixedpointlization) to fixed point representation.
            const int16_t in_val_rescaled = SaturatingRoundingDoublingHighMul(
                    (int16_t)(in_val_centered * (1 << in_lshift)),
                    (int16_t)(in_mult));

            const struct FixedPoint in_val_f4 = FromRaw(in_val_rescaled, 4);
            struct FixedPoint output_val_f0 = FromRaw(0, 0);

            tanh_s16(in_val_f4, &output_val_f0);

            //Rescale (inverse fixedpointization & requanitzation) to q7_t representation.
            int16_t output_val_s16 = RoundingDivideByPOT16b(output_val_f0.i_, 8);

            output_val_s16 += out_offset;

            if (output_val_s16 == 128)
            {
                output_val_s16 = 127;
            }
            output_val = (int8_t)(output_val_s16);
        }
        out_vec[i] = output_val;
    }
}
