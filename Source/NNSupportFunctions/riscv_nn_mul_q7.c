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

//// NN Support Functions

void riscv_nn_mul_q7(q7_t * src1,
                q7_t * src2,
                q7_t * dst,
                const uint16_t out_rshift,
                uint32_t size)
{
    uint32_t blkCnt;                               /* loop counters */

    /* Initialize blkCnt with number of samples */
    blkCnt = size;


    while (blkCnt > 0U)
    {
        /* C = A * B */
        /* Multiply the inputs and store the result in the destination buffer */
        *dst++ = (q7_t) NDS_ISA_SATS((((q15_t) (*src1++) * (*src2++) + NN_ROUND(out_rshift)) >> out_rshift), 8);

        /* Decrement the size loop counter */
        blkCnt--;
    }
}
