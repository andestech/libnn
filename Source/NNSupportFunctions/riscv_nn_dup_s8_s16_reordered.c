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

void riscv_nn_dup_s8_s16_reordered(const q7_t * src, q15_t * dst, uint32_t size)
{
    const q7_t *pIn = src;     /* Src pointer */
    uint32_t  blkCnt;           /* loop counter */

// #ifdef ENA_VEC_ISA

    blkCnt = size >> 2;
    while(blkCnt > 0)
    {
        q15_t in0 = (q15_t) *pIn++;
        q15_t in1 = (q15_t) *pIn++;
        q15_t in2 = (q15_t) *pIn++;
        q15_t in3 = (q15_t) *pIn++;

        *dst++ = in0;
        *dst++ = in2;
        *dst++ = in1;
        *dst++ = in3;

        blkCnt--;
    }

    blkCnt = size & 0x3u;

    while (blkCnt > 0u)
    {
        /* C = (q15_t) A << 8 */
        /* convert from q7 to q15 and then store the results in the destination buffer */
        *dst++ = (q15_t) * pIn++;

        /* Decrement the loop counter */
        blkCnt--;
    }

}
