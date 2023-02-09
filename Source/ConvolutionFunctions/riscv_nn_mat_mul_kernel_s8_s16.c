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

q7_t *riscv_nn_mat_mult_kernel_s8_s16(const q7_t *input_a,
                                    const q15_t *input_b,
                                    const uint16_t output_ch,
                                    const int32_t *out_shift,
                                    const int32_t *out_mult,
                                    const int32_t out_offset,
                                    const int16_t activation_min,
                                    const int16_t activation_max,
                                    const uint16_t num_col_a,
                                    const int32_t *const output_bias,
                                    q7_t *out_0)
{
// #ifdef ENA_VEC_ISA
//     /* set up the second output pointers */
//     q7_t *out_1 = out_0 + output_ch;
//     const int32_t *bias = output_bias;
//     uint16_t row_count = output_ch;
//     const q7_t *ip_a0 = input_a;
//     int32_t vl;

//     //set SEW for 1st time VMV_S_X
//     //LMUL=2 is just for clear v6~v9 (vmv ignores LMUL)
//     NDS_VEC_VSETVLI_E32_M2(vl, num_col_a);

//     while(row_count-- > 0)
//     {
//         /* setup pointers for B */
//         const q15_t *ip_b0 = input_b;
//         const q15_t *ip_b1 = ip_b0 + num_col_a;

//         /* load the bias */
//         q31_t ch_0_out_0 = *bias;
//         q31_t ch_0_out_1 = *bias++;

//         // init
//         NDS_VEC_VMV_S_X(NDS_VEC_V0, ch_0_out_0);
//         NDS_VEC_VMV_S_X(NDS_VEC_V1, ch_0_out_1);
//         //clear v6~v9 (keep multiplication results)
//         NDS_VEC_VAND_VI(NDS_VEC_V6, NDS_VEC_V6, 0x0);
//         NDS_VEC_VAND_VI(NDS_VEC_V8, NDS_VEC_V8, 0x0);

//         uint16_t col_count = num_col_a;
//         while(col_count)
//         {
//             NDS_VEC_VSETVLI_E16(vl, col_count);
//             NDS_VEC_VLB_V(NDS_VEC_V2, ip_a0);
//             NDS_VEC_VLH_V(NDS_VEC_V3, ip_b0);
//             NDS_VEC_VLH_V(NDS_VEC_V4, ip_b1);
//             NDS_VEC_VWMACC_VV(NDS_VEC_V6, NDS_VEC_V2, NDS_VEC_V3);
//             NDS_VEC_VWMACC_VV(NDS_VEC_V8, NDS_VEC_V2, NDS_VEC_V4);
//             col_count -= vl;
//             ip_a0 += vl;
//             ip_b0 += vl;
//             ip_b1 += vl;
//         }

//         NDS_VEC_VSETVLI_E32_M2(vl, num_col_a);   //note the avl here
//         NDS_VEC_VREDSUM_VS(NDS_VEC_V0, NDS_VEC_V6, NDS_VEC_V0);
//         NDS_VEC_VREDSUM_VS(NDS_VEC_V1, NDS_VEC_V8, NDS_VEC_V1);
//         //move vr results to gpr
//         NDS_VEC_VMV_X_S(ch_0_out_0, NDS_VEC_V0);
//         NDS_VEC_VMV_X_S(ch_0_out_1, NDS_VEC_V1);
//         // printf("out=%08x\n", ch_0_out_0);
//         // printf("out=%08x\n", ch_0_out_1);
// //-----------------------------------------------
//         // uint16_t col_count = num_col_a;
//         // while (col_count)
//         // {
//         //     q7_t a0 = *ip_a0++;
//         //     q15_t b0 = *ip_b0++;
//         //     q15_t b1 = *ip_b1++;

//         //     ch_0_out_0 += a0 * b0;
//         //     ch_0_out_1 += a0 * b1;
//         //     col_count--;
//         // }

//         ch_0_out_0 = riscv_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
//         ch_0_out_0 += out_offset;
//         ch_0_out_0 = MAX(ch_0_out_0, activation_min);
//         ch_0_out_0 = MIN(ch_0_out_0, activation_max);
//         *out_0++ = (q7_t)ch_0_out_0;

//         ch_0_out_1 = riscv_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
//         ch_0_out_1 += out_offset;
//         ch_0_out_1 = MAX(ch_0_out_1, activation_min);
//         ch_0_out_1 = MIN(ch_0_out_1, activation_max);
//         *out_1++ = (q7_t)ch_0_out_1;
//         out_mult++;
//         out_shift++;
//     }

//     out_0 += output_ch;

//     /* return the new output pointer with offset */
//     return out_0;
// #elif defined(ENA_DSP_ISA_64)
    (void)input_a;
    (void)input_b;
    (void)output_ch;
    (void)out_shift;
    (void)out_mult;
    (void)out_offset;
    (void)activation_min;
    (void)activation_max;
    (void)num_col_a;
    (void)output_bias;
    (void)out_0;
    /* To be completed */
    return NULL;
}
