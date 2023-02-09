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
#if (NDS_VEC_RVV_VERSION >= 100)
q7_t *riscv_nn_mat_mul_kernel_u8_q7_2sft_unroll4(const q7_t * src1,
                                            const u8_t * src2,
                                            const uint16_t out_tensor_ch,
                                            const uint16_t col_src1,
                                            const uint16_t pre_rshift,
                                            const uint16_t out_scale,
                                            const uint16_t post_rshift,
                                            q7_t * out)
{
#ifdef ENA_NDS_V5_VEC_DOT_PROD
    unsigned long vl;
    unsigned long rowCnt = out_tensor_ch >> 2;
    q7_t *pOut2 = out + out_tensor_ch;
    q7_t *pOut3 = out + out_tensor_ch * 2;
    q7_t *pOut4 = out + out_tensor_ch * 3;

    const q7_t *pA = src1;

#if (__clang__)
    register const long zero asm ("x0") = 0;
#else
    register const long zero asm ("x0");
#endif
    NDS_VEC_VSETVLI(vl, 1, NDS_VEC_VTYPE_SEW_E32, NDS_VEC_VTYPE_LMUL_M1);
    NDS_VEC_VAND_VI(NDS_VEC_V31, NDS_VEC_V31, 0x0);     //keep zero for redsum

    while(rowCnt > 0)
    {
        // setup pointers for A
        const q7_t *pA2 = pA + col_src1;
        const q7_t *pA3 = pA + col_src1 * 2;
        const q7_t *pA4 = pA + col_src1 * 3;
        // setup pointers for B
        const u8_t *pB = src2;
        const u8_t *pB2 = pB + col_src1;
        const u8_t *pB3 = pB + col_src1 * 2;
        const u8_t *pB4 = pB + col_src1 * 3;

        NDS_VEC_VSETVLI(vl, zero, NDS_VEC_VTYPE_SEW_E32, NDS_VEC_VTYPE_LMUL_M8);

        //init
        //clear v8~v23 (keep multiplication results)
        NDS_VEC_VAND_VI(NDS_VEC_V8, NDS_VEC_V8, 0x0);
        NDS_VEC_VAND_VI(NDS_VEC_V16, NDS_VEC_V16, 0x0);

        unsigned long colCnt = col_src1;
        while(colCnt > 0)
        {
            NDS_VEC_VSETVLI(vl, colCnt, NDS_VEC_VTYPE_SEW_E8, NDS_VEC_VTYPE_LMUL_M1);
            //load src1 and src2
            NDS_VEC_VLB_V(NDS_VEC_V0, pA);
            NDS_VEC_VLB_V(NDS_VEC_V1, pA2);
            NDS_VEC_VLB_V(NDS_VEC_V2, pA3);
            NDS_VEC_VLB_V(NDS_VEC_V3, pA4);

            NDS_VEC_VLB_V(NDS_VEC_V4, pB);
            NDS_VEC_VLB_V(NDS_VEC_V5, pB2);
            NDS_VEC_VLB_V(NDS_VEC_V6, pB3);
            NDS_VEC_VLB_V(NDS_VEC_V7, pB4);

            //bump pointers and update loop counter
            colCnt -= vl;
            pA += vl;
            pA2 += vl;
            pA3 += vl;
            pA4 += vl;
            pB += vl;
            pB2 += vl;
            pB3 += vl;
            pB4 += vl;

            //set the proper avl/SEW/LMUL values for vd4dot
            vl >>= 2;
            NDS_VEC_VSETVLI(vl, vl, NDS_VEC_VTYPE_SEW_E32, NDS_VEC_VTYPE_LMUL_M1);

            //acc += src1 * src2
            NDS_VEC_VD4DOTSU_VV(NDS_VEC_V8, NDS_VEC_V0, NDS_VEC_V4);
            NDS_VEC_VD4DOTSU_VV(NDS_VEC_V9, NDS_VEC_V1, NDS_VEC_V4);
            NDS_VEC_VD4DOTSU_VV(NDS_VEC_V10, NDS_VEC_V2, NDS_VEC_V4);
            NDS_VEC_VD4DOTSU_VV(NDS_VEC_V11, NDS_VEC_V3, NDS_VEC_V4);

            NDS_VEC_VD4DOTSU_VV(NDS_VEC_V12, NDS_VEC_V0, NDS_VEC_V5);
            NDS_VEC_VD4DOTSU_VV(NDS_VEC_V13, NDS_VEC_V1, NDS_VEC_V5);
            NDS_VEC_VD4DOTSU_VV(NDS_VEC_V14, NDS_VEC_V2, NDS_VEC_V5);
            NDS_VEC_VD4DOTSU_VV(NDS_VEC_V15, NDS_VEC_V3, NDS_VEC_V5);

            NDS_VEC_VD4DOTSU_VV(NDS_VEC_V16, NDS_VEC_V0, NDS_VEC_V6);
            NDS_VEC_VD4DOTSU_VV(NDS_VEC_V17, NDS_VEC_V1, NDS_VEC_V6);
            NDS_VEC_VD4DOTSU_VV(NDS_VEC_V18, NDS_VEC_V2, NDS_VEC_V6);
            NDS_VEC_VD4DOTSU_VV(NDS_VEC_V19, NDS_VEC_V3, NDS_VEC_V6);

            NDS_VEC_VD4DOTSU_VV(NDS_VEC_V20, NDS_VEC_V0, NDS_VEC_V7);
            NDS_VEC_VD4DOTSU_VV(NDS_VEC_V21, NDS_VEC_V1, NDS_VEC_V7);
            NDS_VEC_VD4DOTSU_VV(NDS_VEC_V22, NDS_VEC_V2, NDS_VEC_V7);
            NDS_VEC_VD4DOTSU_VV(NDS_VEC_V23, NDS_VEC_V3, NDS_VEC_V7);
        }
        pA = pA4;

        NDS_VEC_VSETVLI(vl, col_src1, NDS_VEC_VTYPE_SEW_E32, NDS_VEC_VTYPE_LMUL_M1);    //note the avl here
        NDS_VEC_VREDSUM_VS(NDS_VEC_V0, NDS_VEC_V8, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V1, NDS_VEC_V9, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V2, NDS_VEC_V10, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V3, NDS_VEC_V11, NDS_VEC_V31);

        NDS_VEC_VREDSUM_VS(NDS_VEC_V4, NDS_VEC_V12, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V5, NDS_VEC_V13, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V6, NDS_VEC_V14, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V7, NDS_VEC_V15, NDS_VEC_V31);

        NDS_VEC_VREDSUM_VS(NDS_VEC_V8, NDS_VEC_V16, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V9, NDS_VEC_V17, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V10, NDS_VEC_V18, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V11, NDS_VEC_V19, NDS_VEC_V31);

        NDS_VEC_VREDSUM_VS(NDS_VEC_V12, NDS_VEC_V20, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V13, NDS_VEC_V21, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V14, NDS_VEC_V22, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V15, NDS_VEC_V23, NDS_VEC_V31);

        NDS_VEC_VSETVLI_E32(vl, 4);     //4: 4 acc
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V0, NDS_VEC_V1, 1);
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V0, NDS_VEC_V2, 2);
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V0, NDS_VEC_V3, 3);

        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V4, NDS_VEC_V5, 1);
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V4, NDS_VEC_V6, 2);
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V4, NDS_VEC_V7, 3);

        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V8, NDS_VEC_V9, 1);
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V8, NDS_VEC_V10, 2);
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V8, NDS_VEC_V11, 3);

        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V12, NDS_VEC_V13, 1);
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V12, NDS_VEC_V14, 2);
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V12, NDS_VEC_V15, 3);

        //1st shift of 2-stage shift
        NDS_VEC_VSRA_VX(NDS_VEC_V0, NDS_VEC_V0, pre_rshift);
        NDS_VEC_VSRA_VX(NDS_VEC_V4, NDS_VEC_V4, pre_rshift);
        NDS_VEC_VSRA_VX(NDS_VEC_V8, NDS_VEC_V8, pre_rshift);
        NDS_VEC_VSRA_VX(NDS_VEC_V12, NDS_VEC_V12, pre_rshift);

        NDS_VEC_VMUL_VX(NDS_VEC_V0, NDS_VEC_V0, out_scale);
        NDS_VEC_VMUL_VX(NDS_VEC_V4, NDS_VEC_V4, out_scale);
        NDS_VEC_VMUL_VX(NDS_VEC_V8, NDS_VEC_V8, out_scale);
        NDS_VEC_VMUL_VX(NDS_VEC_V12, NDS_VEC_V12, out_scale);

        NDS_VEC_VADD_VX(NDS_VEC_V0, NDS_VEC_V0, NN_ROUND(post_rshift));
        NDS_VEC_VADD_VX(NDS_VEC_V4, NDS_VEC_V4, NN_ROUND(post_rshift));
        NDS_VEC_VADD_VX(NDS_VEC_V8, NDS_VEC_V8, NN_ROUND(post_rshift));
        NDS_VEC_VADD_VX(NDS_VEC_V12, NDS_VEC_V12, NN_ROUND(post_rshift));

        //saturate into 8-bits
        NDS_VEC_VSETVLI_E16(vl, 4);     //4: 4 acc
        NDS_VEC_VNSRA_WX(NDS_VEC_V0, NDS_VEC_V0, post_rshift);
        NDS_VEC_VNSRA_WX(NDS_VEC_V4, NDS_VEC_V4, post_rshift);
        NDS_VEC_VNSRA_WX(NDS_VEC_V8, NDS_VEC_V8, post_rshift);
        NDS_VEC_VNSRA_WX(NDS_VEC_V12, NDS_VEC_V12, post_rshift);
        NDS_VEC_VSETVLI_E8(vl, 4);     //4: 4 acc
        NDS_VEC_VNCLIP_WI(NDS_VEC_V0, NDS_VEC_V0, 0);
        NDS_VEC_VNCLIP_WI(NDS_VEC_V4, NDS_VEC_V4, 0);
        NDS_VEC_VNCLIP_WI(NDS_VEC_V8, NDS_VEC_V8, 0);
        NDS_VEC_VNCLIP_WI(NDS_VEC_V12, NDS_VEC_V12, 0);

        //store the results
        NDS_VEC_VSB_V(NDS_VEC_V0, out);
        NDS_VEC_VSB_V(NDS_VEC_V4, pOut2);
        NDS_VEC_VSB_V(NDS_VEC_V8, pOut3);
        NDS_VEC_VSB_V(NDS_VEC_V12, pOut4);
        out += vl;
        pOut2 += vl;
        pOut3 += vl;
        pOut4 += vl;

        rowCnt--;
    }

    //reset rows (as out_tensor_ch is 4n+2)
    if(out_tensor_ch & 0x2)
    {
        // setup pointers for A
        const q7_t *pA2 = pA + col_src1;
        // setup pointers for B
        const u8_t *pB = src2;
        const u8_t *pB2 = pB + col_src1;
        const u8_t *pB3 = pB + col_src1 * 2;
        const u8_t *pB4 = pB + col_src1 * 3;

        NDS_VEC_VSETVLI(vl, zero, NDS_VEC_VTYPE_SEW_E32, NDS_VEC_VTYPE_LMUL_M8);

        //init
        //clear v0~v15 (keep multiplication results)
        NDS_VEC_VAND_VI(NDS_VEC_V0, NDS_VEC_V0, 0x0);
        NDS_VEC_VAND_VI(NDS_VEC_V8, NDS_VEC_V8, 0x0);

        unsigned long colCnt = col_src1;
        while(colCnt > 0)
        {
            NDS_VEC_VSETVLI(vl, colCnt, NDS_VEC_VTYPE_SEW_E8, NDS_VEC_VTYPE_LMUL_M2);
            //load src1
            NDS_VEC_VLB_V(NDS_VEC_V16, pA);
            NDS_VEC_VLB_V(NDS_VEC_V18, pA2);
            //load src2
            NDS_VEC_VLB_V(NDS_VEC_V20, pB);
            NDS_VEC_VLB_V(NDS_VEC_V22, pB2);
            NDS_VEC_VLB_V(NDS_VEC_V24, pB3);
            NDS_VEC_VLB_V(NDS_VEC_V26, pB4);

            //bump pointers and update loop counter
            colCnt -= vl;
            pA += vl;
            pA2 += vl;
            pB += vl;
            pB2 += vl;
            pB3 += vl;
            pB4 += vl;

            //set the proper avl/SEW/LMUL values for vd4dot
            vl >>= 2;
            NDS_VEC_VSETVLI(vl, vl, NDS_VEC_VTYPE_SEW_E32, NDS_VEC_VTYPE_LMUL_M2);

            //acc += src1 * src2
            NDS_VEC_VD4DOTSU_VV(NDS_VEC_V0, NDS_VEC_V16, NDS_VEC_V20);
            NDS_VEC_VD4DOTSU_VV(NDS_VEC_V2, NDS_VEC_V18, NDS_VEC_V20);
            NDS_VEC_VD4DOTSU_VV(NDS_VEC_V4, NDS_VEC_V16, NDS_VEC_V22);
            NDS_VEC_VD4DOTSU_VV(NDS_VEC_V6, NDS_VEC_V18, NDS_VEC_V22);
            NDS_VEC_VD4DOTSU_VV(NDS_VEC_V8, NDS_VEC_V16, NDS_VEC_V24);
            NDS_VEC_VD4DOTSU_VV(NDS_VEC_V10, NDS_VEC_V18, NDS_VEC_V24);
            NDS_VEC_VD4DOTSU_VV(NDS_VEC_V12, NDS_VEC_V16, NDS_VEC_V26);
            NDS_VEC_VD4DOTSU_VV(NDS_VEC_V14, NDS_VEC_V18, NDS_VEC_V26);
        }

        NDS_VEC_VSETVLI(vl, col_src1, NDS_VEC_VTYPE_SEW_E32, NDS_VEC_VTYPE_LMUL_M2);    //note the avl here
        NDS_VEC_VREDSUM_VS(NDS_VEC_V16, NDS_VEC_V0, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V17, NDS_VEC_V2, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V18, NDS_VEC_V4, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V19, NDS_VEC_V6, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V20, NDS_VEC_V8, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V21, NDS_VEC_V10, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V22, NDS_VEC_V12, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V23, NDS_VEC_V14, NDS_VEC_V31);

        NDS_VEC_VSETVLI_E32(vl, 2);     //2: 2 acc
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V16, NDS_VEC_V17, 1);
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V18, NDS_VEC_V19, 1);
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V20, NDS_VEC_V21, 1);
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V22, NDS_VEC_V23, 1);

        //1st shift of 2-stage shift
        NDS_VEC_VSRA_VX(NDS_VEC_V16, NDS_VEC_V16, pre_rshift);
        NDS_VEC_VSRA_VX(NDS_VEC_V18, NDS_VEC_V18, pre_rshift);
        NDS_VEC_VSRA_VX(NDS_VEC_V20, NDS_VEC_V20, pre_rshift);
        NDS_VEC_VSRA_VX(NDS_VEC_V22, NDS_VEC_V22, pre_rshift);

        NDS_VEC_VMUL_VX(NDS_VEC_V16, NDS_VEC_V16, out_scale);
        NDS_VEC_VMUL_VX(NDS_VEC_V18, NDS_VEC_V18, out_scale);
        NDS_VEC_VMUL_VX(NDS_VEC_V20, NDS_VEC_V20, out_scale);
        NDS_VEC_VMUL_VX(NDS_VEC_V22, NDS_VEC_V22, out_scale);

        NDS_VEC_VADD_VX(NDS_VEC_V16, NDS_VEC_V16, NN_ROUND(post_rshift));
        NDS_VEC_VADD_VX(NDS_VEC_V18, NDS_VEC_V18, NN_ROUND(post_rshift));
        NDS_VEC_VADD_VX(NDS_VEC_V20, NDS_VEC_V20, NN_ROUND(post_rshift));
        NDS_VEC_VADD_VX(NDS_VEC_V22, NDS_VEC_V22, NN_ROUND(post_rshift));

        //saturate into 8-bits
        NDS_VEC_VSETVLI_E16(vl, 2);     //2: 2 acc
        NDS_VEC_VNSRA_WX(NDS_VEC_V16, NDS_VEC_V16, post_rshift);
        NDS_VEC_VNSRA_WX(NDS_VEC_V18, NDS_VEC_V18, post_rshift);
        NDS_VEC_VNSRA_WX(NDS_VEC_V20, NDS_VEC_V20, post_rshift);
        NDS_VEC_VNSRA_WX(NDS_VEC_V22, NDS_VEC_V22, post_rshift);
        NDS_VEC_VSETVLI_E8(vl, 2);     //2: 2 acc
        NDS_VEC_VNCLIP_WI(NDS_VEC_V16, NDS_VEC_V16, 0);
        NDS_VEC_VNCLIP_WI(NDS_VEC_V18, NDS_VEC_V18, 0);
        NDS_VEC_VNCLIP_WI(NDS_VEC_V20, NDS_VEC_V20, 0);
        NDS_VEC_VNCLIP_WI(NDS_VEC_V22, NDS_VEC_V22, 0);

        //store the results
        NDS_VEC_VSB_V(NDS_VEC_V16, out);
        NDS_VEC_VSB_V(NDS_VEC_V18, pOut2);
        NDS_VEC_VSB_V(NDS_VEC_V20, pOut3);
        NDS_VEC_VSB_V(NDS_VEC_V22, pOut4);
        out += vl;
        pOut2 += vl;
        pOut3 += vl;
        pOut4 += vl;
    }

    out += out_tensor_ch * 3;

    return out;
#else
    unsigned long vl;
    unsigned long rowCnt = out_tensor_ch >> 2;
    q7_t *pOut2 = out + out_tensor_ch;
    q7_t *pOut3 = out + out_tensor_ch * 2;
    q7_t *pOut4 = out + out_tensor_ch * 3;

    const q7_t *pA = src1;

    while(rowCnt > 0)
    {
        // setup pointers for A
        const q7_t *pA2 = pA + col_src1;
        const q7_t *pA3 = pA + col_src1 * 2;
        const q7_t *pA4 = pA + col_src1 * 3;
        // setup pointers for B
        const u8_t *pB = src2;
        const u8_t *pB2 = pB + col_src1;
        const u8_t *pB3 = pB + col_src1 * 2;
        const u8_t *pB4 = pB + col_src1 * 3;

#if (__clang__)
        register const long zero asm ("x0") = 0;
#else
        register const long zero asm ("x0");
#endif
        NDS_VEC_VSETVLI(vl, zero, NDS_VEC_VTYPE_SEW_E32, NDS_VEC_VTYPE_LMUL_M8);

        //init
        //clear v8~v23 (keep multiplication results)
        NDS_VEC_VAND_VI(NDS_VEC_V8, NDS_VEC_V8, 0x0);
        NDS_VEC_VAND_VI(NDS_VEC_V16, NDS_VEC_V16, 0x0);
        NDS_VEC_VSETVLI(vl, col_src1, NDS_VEC_VTYPE_SEW_E32, NDS_VEC_VTYPE_LMUL_M1);
        NDS_VEC_VAND_VI(NDS_VEC_V31, NDS_VEC_V31, 0x0);     //keep zero for redsum

        unsigned long colCnt = col_src1;
        while(colCnt > 0)
        {
            NDS_VEC_VSETVLI(vl, colCnt, NDS_VEC_VTYPE_SEW_E8, NDS_VEC_VTYPE_LMUL_MF4);
            //load src1 and src2
            NDS_VEC_VLB_V(NDS_VEC_V0, pA);
            NDS_VEC_VLB_V(NDS_VEC_V1, pA2);
            NDS_VEC_VLB_V(NDS_VEC_V2, pA3);
            NDS_VEC_VLB_V(NDS_VEC_V3, pA4);

            NDS_VEC_VLB_V(NDS_VEC_V4, pB);
            NDS_VEC_VLB_V(NDS_VEC_V5, pB2);
            NDS_VEC_VLB_V(NDS_VEC_V6, pB3);
            NDS_VEC_VLB_V(NDS_VEC_V7, pB4);

            //acc += src1 * src2
            NDS_VEC_VQMACCSU_VV(NDS_VEC_V8, NDS_VEC_V0, NDS_VEC_V4);
            NDS_VEC_VQMACCSU_VV(NDS_VEC_V9, NDS_VEC_V1, NDS_VEC_V4);
            NDS_VEC_VQMACCSU_VV(NDS_VEC_V10, NDS_VEC_V2, NDS_VEC_V4);
            NDS_VEC_VQMACCSU_VV(NDS_VEC_V11, NDS_VEC_V3, NDS_VEC_V4);

            NDS_VEC_VQMACCSU_VV(NDS_VEC_V12, NDS_VEC_V0, NDS_VEC_V5);
            NDS_VEC_VQMACCSU_VV(NDS_VEC_V13, NDS_VEC_V1, NDS_VEC_V5);
            NDS_VEC_VQMACCSU_VV(NDS_VEC_V14, NDS_VEC_V2, NDS_VEC_V5);
            NDS_VEC_VQMACCSU_VV(NDS_VEC_V15, NDS_VEC_V3, NDS_VEC_V5);

            NDS_VEC_VQMACCSU_VV(NDS_VEC_V16, NDS_VEC_V0, NDS_VEC_V6);
            NDS_VEC_VQMACCSU_VV(NDS_VEC_V17, NDS_VEC_V1, NDS_VEC_V6);
            NDS_VEC_VQMACCSU_VV(NDS_VEC_V18, NDS_VEC_V2, NDS_VEC_V6);
            NDS_VEC_VQMACCSU_VV(NDS_VEC_V19, NDS_VEC_V3, NDS_VEC_V6);

            NDS_VEC_VQMACCSU_VV(NDS_VEC_V20, NDS_VEC_V0, NDS_VEC_V7);
            NDS_VEC_VQMACCSU_VV(NDS_VEC_V21, NDS_VEC_V1, NDS_VEC_V7);
            NDS_VEC_VQMACCSU_VV(NDS_VEC_V22, NDS_VEC_V2, NDS_VEC_V7);
            NDS_VEC_VQMACCSU_VV(NDS_VEC_V23, NDS_VEC_V3, NDS_VEC_V7);

            //bump pointers and update loop counter
            colCnt -= vl;
            pA += vl;
            pA2 += vl;
            pA3 += vl;
            pA4 += vl;
            pB += vl;
            pB2 += vl;
            pB3 += vl;
            pB4 += vl;
        }
        pA = pA4;

        NDS_VEC_VSETVLI(vl, col_src1, NDS_VEC_VTYPE_SEW_E32, NDS_VEC_VTYPE_LMUL_M1);    //note the avl here
        NDS_VEC_VREDSUM_VS(NDS_VEC_V0, NDS_VEC_V8, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V1, NDS_VEC_V9, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V2, NDS_VEC_V10, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V3, NDS_VEC_V11, NDS_VEC_V31);

        NDS_VEC_VREDSUM_VS(NDS_VEC_V4, NDS_VEC_V12, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V5, NDS_VEC_V13, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V6, NDS_VEC_V14, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V7, NDS_VEC_V15, NDS_VEC_V31);

        NDS_VEC_VREDSUM_VS(NDS_VEC_V8, NDS_VEC_V16, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V9, NDS_VEC_V17, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V10, NDS_VEC_V18, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V11, NDS_VEC_V19, NDS_VEC_V31);

        NDS_VEC_VREDSUM_VS(NDS_VEC_V12, NDS_VEC_V20, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V13, NDS_VEC_V21, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V14, NDS_VEC_V22, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V15, NDS_VEC_V23, NDS_VEC_V31);

        NDS_VEC_VSETVLI_E32(vl, 4);     //4: 4 acc
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V0, NDS_VEC_V1, 1);
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V0, NDS_VEC_V2, 2);
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V0, NDS_VEC_V3, 3);

        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V4, NDS_VEC_V5, 1);
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V4, NDS_VEC_V6, 2);
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V4, NDS_VEC_V7, 3);

        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V8, NDS_VEC_V9, 1);
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V8, NDS_VEC_V10, 2);
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V8, NDS_VEC_V11, 3);

        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V12, NDS_VEC_V13, 1);
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V12, NDS_VEC_V14, 2);
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V12, NDS_VEC_V15, 3);

        //1st shift of 2-stage shift
        NDS_VEC_VSRA_VX(NDS_VEC_V0, NDS_VEC_V0, pre_rshift);
        NDS_VEC_VSRA_VX(NDS_VEC_V4, NDS_VEC_V4, pre_rshift);
        NDS_VEC_VSRA_VX(NDS_VEC_V8, NDS_VEC_V8, pre_rshift);
        NDS_VEC_VSRA_VX(NDS_VEC_V12, NDS_VEC_V12, pre_rshift);

        NDS_VEC_VMUL_VX(NDS_VEC_V0, NDS_VEC_V0, out_scale);
        NDS_VEC_VMUL_VX(NDS_VEC_V4, NDS_VEC_V4, out_scale);
        NDS_VEC_VMUL_VX(NDS_VEC_V8, NDS_VEC_V8, out_scale);
        NDS_VEC_VMUL_VX(NDS_VEC_V12, NDS_VEC_V12, out_scale);

        NDS_VEC_VADD_VX(NDS_VEC_V0, NDS_VEC_V0, NN_ROUND(post_rshift));
        NDS_VEC_VADD_VX(NDS_VEC_V4, NDS_VEC_V4, NN_ROUND(post_rshift));
        NDS_VEC_VADD_VX(NDS_VEC_V8, NDS_VEC_V8, NN_ROUND(post_rshift));
        NDS_VEC_VADD_VX(NDS_VEC_V12, NDS_VEC_V12, NN_ROUND(post_rshift));

        //saturate into 8-bits
        NDS_VEC_VSETVLI_E16(vl, 4);     //4: 4 acc
        NDS_VEC_VNSRA_WX(NDS_VEC_V0, NDS_VEC_V0, post_rshift);
        NDS_VEC_VNSRA_WX(NDS_VEC_V4, NDS_VEC_V4, post_rshift);
        NDS_VEC_VNSRA_WX(NDS_VEC_V8, NDS_VEC_V8, post_rshift);
        NDS_VEC_VNSRA_WX(NDS_VEC_V12, NDS_VEC_V12, post_rshift);
        NDS_VEC_VSETVLI_E8(vl, 4);     //4: 4 acc
        NDS_VEC_VNCLIP_WI(NDS_VEC_V0, NDS_VEC_V0, 0);
        NDS_VEC_VNCLIP_WI(NDS_VEC_V4, NDS_VEC_V4, 0);
        NDS_VEC_VNCLIP_WI(NDS_VEC_V8, NDS_VEC_V8, 0);
        NDS_VEC_VNCLIP_WI(NDS_VEC_V12, NDS_VEC_V12, 0);

        //store the results
        NDS_VEC_VSB_V(NDS_VEC_V0, out);
        NDS_VEC_VSB_V(NDS_VEC_V4, pOut2);
        NDS_VEC_VSB_V(NDS_VEC_V8, pOut3);
        NDS_VEC_VSB_V(NDS_VEC_V12, pOut4);
        out += vl;
        pOut2 += vl;
        pOut3 += vl;
        pOut4 += vl;

        rowCnt--;
    }

    //reset rows (as out_tensor_ch is 4n+2)
    if(out_tensor_ch & 0x2)
    {
        // setup pointers for A
        const q7_t *pA2 = pA + col_src1;
        // setup pointers for B
        const u8_t *pB = src2;
        const u8_t *pB2 = pB + col_src1;
        const u8_t *pB3 = pB + col_src1 * 2;
        const u8_t *pB4 = pB + col_src1 * 3;

#if (__clang__)
        register const long zero asm ("x0") = 0;
#else
        register const long zero asm ("x0");
#endif
        NDS_VEC_VSETVLI(vl, zero, NDS_VEC_VTYPE_SEW_E32, NDS_VEC_VTYPE_LMUL_M8);

        //init
        //clear v8~v23 (keep multiplication results)
        NDS_VEC_VAND_VI(NDS_VEC_V8, NDS_VEC_V8, 0x0);
        NDS_VEC_VAND_VI(NDS_VEC_V16, NDS_VEC_V16, 0x0);
        NDS_VEC_VSETVLI(vl, col_src1, NDS_VEC_VTYPE_SEW_E32, NDS_VEC_VTYPE_LMUL_M1);
        NDS_VEC_VAND_VI(NDS_VEC_V31, NDS_VEC_V31, 0x0);     //keep zero for redsum

        unsigned long colCnt = col_src1;
        while(colCnt > 0)
        {
            NDS_VEC_VSETVLI(vl, colCnt, NDS_VEC_VTYPE_SEW_E8, NDS_VEC_VTYPE_LMUL_MF2);
            //load src1 and src2
            NDS_VEC_VLB_V(NDS_VEC_V0, pA);
            NDS_VEC_VLB_V(NDS_VEC_V1, pA2);

            NDS_VEC_VLB_V(NDS_VEC_V4, pB);
            NDS_VEC_VLB_V(NDS_VEC_V5, pB2);
            NDS_VEC_VLB_V(NDS_VEC_V6, pB3);
            NDS_VEC_VLB_V(NDS_VEC_V7, pB4);

            //acc += src1 * src2
            NDS_VEC_VQMACCSU_VV(NDS_VEC_V8, NDS_VEC_V0, NDS_VEC_V4);
            NDS_VEC_VQMACCSU_VV(NDS_VEC_V10, NDS_VEC_V1, NDS_VEC_V4);

            NDS_VEC_VQMACCSU_VV(NDS_VEC_V12, NDS_VEC_V0, NDS_VEC_V5);
            NDS_VEC_VQMACCSU_VV(NDS_VEC_V14, NDS_VEC_V1, NDS_VEC_V5);

            NDS_VEC_VQMACCSU_VV(NDS_VEC_V16, NDS_VEC_V0, NDS_VEC_V6);
            NDS_VEC_VQMACCSU_VV(NDS_VEC_V18, NDS_VEC_V1, NDS_VEC_V6);

            NDS_VEC_VQMACCSU_VV(NDS_VEC_V20, NDS_VEC_V0, NDS_VEC_V7);
            NDS_VEC_VQMACCSU_VV(NDS_VEC_V22, NDS_VEC_V1, NDS_VEC_V7);

            //bump pointers and update loop counter
            colCnt -= vl;
            pA += vl;
            pA2 += vl;
            pB += vl;
            pB2 += vl;
            pB3 += vl;
            pB4 += vl;
        }

        NDS_VEC_VSETVLI(vl, col_src1, NDS_VEC_VTYPE_SEW_E32, NDS_VEC_VTYPE_LMUL_M2);    //note the avl here
        NDS_VEC_VREDSUM_VS(NDS_VEC_V0, NDS_VEC_V8, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V1, NDS_VEC_V10, NDS_VEC_V31);

        NDS_VEC_VREDSUM_VS(NDS_VEC_V4, NDS_VEC_V12, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V5, NDS_VEC_V14, NDS_VEC_V31);

        NDS_VEC_VREDSUM_VS(NDS_VEC_V8, NDS_VEC_V16, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V9, NDS_VEC_V18, NDS_VEC_V31);

        NDS_VEC_VREDSUM_VS(NDS_VEC_V12, NDS_VEC_V20, NDS_VEC_V31);
        NDS_VEC_VREDSUM_VS(NDS_VEC_V13, NDS_VEC_V22, NDS_VEC_V31);

        NDS_VEC_VSETVLI_E32(vl, 2);     //2: 2 acc
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V0, NDS_VEC_V1, 1);
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V4, NDS_VEC_V5, 1);
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V8, NDS_VEC_V9, 1);
        NDS_VEC_VSLIDEUP_VI(NDS_VEC_V12, NDS_VEC_V13, 1);

        //1st shift of 2-stage shift
        NDS_VEC_VSRA_VX(NDS_VEC_V0, NDS_VEC_V0, pre_rshift);
        NDS_VEC_VSRA_VX(NDS_VEC_V4, NDS_VEC_V4, pre_rshift);
        NDS_VEC_VSRA_VX(NDS_VEC_V8, NDS_VEC_V8, pre_rshift);
        NDS_VEC_VSRA_VX(NDS_VEC_V12, NDS_VEC_V12, pre_rshift);

        NDS_VEC_VMUL_VX(NDS_VEC_V0, NDS_VEC_V0, out_scale);
        NDS_VEC_VMUL_VX(NDS_VEC_V4, NDS_VEC_V4, out_scale);
        NDS_VEC_VMUL_VX(NDS_VEC_V8, NDS_VEC_V8, out_scale);
        NDS_VEC_VMUL_VX(NDS_VEC_V12, NDS_VEC_V12, out_scale);

        NDS_VEC_VADD_VX(NDS_VEC_V0, NDS_VEC_V0, NN_ROUND(post_rshift));
        NDS_VEC_VADD_VX(NDS_VEC_V4, NDS_VEC_V4, NN_ROUND(post_rshift));
        NDS_VEC_VADD_VX(NDS_VEC_V8, NDS_VEC_V8, NN_ROUND(post_rshift));
        NDS_VEC_VADD_VX(NDS_VEC_V12, NDS_VEC_V12, NN_ROUND(post_rshift));

        //saturate into 8-bits
        NDS_VEC_VSETVLI_E16(vl, 2);     //2: 2 acc
        NDS_VEC_VNSRA_WX(NDS_VEC_V0, NDS_VEC_V0, post_rshift);
        NDS_VEC_VNSRA_WX(NDS_VEC_V4, NDS_VEC_V4, post_rshift);
        NDS_VEC_VNSRA_WX(NDS_VEC_V8, NDS_VEC_V8, post_rshift);
        NDS_VEC_VNSRA_WX(NDS_VEC_V12, NDS_VEC_V12, post_rshift);
        NDS_VEC_VSETVLI_E8(vl, 2);     //2: 2 acc
        NDS_VEC_VNCLIP_WI(NDS_VEC_V0, NDS_VEC_V0, 0);
        NDS_VEC_VNCLIP_WI(NDS_VEC_V4, NDS_VEC_V4, 0);
        NDS_VEC_VNCLIP_WI(NDS_VEC_V8, NDS_VEC_V8, 0);
        NDS_VEC_VNCLIP_WI(NDS_VEC_V12, NDS_VEC_V12, 0);

        //store the results
        NDS_VEC_VSB_V(NDS_VEC_V0, out);
        NDS_VEC_VSB_V(NDS_VEC_V4, pOut2);
        NDS_VEC_VSB_V(NDS_VEC_V8, pOut3);
        NDS_VEC_VSB_V(NDS_VEC_V12, pOut4);
        out += vl;
        pOut2 += vl;
        pOut3 += vl;
        pOut4 += vl;
    }

    out += out_tensor_ch * 3;

    return out;
#endif
}
#endif

q7_t *riscv_nn_mat_mul_kernel_u8_q7_2sft(const q7_t * src1,
                                       const u8_t * src2,
                                       const uint16_t out_tensor_ch,
                                       const uint16_t col_src1,
                                       const uint16_t pre_rshift,
                                       const uint16_t out_scale,
                                       const uint16_t post_rshift,
                                       q7_t * out)
{
    /* To be completed */
    return NULL;

}
