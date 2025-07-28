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

#include "internal_nn_math.h"
#include "riscv_nn_support.h"
#include "riscv_nn_types.h"

int32_t riscv_nn_transpose_4d_s8(int8_t * in_tensor,
                                 const uint32_t in_dim_w,
                                 const uint32_t in_dim_z,
                                 const uint32_t in_dim_y,
                                 const uint32_t in_dim_x,
                                 const riscv_nn_transpose_format tran_fmt,
                                 int8_t * out_tensor)
{
	uint32_t i_dst_w, i_dst_z, i_dst_y, i_dst_x, i_src = 0u, i_dst = 0u;
    uint32_t vec_dst[4];	//[i_dst_w, i_dst_z, i_dst_y, i_dst_x]
    uint32_t inv_perm[4] = {0, 1, 2, 3};	//map back to in_tensor's index.
    uint32_t dst_w_end = in_dim_w, dst_z_end = in_dim_z, dst_y_end = in_dim_y, dst_x_end = in_dim_x;

    switch(tran_fmt)
    {
        case NN_WZYX_2_WZXY: //perm:[0,1,3,2]
            dst_w_end = in_dim_w; dst_z_end = in_dim_z;
            dst_y_end = in_dim_x; dst_x_end = in_dim_y;
            inv_perm[0] = 0; inv_perm[1] = 1; inv_perm[3] = 2; inv_perm[2] = 3;
            break;
        case NN_WZYX_2_WYZX: //perm:[0,2,1,3]
            dst_w_end = in_dim_w; dst_z_end = in_dim_y;
            dst_y_end = in_dim_z; dst_x_end = in_dim_x;
            inv_perm[0] = 0; inv_perm[2] = 1; inv_perm[1] = 2; inv_perm[3] = 3;
            break;
        case NN_WZYX_2_WYXZ: //perm:[0,2,3,1]
            dst_w_end = in_dim_w; dst_z_end = in_dim_y;
            dst_y_end = in_dim_x; dst_x_end = in_dim_z;
            inv_perm[0] = 0; inv_perm[2] = 1; inv_perm[3] = 2; inv_perm[1] = 3;
            break;
        case NN_WZYX_2_WXZY: //perm:[0,3,1,2]
            dst_w_end = in_dim_w; dst_z_end = in_dim_x;
            dst_y_end = in_dim_z; dst_x_end = in_dim_y;
            inv_perm[0] = 0; inv_perm[3] = 1; inv_perm[1] = 2; inv_perm[2] = 3;
            break;
        case NN_WZYX_2_WXYZ: //perm:[0,3,2,1]
            dst_w_end = in_dim_w; dst_z_end = in_dim_x;
            dst_y_end = in_dim_y; dst_x_end = in_dim_z;
            inv_perm[0] = 0; inv_perm[3] = 1; inv_perm[2] = 2; inv_perm[1] = 3;
            break;
        case NN_WZYX_2_ZWXY: //perm:[1,0,3,2]
            dst_w_end = in_dim_z; dst_z_end = in_dim_w;
            dst_y_end = in_dim_x; dst_x_end = in_dim_y;
            inv_perm[1] = 0; inv_perm[0] = 1; inv_perm[3] = 2; inv_perm[2] = 3;
            break;
        case NN_WZYX_2_ZWYX: //perm:[1,0,2,3]
            dst_w_end = in_dim_z; dst_z_end = in_dim_w;
            dst_y_end = in_dim_y; dst_x_end = in_dim_x;
            inv_perm[1] = 0; inv_perm[0] = 1; inv_perm[2] = 2; inv_perm[3] = 3;
            break;
        case NN_WZYX_2_YWZX: //perm:[2,0,1,3]
            dst_w_end = in_dim_y; dst_z_end = in_dim_w;
            dst_y_end = in_dim_z; dst_x_end = in_dim_x;
            inv_perm[2] = 0; inv_perm[0] = 1; inv_perm[1] = 2; inv_perm[3] = 3;
            break;
        default:
            return -1;
            break;
    }

    for (i_dst_w = 0u; i_dst_w < dst_w_end; i_dst_w++)
    {
        vec_dst[0]=i_dst_w;
        for (i_dst_z = 0u; i_dst_z < dst_z_end; i_dst_z++)
        {
            vec_dst[1]=i_dst_z;
            for (i_dst_y = 0u; i_dst_y < dst_y_end; i_dst_y++)
            {
                vec_dst[2]=i_dst_y;
                for (i_dst_x = 0u; i_dst_x < dst_x_end; i_dst_x++)
                {
                    vec_dst[3]=i_dst_x;
                    i_src = vec_dst[inv_perm[0]] * in_dim_z * in_dim_y * in_dim_x +
                            vec_dst[inv_perm[1]] * in_dim_y * in_dim_x +
                            vec_dst[inv_perm[2]] * in_dim_x +
                            vec_dst[inv_perm[3]];
                    out_tensor[i_dst++] = in_tensor[i_src];
                }
            }
        }
    }
	return 0;
}
