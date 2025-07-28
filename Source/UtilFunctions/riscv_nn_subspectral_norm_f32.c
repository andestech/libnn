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
// #include "riscv_nn_util.h"
// #include "riscv_nn_convolution.h"
#include "riscv_nn_types.h"

int32_t riscv_nn_subspectral_norm_f32(float32_t * in_tensor,
                                      const uint32_t in_dim_batch,
                                      const uint32_t in_dim_freq,
                                      const uint32_t in_dim_time,
                                      const uint32_t in_dim_ch,
                                      const float32_t epsilon,
                                      const float32_t * beta,           // element size: [spec_groups_num * in_dim_ch]
                                      const float32_t * gamma,          // element size: [spec_groups_num * in_dim_ch]
                                      const float32_t * means,          // element size: [spec_groups_num * in_dim_ch]
                                      const float32_t * vars,           // element size: [spec_groups_num * in_dim_ch]
                                      const uint16_t ker_dim_x,         // always 1
                                      const uint16_t ker_dim_y,         // always 1
                                      const uint32_t spec_groups_num,
                                      float32_t * out_tensor,           // element size: same as in_tensor
                                      float32_t * out_tensor_tmp_buff,  // element size: same as in_tensor
                                      float32_t * ker_weight_tmp_buff,  // element size: [spec_groups_num * in_dim_ch]
                                      float32_t * bias_tmp_buff         // element size: [spec_groups_num * in_dim_ch]
)
{
    if(in_dim_freq % spec_groups_num != 0)
        return -1;

    int32_t freq_step = in_dim_freq / spec_groups_num;

    //Construct weight, bias for conv_dw.
    for(int gp = 0 ; gp < spec_groups_num ; gp++)
    {
        for(int i = 0; i < in_dim_ch; i++)
        {
            uint32_t out_idx = gp * in_dim_ch + i;
            uint32_t in_idx = out_idx;
            ker_weight_tmp_buff[out_idx] = gamma[in_idx] / sqrtf(vars[in_idx] + epsilon);
            bias_tmp_buff[out_idx] = beta[in_idx] - gamma[in_idx] * means[in_idx] / sqrtf(vars[in_idx] + epsilon);
        }
    }

    for(int batch_i = 0; batch_i < in_dim_batch; batch_i++)
    {
        float32_t* p_In  = in_tensor  + batch_i * (in_dim_freq * in_dim_time * in_dim_ch);
        float32_t* p_Out = out_tensor + batch_i * (in_dim_freq * in_dim_time * in_dim_ch);

        float32_t* p_weight = ker_weight_tmp_buff;
        float32_t* p_bias = bias_tmp_buff;

        for(int group_i = 0; group_i < spec_groups_num; group_i++)
        {
            for (int j = 0; j < freq_step * in_dim_time; j++)
            {
                for (int ch = 0; ch < in_dim_ch; ch++)
                {
                    *p_Out++ = *p_In++ * p_weight[ch] + p_bias[ch];
                }
            }
            p_weight += in_dim_ch;
            p_bias += in_dim_ch;
        }
    }
    return 0;
}
