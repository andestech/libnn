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

int32_t riscv_nn_channel_shuffle_CHW_s8(int8_t * in_tensor,
                                        const uint32_t in_tensor_dim_x,
                                        const uint32_t in_tensor_dim_y,
                                        const uint32_t in_tensor_ch,
                                        const uint32_t in_tensor_batch,
                                        const uint32_t group,
                                        int8_t * out_tensor)
{
    uint32_t b, c, n, set;
    uint32_t i_src = 0u;

    if ((in_tensor_ch % group) != 0)
    {
        return -1;
    }

    set = in_tensor_ch / group;
    const uint32_t tensor_size = in_tensor_dim_x * in_tensor_dim_y;
    int8_t *out_tensor_tmp = out_tensor;

    for (b = 0; b < in_tensor_batch; b++)
    {
        for (c = 0; c < set; c++)
        {
            for (n = 0; n < group; n++)
            {
                i_src = tensor_size * (set * n + c);

                memcpy(out_tensor_tmp, in_tensor + i_src, tensor_size * sizeof(*in_tensor));

                out_tensor_tmp += tensor_size;
            }
        }
    }
    return 0;
}
