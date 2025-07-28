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

static int32_t nn_flat_size(const int32_t* shape, const int32_t dims)
{
    int32_t size = 1;
    for(int i = 0; i < dims ; i++)
    {
        size *= shape[i];
    }
    return size;
}

int32_t riscv_nn_scatter_nd_s8(int8_t * out_tensor,
                               const int32_t * out_tensor_shape,
                               const int32_t out_tensor_dim,
                               const int32_t init_val,
                               const int32_t * idx_tensor,
                               const int32_t * idx_tensor_shape,
                               const int32_t idx_tensor_dim,
                               const int8_t * update_tensor,
                               const int32_t * update_tensor_shape,
                               const int32_t update_tensor_dim,
                               int32_t * tmp_buf)
{
    int n_slices = 1;
    int slice_size = 1;
    const int outer_dim = idx_tensor_dim - 1;
    const int idx_nd = idx_tensor_shape[outer_dim];
    const int update_dim = update_tensor_dim;

    for (int i = 0; i < outer_dim; i++)
    {
        n_slices *= idx_tensor_shape[i]; //number of slices to be updated
    }
    for (int i = outer_dim; i < update_dim; i++)
    {
        slice_size *= update_tensor_shape[i]; //element size per slice
    }

    int output_flat_size = nn_flat_size(out_tensor_shape, out_tensor_dim);
    int remain_flat_size = output_flat_size;

    for (int i = 0; i < idx_nd; i++)
    {
        tmp_buf[i] = remain_flat_size / out_tensor_shape[i];
        remain_flat_size = tmp_buf[i];
    }

    // Constraint!
    if (n_slices * slice_size > nn_flat_size(update_tensor_shape, update_tensor_dim))
    {
        return -1;
    }

    memset(out_tensor, init_val, sizeof(out_tensor[0]) * output_flat_size);
    for (int i = 0; i < n_slices; i++)
    {
        int to_pos = 0;
        for (int j = 0; j < idx_nd; j++)
        {
            int32_t idx = idx_tensor[i * idx_nd + j];
            to_pos += idx * tmp_buf[j];
        }

        for (int j = 0; j < slice_size; j++)
        {
            out_tensor[to_pos + j] = update_tensor[i * slice_size + j];
        }
    }
    return 0;
}
