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

/** @file*/

#include "internal_nn_math.h"
#include "riscv_nn_support.h"

//// Convolution Functions

static void nn_depthwise_conv_s4_generic(const int8_t *in_tensor,
                                         const int32_t in_tensor_batch,
                                         const int32_t in_tensor_dim_x,
                                         const int32_t in_tensor_dim_y,
                                         const int32_t in_tensor_ch,
                                         const int8_t *ker_weight,
                                         const int32_t out_tensor_ch,
                                         const int32_t ch_mult,
                                         const int32_t ker_dim_x,
                                         const int32_t ker_dim_y,
                                         const int32_t pad_x,
                                         const int32_t pad_y,
                                         const int32_t stride_x,
                                         const int32_t stride_y,
                                         const int32_t *bias,
                                         int8_t *out_tensor,
                                         const int32_t *out_shift,
                                         const int32_t *out_scale,
                                         const int32_t out_tensor_dim_x,
                                         const int32_t out_tensor_dim_y,
                                         const int32_t out_offset,
                                         const int32_t in_offset,
                                         const int32_t act_min,
                                         const int32_t act_max,
                                         const int32_t dilation_x,
                                         const int32_t dilation_y)

{
    (void)out_tensor_ch;
    int i_out = 0;
    int i_batch;

    const int32_t kernel_index_offset = in_tensor_ch >> 1;
    if (!(in_tensor_ch % 2))
    {
        for (i_batch = 0; i_batch < in_tensor_batch; i_batch++)
        {
            for (int i_out_y = 0; i_out_y < out_tensor_dim_y; i_out_y++)
            {
                const int16_t base_idx_y = (i_out_y * stride_y) - pad_y;
                for (int i_out_x = 0; i_out_x < out_tensor_dim_x; i_out_x++)
                {
                    const int16_t base_idx_x = (i_out_x * stride_x) - pad_x;
                    int idx_out_ch_s4 = 0;
                    int get_low_nibble = 1;

                    // If ch_mult is 1 we can process 2 outputs at a time by doing 2 in_tensor_ch iterations
                    if (ch_mult == 1)
                    {
                        for (int i_input_ch = 0; i_input_ch < in_tensor_ch; i_input_ch += 2, idx_out_ch_s4++)
                        {
                            int32_t acc_0 = 0;
                            int32_t acc_1 = 0;

                            int ker_y_start;
                            int ker_x_start;
                            int ker_y_end;
                            int ker_x_end;

                            if (dilation_x > 1)
                            {
                                const int32_t start_x_max = (-base_idx_x + dilation_x - 1) / dilation_x;
                                ker_x_start = MAX(0, start_x_max);
                                const int32_t end_min_x = (in_tensor_dim_x - base_idx_x + dilation_x - 1) / dilation_x;
                                ker_x_end = MIN(ker_dim_x, end_min_x);
                            }
                            else
                            {
                                ker_x_start = MAX(0, -base_idx_x);
                                ker_x_end = MIN(ker_dim_x, in_tensor_dim_x - base_idx_x);
                            }

                            if (dilation_y > 1)
                            {
                                const int32_t start_y_max = (-base_idx_y + dilation_y - 1) / dilation_y;
                                ker_y_start = MAX(0, start_y_max);
                                const int32_t end_min_y = (in_tensor_dim_y - base_idx_y + dilation_y - 1) / dilation_y;
                                ker_y_end = MIN(ker_dim_y, end_min_y);
                            }
                            else
                            {
                                ker_y_start = MAX(0, -base_idx_y);
                                ker_y_end = MIN(ker_dim_y, in_tensor_dim_y - base_idx_y);
                            }

                            if (bias)
                            {
                                acc_0 = bias[i_input_ch];
                                acc_1 = bias[i_input_ch + 1];
                            }

                            int32_t idx_y = base_idx_y + dilation_y * ker_y_start;
                            for (int i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
                            {
                                int32_t idx_x = base_idx_x + dilation_x * ker_x_start;
                                int32_t idx_0 = (idx_y * in_tensor_dim_x + idx_x) * in_tensor_ch + i_input_ch;

                                int32_t ker_idx_0 =
                                    (i_ker_y * ker_dim_x + ker_x_start) * kernel_index_offset + idx_out_ch_s4;

                                for (int i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                                {
                                    int8_t ker_val0, ker_val1;

                                    ker_val0 = ((int8_t)(ker_weight[ker_idx_0] << 4) >> 4);
                                    ker_val1 = (ker_weight[ker_idx_0] >> 4);

                                    acc_0 += (in_tensor[idx_0] + in_offset) * ker_val0;
                                    acc_1 += (in_tensor[idx_0 + 1] + in_offset) * ker_val1;

                                    idx_0 += dilation_x * in_tensor_ch;
                                    idx_x += dilation_x;
                                    ker_idx_0 += kernel_index_offset;
                                }
                                idx_y += dilation_y;
                            }

                            /* Requantize and clamp out_tensor to provided range */
                            acc_0 = riscv_nn_requantize(acc_0, out_scale[i_input_ch], out_shift[i_input_ch]);
                            acc_0 += out_offset;
                            acc_0 = MAX(acc_0, act_min);
                            acc_0 = MIN(acc_0, act_max);
                            out_tensor[i_out++] = acc_0;

                            acc_1 = riscv_nn_requantize(acc_1, out_scale[i_input_ch + 1], out_shift[i_input_ch + 1]);
                            acc_1 += out_offset;
                            acc_1 = MAX(acc_1, act_min);
                            acc_1 = MIN(acc_1, act_max);
                            out_tensor[i_out++] = acc_1;
                        }
                    }
                    // if ch_mult is odd and greater than 1, we need to continue to process 1 out_tensor at a time
                    else if (ch_mult % 2)
                    {
                        for (int i_input_ch = 0; i_input_ch < in_tensor_ch; i_input_ch++)
                        {
                            for (int i_ch_mult = 0; i_ch_mult < ch_mult; i_ch_mult++)
                            {
                                const int idx_out_ch = i_ch_mult + i_input_ch * ch_mult;
                                if (idx_out_ch && (idx_out_ch % 2 == 0))
                                {
                                    idx_out_ch_s4++;
                                }

                                int32_t acc_0 = 0;

                                int ker_y_start;
                                int ker_x_start;
                                int ker_y_end;
                                int ker_x_end;

                                if (dilation_x > 1)
                                {
                                    const int32_t start_x_max = (-base_idx_x + dilation_x - 1) / dilation_x;
                                    ker_x_start = MAX(0, start_x_max);
                                    const int32_t end_min_x = (in_tensor_dim_x - base_idx_x + dilation_x - 1) / dilation_x;
                                    ker_x_end = MIN(ker_dim_x, end_min_x);
                                }
                                else
                                {
                                    ker_x_start = MAX(0, -base_idx_x);
                                    ker_x_end = MIN(ker_dim_x, in_tensor_dim_x - base_idx_x);
                                }

                                if (dilation_y > 1)
                                {
                                    const int32_t start_y_max = (-base_idx_y + dilation_y - 1) / dilation_y;
                                    ker_y_start = MAX(0, start_y_max);
                                    const int32_t end_min_y = (in_tensor_dim_y - base_idx_y + dilation_y - 1) / dilation_y;
                                    ker_y_end = MIN(ker_dim_y, end_min_y);
                                }
                                else
                                {
                                    ker_y_start = MAX(0, -base_idx_y);
                                    ker_y_end = MIN(ker_dim_y, in_tensor_dim_y - base_idx_y);
                                }

                                if (bias)
                                {
                                    acc_0 = bias[idx_out_ch];
                                }

                                int32_t idx_y = base_idx_y + dilation_y * ker_y_start;
                                for (int i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
                                {
                                    int32_t idx_x = base_idx_x + dilation_x * ker_x_start;
                                    int32_t idx_0 = (idx_y * in_tensor_dim_x + idx_x) * in_tensor_ch + i_input_ch;

                                    int32_t ker_idx_0 =
                                        (i_ker_y * ker_dim_x + ker_x_start) * (kernel_index_offset * ch_mult) +
                                        idx_out_ch_s4;

                                    for (int i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                                    {
                                        int8_t ker_val0;

                                        if (get_low_nibble)
                                        {
                                            ker_val0 = ((int8_t)(ker_weight[ker_idx_0] << 4) >> 4);
                                        }
                                        else
                                        {
                                            ker_val0 = (ker_weight[ker_idx_0] >> 4);
                                        }

                                        acc_0 += (in_tensor[idx_0] + in_offset) * ker_val0;

                                        idx_0 += dilation_x * in_tensor_ch;
                                        idx_x += dilation_x;
                                        ker_idx_0 += (kernel_index_offset * ch_mult);
                                    }
                                    idx_y += dilation_y;
                                }
                                get_low_nibble = !get_low_nibble;

                                /* Requantize and clamp out_tensor to provided range */
                                acc_0 = riscv_nn_requantize(acc_0, out_scale[idx_out_ch], out_shift[idx_out_ch]);
                                acc_0 += out_offset;
                                acc_0 = MAX(acc_0, act_min);
                                acc_0 = MIN(acc_0, act_max);
                                out_tensor[i_out++] = acc_0;
                            }
                        }
                    }
                    // if ch_mult is even then we can do 2 outputs at a time by processing 2 ch_mult iterations
                    else
                    {
                        for (int i_input_ch = 0; i_input_ch < in_tensor_ch; i_input_ch++)
                        {
                            // ch_mult is limited to being a multiple of in_tensor_ch.
                            // This means that we can assume ch_mult is a multiple of 2 given that in_tensor_ch is even
                            for (int i_ch_mult = 0; i_ch_mult < ch_mult; i_ch_mult += 2, idx_out_ch_s4++)
                            {
                                const int idx_out_ch = i_ch_mult + i_input_ch * ch_mult;

                                int32_t acc_0 = 0;
                                int32_t acc_1 = 0;

                                int ker_y_start;
                                int ker_x_start;
                                int ker_y_end;
                                int ker_x_end;

                                if (dilation_x > 1)
                                {
                                    const int32_t start_x_max = (-base_idx_x + dilation_x - 1) / dilation_x;
                                    ker_x_start = MAX(0, start_x_max);
                                    const int32_t end_min_x = (in_tensor_dim_x - base_idx_x + dilation_x - 1) / dilation_x;
                                    ker_x_end = MIN(ker_dim_x, end_min_x);
                                }
                                else
                                {
                                    ker_x_start = MAX(0, -base_idx_x);
                                    ker_x_end = MIN(ker_dim_x, in_tensor_dim_x - base_idx_x);
                                }

                                if (dilation_y > 1)
                                {
                                    const int32_t start_y_max = (-base_idx_y + dilation_y - 1) / dilation_y;
                                    ker_y_start = MAX(0, start_y_max);
                                    const int32_t end_min_y = (in_tensor_dim_y - base_idx_y + dilation_y - 1) / dilation_y;
                                    ker_y_end = MIN(ker_dim_y, end_min_y);
                                }
                                else
                                {
                                    ker_y_start = MAX(0, -base_idx_y);
                                    ker_y_end = MIN(ker_dim_y, in_tensor_dim_y - base_idx_y);
                                }

                                if (bias)
                                {
                                    acc_0 = bias[idx_out_ch];
                                    acc_1 = bias[idx_out_ch + 1];
                                }

                                int32_t idx_y = base_idx_y + dilation_y * ker_y_start;
                                for (int i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
                                {
                                    int32_t idx_x = base_idx_x + dilation_x * ker_x_start;
                                    int32_t idx_0 = (idx_y * in_tensor_dim_x + idx_x) * in_tensor_ch + i_input_ch;

                                    int32_t ker_idx_0 =
                                        (i_ker_y * ker_dim_x + ker_x_start) * (kernel_index_offset * ch_mult) +
                                        idx_out_ch_s4;

                                    for (int i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                                    {
                                        int8_t ker_val0, ker_val1;

                                        ker_val0 = ((int8_t)(ker_weight[ker_idx_0] << 4) >> 4);
                                        ker_val1 = (ker_weight[ker_idx_0] >> 4);

                                        acc_0 += (in_tensor[idx_0] + in_offset) * ker_val0;
                                        acc_1 += (in_tensor[idx_0] + in_offset) * ker_val1;

                                        idx_0 += dilation_x * in_tensor_ch;
                                        idx_x += dilation_x;
                                        ker_idx_0 += (kernel_index_offset * ch_mult);
                                    }
                                    idx_y += dilation_y;
                                }

                                /* Requantize and clamp out_tensor to provided range */
                                acc_0 = riscv_nn_requantize(acc_0, out_scale[idx_out_ch], out_shift[idx_out_ch]);
                                acc_0 += out_offset;
                                acc_0 = MAX(acc_0, act_min);
                                acc_0 = MIN(acc_0, act_max);
                                out_tensor[i_out++] = acc_0;

                                acc_1 =
                                    riscv_nn_requantize(acc_1, out_scale[idx_out_ch + 1], out_shift[idx_out_ch + 1]);
                                acc_1 += out_offset;
                                acc_1 = MAX(acc_1, act_min);
                                acc_1 = MIN(acc_1, act_max);
                                out_tensor[i_out++] = acc_1;
                            }
                        }
                    }
                }
            }
            /* Advance to the next batch */
            in_tensor += (in_tensor_dim_x * in_tensor_dim_y * in_tensor_ch);
        }
    }
    else
    {
        for (i_batch = 0; i_batch < in_tensor_batch; i_batch++)
        {
            for (int i_out_y = 0; i_out_y < out_tensor_dim_y; i_out_y++)
            {
                const int16_t base_idx_y = (i_out_y * stride_y) - pad_y;
                for (int i_out_x = 0; i_out_x < out_tensor_dim_x; i_out_x++)
                {
                    const int16_t base_idx_x = (i_out_x * stride_x) - pad_x;
                    int get_low_nibble = 1;

                    for (int i_input_ch = 0; i_input_ch < in_tensor_ch; i_input_ch++)
                    {
                        for (int i_ch_mult = 0; i_ch_mult < ch_mult; i_ch_mult++)
                        {
                            const int idx_out_ch = i_ch_mult + i_input_ch * ch_mult;
                            int32_t acc_0 = 0;

                            int ker_y_start;
                            int ker_x_start;
                            int ker_y_end;
                            int ker_x_end;

                            if (dilation_x > 1)
                            {
                                const int32_t start_x_max = (-base_idx_x + dilation_x - 1) / dilation_x;
                                ker_x_start = MAX(0, start_x_max);
                                const int32_t end_min_x = (in_tensor_dim_x - base_idx_x + dilation_x - 1) / dilation_x;
                                ker_x_end = MIN(ker_dim_x, end_min_x);
                            }
                            else
                            {
                                ker_x_start = MAX(0, -base_idx_x);
                                ker_x_end = MIN(ker_dim_x, in_tensor_dim_x - base_idx_x);
                            }

                            if (dilation_y > 1)
                            {
                                const int32_t start_y_max = (-base_idx_y + dilation_y - 1) / dilation_y;
                                ker_y_start = MAX(0, start_y_max);
                                const int32_t end_min_y = (in_tensor_dim_y - base_idx_y + dilation_y - 1) / dilation_y;
                                ker_y_end = MIN(ker_dim_y, end_min_y);
                            }
                            else
                            {
                                ker_y_start = MAX(0, -base_idx_y);
                                ker_y_end = MIN(ker_dim_y, in_tensor_dim_y - base_idx_y);
                            }

                            if (bias)
                            {
                                acc_0 = bias[idx_out_ch];
                            }

                            int32_t idx_y = base_idx_y + dilation_y * ker_y_start;
                            for (int i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
                            {
                                int32_t idx_x = base_idx_x + dilation_x * ker_x_start;
                                int32_t idx_0 = (idx_y * in_tensor_dim_x + idx_x) * in_tensor_ch + i_input_ch;

                                for (int i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                                {
                                    int8_t ker_val;
                                    int32_t ker_idx_0 = (i_ker_y * ker_dim_x + i_ker_x) * (in_tensor_ch * ch_mult) + idx_out_ch;
                                    get_low_nibble = !(ker_idx_0 & 0x1);
                                    ker_idx_0 >>= 1;    //convert the idx from s4 to s8 unit

                                    if (get_low_nibble)
                                    {
                                        ker_val = ((int8_t)(ker_weight[ker_idx_0] << 4) >> 4);
                                    }
                                    else
                                    {
                                        ker_val = (ker_weight[ker_idx_0] >> 4);
                                    }

                                    acc_0 += (in_tensor[idx_0] + in_offset) * ker_val;
                                    idx_0 += dilation_x * in_tensor_ch;
                                    idx_x += dilation_x;
                                }
                                idx_y += dilation_y;
                            }

                            /* Requantize and clamp out_tensor to provided range */
                            acc_0 = riscv_nn_requantize(acc_0, out_scale[idx_out_ch], out_shift[idx_out_ch]);
                            acc_0 += out_offset;
                            acc_0 = MAX(acc_0, act_min);
                            acc_0 = MIN(acc_0, act_max);

                            out_tensor[i_out++] = acc_0;
                        }
                    }
                }
            }

            /* Advance to the next batch */
            in_tensor += (in_tensor_dim_x * in_tensor_dim_y * in_tensor_ch);
        }
    }
}

int32_t riscv_nn_conv_dw_HWC_s8_s8_s4_asym_bias_any(const int8_t * in_tensor,
                                                    const int32_t in_tensor_batch,
                                                    const int32_t in_tensor_dim_x,
                                                    const int32_t in_tensor_dim_y,
                                                    const int32_t in_tensor_ch,
                                                    const int8_t * ker_weight,
                                                    const int32_t out_tensor_ch,
                                                    const int32_t ch_mult,
                                                    const int32_t ker_dim_x,
                                                    const int32_t ker_dim_y,
                                                    const int32_t pad_x,
                                                    const int32_t pad_y,
                                                    const int32_t stride_x,
                                                    const int32_t stride_y,
                                                    const int32_t * bias,
                                                    int8_t * out_tensor,
                                                    const int32_t * out_shift,
                                                    const int32_t * out_scale,
                                                    const int32_t out_tensor_dim_x,
                                                    const int32_t out_tensor_dim_y,
                                                    const int32_t out_offset,
                                                    const int32_t in_offset,
                                                    const int32_t act_min,
                                                    const int32_t act_max,
                                                    const int32_t dilation_x,
                                                    const int32_t dilation_y,
                                                    int8_t * tmp_buf)
{
    (void)tmp_buf;

    nn_depthwise_conv_s4_generic(in_tensor,
                              in_tensor_batch,
                              in_tensor_dim_x,
                              in_tensor_dim_y,
                              in_tensor_ch,
                              ker_weight,
                              out_tensor_ch,
                              ch_mult,
                              ker_dim_x,
                              ker_dim_y,
                              pad_x,
                              pad_y,
                              stride_x,
                              stride_y,
                              bias,
                              out_tensor,
                              out_shift,
                              out_scale,
                              out_tensor_dim_x,
                              out_tensor_dim_y,
                              out_offset,
                              in_offset,
                              act_min,
                              act_max,
                              dilation_x,
                              dilation_y);

    return 0;
}

int32_t riscv_nn_conv_dw_HWC_s8_s8_s4_asym_bias_any_get_buffer_size(const int32_t in_tensor_ch,
                                                                    const int32_t ker_dim_x,
                                                                    const int32_t ker_dim_y,
                                                                    const int32_t ch_mult)
{
    (void)in_tensor_ch;
    (void)ker_dim_x;
    (void)ker_dim_y;
    return 0;

}
