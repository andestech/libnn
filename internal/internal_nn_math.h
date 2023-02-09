/***************************************************************************
 *  Copyright (C) 2018 Andes Technology Corporation                        *
 *  All rights reserved.                                                   *
 ***************************************************************************/

#ifndef __INTERNAL_NN_MATH_H__
#define __INTERNAL_NN_MATH_H__

#ifdef   __cplusplus
extern "C"
{
#endif
// #include "internal_nds_config.h"
#include "internal_config.h"
#include "riscv_math_types.h"
#include <string.h>
#include <math.h>

#ifdef ENA_VEC_ISA
#include "internal_vec_isa.h"
#define NDS_VEC_VSETVLI_E8_M2(OUT, AVL) \
            NDS_VEC_VSETVLI(OUT, AVL, NDS_VEC_VTYPE_SEW_E8, NDS_VEC_VTYPE_LMUL_M2)
#define NDS_VEC_VSETVLI_E8_M4(OUT, AVL) \
            NDS_VEC_VSETVLI(OUT, AVL, NDS_VEC_VTYPE_SEW_E8, NDS_VEC_VTYPE_LMUL_M4)
#define NDS_VEC_VSETVLI_E16_M2(OUT, AVL) \
            NDS_VEC_VSETVLI(OUT, AVL, NDS_VEC_VTYPE_SEW_E16, NDS_VEC_VTYPE_LMUL_M2)
#define NDS_VEC_VSETVLI_E16_M4(OUT, AVL) \
            NDS_VEC_VSETVLI(OUT, AVL, NDS_VEC_VTYPE_SEW_E16, NDS_VEC_VTYPE_LMUL_M4)
#define NDS_VEC_VSETVLI_E32_M2(OUT, AVL) \
            NDS_VEC_VSETVLI(OUT, AVL, NDS_VEC_VTYPE_SEW_E32, NDS_VEC_VTYPE_LMUL_M2)
#define NDS_VEC_VSETVLI_E32_M4(OUT, AVL) \
            NDS_VEC_VSETVLI(OUT, AVL, NDS_VEC_VTYPE_SEW_E32, NDS_VEC_VTYPE_LMUL_M4)
#define NDS_VEC_VSETVLI_E32_M8(OUT, AVL) \
            NDS_VEC_VSETVLI(OUT, AVL, NDS_VEC_VTYPE_SEW_E32, NDS_VEC_VTYPE_LMUL_M8)
#endif

#ifdef ENA_DSP_ISA
#include "internal_dsp_isa.h"
#else
#include "internal_isa.h"
#endif

#ifdef ALWAY_INLINE
#define FUNCTION_INLINE  __attribute__((always_inline))
#else
#define FUNCTION_INLINE
#endif

#ifndef NDS_NN_TRUNCATE
    #define NN_ROUND(out_shift) ( 0x1 << (out_shift - 1) )
#else
    #define NN_ROUND(out_shift) 0
#endif

#define Q31_MAX   ((long)(q31_t)(0x7FFFFFFFL))
#define Q31_MIN   ((long)(q31_t)(0x80000000L))
#define Q15_MAX   ((long)(q15_t)(0x7FFF))
#define Q15_MIN   ((long)(q15_t)(0x8000))
#define Q7_MAX    ((long)(q7_t)(0x7F))
#define Q7_MIN    ((long)(q7_t)(0x80))
#define U8_MAX    ((unsigned long)(u8_t)(0xFF))
#define U8_MIN    ((unsigned long)(u8_t)(0x0))

#define LEFT_SHIFT(_shift)  (_shift > 0 ? _shift : 0)
#define RIGHT_SHIFT(_shift) (_shift > 0 ? 0 : -_shift)
#define MASK_IF_ZERO(x)     (x) == 0 ? ~0 : 0
#define MASK_IF_NON_ZERO(x) (x) != 0 ? ~0 : 0
#define SELECT_USING_MASK(mask, a, b) ((mask) & (a)) ^ (~(mask) & (b))

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define riscv_nn_clip_any(in, clip_min, clip_max) MIN(MAX((in), (clip_min)), (clip_max))

#ifdef ENA_DSP_ISA_64
union riscv_nnDoubleWord
{
    q63_t     dword;
               /**< Q63 type */
    q31_t     word[2];
               /**< Q31 type */
    q15_t     half_words[4];
               /**< Q15 type */
    q7_t      bytes[8];
               /**< Q7 type */
};
#endif
union riscv_nnword
{
    q31_t     word;
               /**< Q31 type */
    q15_t     half_words[2];
               /**< Q15 type */
    q7_t      bytes[4];
               /**< Q7 type */
};


// Need modified for using simd_load
//#define __SIMD32_TYPE int32_t
#define __SIMD32(addr)        (*(int32_t **) & (addr))
#define __SIMD64(addr)        (*(int64_t **) & (addr))


#ifdef ENA_DSP_ISA
#ifdef ENA_DSP_ISA_64
#define NDS_DSP_ROR(X, Y)     __nds__rotr((X), (Y))
#else
#define NDS_DSP_ROR(X, Y)     __nds32__rotr((X), (Y))
#endif
#else
#define NDS_DSP_ROR(X, Y)     NDS_ISA_ROTR((X), (Y))
#endif

static inline unsigned int NDS_ISA_ROTR(unsigned int val, unsigned int ror)
{
    unsigned int rotr = ror & 0x1f;
    return (val >> rotr) | (val << (sizeof(val)*8 - rotr));
}

#if ENA_DSP_BE
#define NDS_PACKQ7(v0,v1,v2,v3) ( (((int32_t)(v3) <<  0) & (int32_t)0x000000FF) | \
                                (((int32_t)(v2) <<  8) & (int32_t)0x0000FF00) | \
                                (((int32_t)(v1) << 16) & (int32_t)0x00FF0000) | \
                                (((int32_t)(v0) << 24) & (int32_t)0xFF000000)  )
#else
#define NDS_PACKQ7(v0,v1,v2,v3) ( (((int32_t)(v0) <<  0) & (int32_t)0x000000FF) | \
                                (((int32_t)(v1) <<  8) & (int32_t)0x0000FF00) | \
                                (((int32_t)(v2) << 16) & (int32_t)0x00FF0000) | \
                                (((int32_t)(v3) << 24) & (int32_t)0xFF000000)  )
#endif

// #if defined (ENA_DSP_ISA) || defined(ENA_VEC_ISA)
#define __STATIC_FORCEINLINE __attribute__((always_inline)) static inline
// #endif

/**
 * @brief read and expand one Q7 word into two Q15 words
 */
#ifdef ENA_DSP_ISA
__STATIC_FORCEINLINE void *read_and_pad(void *source, q31_t * out1, q31_t * out2)
{
        q31_t     inA = *__SIMD32(source)++;

#if ENA_DSP_BE
#ifdef ENA_DSP_ISA2
        *out1 = NDS_DSP_SUNPKD832(inA);
#else
        *out1 = NDS_DSP_SUNPKD810(inA >> 16);
#endif
        *out2 = NDS_DSP_SUNPKD810(inA);
#else
        *out1 = NDS_DSP_SUNPKD810(inA);
#ifdef ENA_DSP_ISA2
        *out2 = NDS_DSP_SUNPKD832(inA);
#else
        *out2 = NDS_DSP_SUNPKD810(inA >> 16);
#endif
#endif
        return source;
}
#endif

#ifdef ENA_DSP_ISA_64
__STATIC_FORCEINLINE void *read_and_pad_q63(void *source, q63_t * out1, q63_t * out2)
{
        q63_t     inA = *__SIMD64(source)++;
#if ENA_DSP_BE
        *out1 = NDS_DSP_SUNPKD832(inA);
        *out2 = NDS_DSP_SUNPKD810(inA);
#else
        *out1 = NDS_DSP_SUNPKD810(inA);
        *out2 = NDS_DSP_SUNPKD832(inA);
#endif
        return source;
}
#endif

/**
 * @brief read and expand one Q7 word into two Q15 words with reordering
 */
#ifdef ENA_DSP_ISA
__STATIC_FORCEINLINE void *read_and_pad_reordered(void *source, q31_t * out1, q31_t * out2)
{
        q31_t     inA = *__SIMD32(source)++;
#if ENA_DSP_BE
        *out1 = NDS_DSP_SUNPKD831(inA);
        *out2 = NDS_DSP_SUNPKD820(inA);
#else
        *out2 = NDS_DSP_SUNPKD831(inA);
        *out1 = NDS_DSP_SUNPKD820(inA);
#endif
        return source;
}
#endif

#ifdef ENA_DSP_ISA_64
__STATIC_FORCEINLINE void *read_and_pad_reordered_q63(void *source, long * out1, long * out2)
{
        long     inA = *__SIMD64(source)++;
#if ENA_DSP_BE
        *out1 = NDS_DSP_SUNPKD831(inA);
        *out2 = NDS_DSP_SUNPKD820(inA);
#else
        *out2 = NDS_DSP_SUNPKD831(inA);
        *out1 = NDS_DSP_SUNPKD820(inA);
#endif
        return source;
}
#endif /* ENA_DSP_ISA_64 */

/**
 * @brief used to accumulate q7 to q15 in avepool q7.
 */
#if defined(ENA_DSP_ISA) || defined(ENA_VEC_ISA)
__STATIC_FORCEINLINE void accumulate_q7_to_q15(q15_t * base, q7_t * target, const uint16_t length)
{
#ifdef ENA_VEC_ISA
    int32_t vl;
    int16_t avl = length;

    while(avl > 0)
    {
        NDS_VEC_VSETVLI_E16_M2(vl, avl);
        NDS_VEC_VLH_V(NDS_VEC_V0, base);

        NDS_VEC_VSETVLI_E8(vl, avl);
        NDS_VEC_VLB_V(NDS_VEC_V2, target);

        NDS_VEC_VWADD_WV(NDS_VEC_V0, NDS_VEC_V0, NDS_VEC_V2);

        NDS_VEC_VSETVLI_E16_M2(vl, avl);
        NDS_VEC_VSH_V(NDS_VEC_V0, base);
        base += vl;
        target += vl;
        avl -= vl;
    }
#else
    q15_t    *pCnt = base;
    q7_t     *pV = target;
    q31_t     v1, v2, vo1, vo2;
    uint16_t  cnt = length >> 2;
    q31_t     in;

    while (cnt > 0u)
    {
        q31_t     value = *__SIMD32(pV)++;
        v1 = NDS_DSP_SUNPKD831(value);
        v2 = NDS_DSP_SUNPKD820(value);
#if ENA_DSP_BE
        vo1 = NDS_DSP_PKTT16(v1, v2);
        vo2 = NDS_DSP_PKBB16(v1, v2);
#else
        vo2 = NDS_DSP_PKTT16(v1, v2);
        vo1 = NDS_DSP_PKBB16(v1, v2);
#endif

        in = *__SIMD32(pCnt);
        *__SIMD32(pCnt)++ = NDS_DSP_KADD16(vo1, in);

        in = *__SIMD32(pCnt);
        *__SIMD32(pCnt)++ = NDS_DSP_KADD16(vo2, in);

        cnt--;
    }
    cnt = length & 0x3;
    while (cnt > 0u)
    {
        *pCnt++ += *pV++;
        cnt--;
    }
#endif
}
#endif

__STATIC_FORCEINLINE void buffer_scale_back_q15_to_q7_shift(q15_t * buffer,
        q7_t * target,
        uint16_t length,
        uint16_t scale,
        const uint16_t shift)
{
#ifdef ENA_VEC_ISA
    while(length > 0)
    {
        uint32_t vl;
        NDS_VEC_VSETVLI_E16_M2(vl, length);
        NDS_VEC_VLH_V(NDS_VEC_V2, buffer);
        NDS_VEC_VSLL_VX(NDS_VEC_V2, NDS_VEC_V2, shift);
        NDS_VEC_VDIV_VX(NDS_VEC_V2, NDS_VEC_V2, scale);
        NDS_VEC_VSB_V(NDS_VEC_V2, target);

        length -= vl;
        buffer += vl;
        target += vl;
    }
#else
    int i;
    for (i = 0; i < length; i++)
        target[i] = (q7_t) ((buffer[i] << shift) / scale);
#endif
}

__STATIC_FORCEINLINE void buffer_scale_back_q15_to_q7(q15_t * buffer,
        q7_t * target,
        uint16_t length,
        uint16_t scale)
{
#ifdef ENA_VEC_ISA
    while(length > 0)
    {
        uint32_t vl;
        NDS_VEC_VSETVLI_E16_M2(vl, length);
        NDS_VEC_VLH_V(NDS_VEC_V2, buffer);
        NDS_VEC_VDIV_VX(NDS_VEC_V2, NDS_VEC_V2, scale);
        NDS_VEC_VSB_V(NDS_VEC_V2, target);

        length -= vl;
        buffer += vl;
        target += vl;
    }
#else
    int i;
    for (i = 0; i < length; i++)
        target[i] = (q7_t) (buffer[i] / scale);
#endif
}

/**
 * @brief           Saturating doubling high multiply.
 * @param[in]       m1        Multiplicand
 * @param[in]       m2        Multiplier
 * @return          Result of multiplication.
 *
 */
__STATIC_FORCEINLINE q31_t riscv_nn_sat_doubling_high_mult(const q31_t m1, const q31_t m2)
{
#ifdef ENA_DSP_ISA
    q31_t result = NDS_DSP_KWMMUL_U(m1, m2);
#else
    q31_t result = 0;
    // Rounding offset to add for a right shift of 31
    q63_t mult = 1 << 30;

    if ((m1 < 0) ^ (m2 < 0))
    {
        mult = 1 - mult;
    }
    mult = mult + (q63_t)m1 * m2;

    // Utilize all of the upper 32 bits. This is the doubling step
    // as well.
    result = mult / (1UL << 31);

    if ((m1 == m2) && (m1 == (int32_t)Q31_MIN))
    {
        result = Q31_MAX;
    }
#endif
    return result;
}

/**
 * @brief           Rounding divide by power of two.
 * @param[in]       dividend - Dividend
 * @param[in]       exponent - Divisor = power(2, exponent)
 *                             Range: [0, 31]
 * @return          Rounded result of division. Midpoint is rounded away from zero.
 *
 */
__STATIC_FORCEINLINE q31_t riscv_nn_divide_by_power_of_two(const q31_t dividend, const q31_t exponent)
{
#ifdef ENA_DSP_ISA
    q31_t result = NDS_DSP_SRA_U(dividend, exponent);
#else
    q31_t result = 0;
    const q31_t remainder_mask = (1l << exponent) - 1;
    int32_t remainder = remainder_mask & dividend;

    // Basic division
    result = dividend >> exponent;

    // Adjust 'result' for rounding (mid point away from zero)
    q31_t threshold = remainder_mask >> 1;
    if (result < 0)
    {
        threshold++;
    }
    if (remainder > threshold)
    {
        result++;
    }
#endif

    return result;
}

// following is a variant of riscv_nn_divide_by_power_of_two for softmanx
// functions to pass the coverage test
__STATIC_FORCEINLINE q31_t riscv_nn_divide_by_power_of_two_v2(const q31_t dividend, const q31_t exponent)
{
    q31_t result = 0;
    const q31_t remainder_mask = (1l << exponent) - 1;
    int32_t remainder = remainder_mask & dividend;

    // Basic division
    result = dividend >> exponent;

    // Adjust 'result' for rounding (mid point away from zero)
    q31_t threshold = remainder_mask >> 1;
    // if (result < 0)
    // {
    //     threshold++;
    // }
    if (remainder > threshold)
    {
        result++;
    }

    return result;
}

/**
 * @brief           Requantize a given value.
 * @param[in]       val         Value to be requantized
 * @param[in]       multiplier  multiplier
 * @param[in]       shift       left or right shift for 'val * multiplier'
 *
 * @return          Returns (val * multiplier)/(2 ^ shift)
 *
 */
__STATIC_FORCEINLINE q31_t riscv_nn_requantize(const q31_t val, const q31_t multiplier, const q31_t shift)
{
#ifdef ENA_DSP_ISA
    q31_t tmp = val << NDS_DSP_UCLIP32(shift, 5);
    q31_t out = riscv_nn_sat_doubling_high_mult(tmp, multiplier);
    out = riscv_nn_divide_by_power_of_two(out, NDS_DSP_UCLIP32(-shift, 5));
    return out;
#else
    q31_t tmp = val << LEFT_SHIFT(shift);
    q31_t out = riscv_nn_sat_doubling_high_mult(tmp, multiplier);
    out = riscv_nn_divide_by_power_of_two(out, RIGHT_SHIFT(shift));
    return out;
#endif
}

// Macros for shortening quantization functions' names and avoid long lines
#define MUL_SAT(a, b)  riscv_nn_sat_doubling_high_mult((a), (b))
#define MUL_POW2(a, b) riscv_nn_mult_by_power_of_two((a), (b))

#define DIV_POW2(a, b) riscv_nn_divide_by_power_of_two((a), (b))
#define DIV_POW2_V2(a, b) riscv_nn_divide_by_power_of_two_v2((a), (b))

#define EXP_ON_NEG(x)  riscv_nn_exp_on_negative_values((x))
#define ONE_OVER1(x)   riscv_nn_one_over_one_plus_x_for_x_in_0_1((x))


//----- sub-functions for softmax layer_begin -----
// @note The following functions are used only for softmax layer, scaled bits = 5 assumed

__STATIC_FORCEINLINE int32_t riscv_nn_exp_on_negative_values(int32_t val)
{
    int32_t mask  = 0;
    int32_t shift = 24;

    const int32_t val_mod_minus_quarter = (val & ((1 << shift) - 1)) - (1 << shift);
    const int32_t remainder             = val_mod_minus_quarter - val;
    const int32_t x                     = (val_mod_minus_quarter << 5) + (1 << 28);
    const int32_t x2                    = MUL_SAT(x, x);

    int32_t result = 1895147668 + MUL_SAT(1895147668, x +
        DIV_POW2(MUL_SAT(DIV_POW2(MUL_SAT(x2, x2), 2) + MUL_SAT(x2, x), 715827883) + x2, 1));

#define SELECT_IF_NON_ZERO(x)                                     \
{                                                                 \
    mask   = MASK_IF_NON_ZERO(remainder & (1 << shift++));        \
    result = SELECT_USING_MASK(mask, MUL_SAT(result, x), result); \
}

    SELECT_IF_NON_ZERO(1672461947)
    SELECT_IF_NON_ZERO(1302514674)
    SELECT_IF_NON_ZERO(790015084)
    SELECT_IF_NON_ZERO(290630308)
    SELECT_IF_NON_ZERO(39332535)
    SELECT_IF_NON_ZERO(720401)
    SELECT_IF_NON_ZERO(242)

#undef SELECT_IF_NON_ZERO

    mask = MASK_IF_ZERO(val);
    return SELECT_USING_MASK(mask, Q31_MAX, result);
}

__STATIC_FORCEINLINE q31_t riscv_nn_mult_by_power_of_two(const int32_t val, const int32_t exp)
{
#ifdef ENA_DSP_ISA
    int32_t result = NDS_DSP_KSLL(val, exp);
#else
    const int32_t thresh = ((1 << (31 - exp)) - 1);
    int32_t result = val << exp;
    result = SELECT_USING_MASK(MASK_IF_NON_ZERO(val > thresh), Q31_MAX, result);
    result = SELECT_USING_MASK(MASK_IF_NON_ZERO(val < -thresh), Q31_MIN, result);
#endif
    return result;
}

__STATIC_FORCEINLINE int32_t riscv_nn_one_over_one_plus_x_for_x_in_0_1(int32_t val)
{
    const int64_t sum = (int64_t)val + (int64_t)Q31_MAX;
    const int32_t half_denominator = (int32_t)((sum + (sum >= 0 ? 1 : -1)) / 2L);
    int32_t x = 1515870810 + MUL_SAT(half_denominator, -1010580540);

    const int32_t shift = (1 << 29);
    x += MUL_POW2(MUL_SAT(x, shift - MUL_SAT(half_denominator, x)), 2);
    x += MUL_POW2(MUL_SAT(x, shift - MUL_SAT(half_denominator, x)), 2);
    x += MUL_POW2(MUL_SAT(x, shift - MUL_SAT(half_denominator, x)), 2);

    return MUL_POW2(x, 1);
}
//----- sub-functions for softmax layer_end -----

// Exponent polynomial coefficients
extern const float CONST_COE0;
extern const float CONST_COE1;
extern const float CONST_COE2;
extern const float CONST_COE3;
extern const float CONST_COE4;
extern const float CONST_COE5;
extern const float CONST_COE6;
extern const float CONST_COE7;

//--- const values for exp_f32 ---
extern const float CONST_LN2;
extern const float CONST_INV_LN2;
extern const float CONST_INF;
extern const float CONST_MAX_INPUT;
extern const float CONST_0;
extern const int   CONST_NEGATIVE_126;

//exponential functions
extern float exp_f32(float x);

#ifdef  __cplusplus
}
#endif

#endif /* __INTERNAL_NN_MATH_H__ */
