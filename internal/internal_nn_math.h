/***************************************************************************
 *  Copyright (C) 2018-2024 Andes Technology Corporation                   *
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

#include "internal_isa.h"

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
#define MASK_IF_GREATER_THAN(a, b) (MASK_IF_NON_ZERO(a > b))
#define MASK_IF_LESS_THAN(a, b) (a < b)? ~0 : 0
#define SELECT_USING_MASK(mask, a, b) ((mask) & (a)) ^ (~(mask) & (b))

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define riscv_nn_clip_any(in, clip_min, clip_max) MIN(MAX((in), (clip_min)), (clip_max))
#define REDUCE_MULTIPLIER(_mult) ((_mult < 0x7FFF0000) ? ((_mult + (1 << 15)) >> 16) : 0x7FFF)

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


#define NDS_DSP_ROR(X, Y)     NDS_ISA_ROTR((X), (Y))

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

/**
 * @brief read and expand one Q7 word into two Q15 words with reordering
 */

/**
 * @brief used to accumulate q7 to q15 in avepool q7.
 */

__STATIC_FORCEINLINE void buffer_scale_back_q15_to_q7_shift(q15_t * buffer,
        q7_t * target,
        uint16_t length,
        uint16_t scale,
        const uint16_t shift)
{
    int i;
    for (i = 0; i < length; i++)
        target[i] = (q7_t) ((buffer[i] << shift) / scale);
}

__STATIC_FORCEINLINE void buffer_scale_back_q15_to_q7(q15_t * buffer,
        q7_t * target,
        uint16_t length,
        uint16_t scale)
{
    int i;
    for (i = 0; i < length; i++)
        target[i] = (q7_t) (buffer[i] / scale);
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
    result = mult >> 31;

    if ((m1 == m2) && (m1 == (int32_t)Q31_MIN))
    {
        result = Q31_MAX;
    }
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
    const long total_shift = 31 - shift;
    int64_t new_val = val * (int64_t)multiplier;
    int32_t result = new_val >> (total_shift - 1);
    result = (result + 1) >> 1;

    return result;
}

// variant of riscv_nn_requantize for the shift is always >= 0 (for fixing the coverage issue on RV32P algo)
__STATIC_FORCEINLINE q31_t riscv_nn_requantize_ps(const q31_t val, const q31_t multiplier, const q31_t shift)
{
    const long total_shift = 31 - shift;
    int64_t new_val = val * (int64_t)multiplier;
    int32_t result = new_val >> (total_shift - 1);
    result = (result + 1) >> 1;

    return result;
}

// variant of riscv_nn_requantize for the shift is always <= 0 (for fixing the coverage issue on RV32P algo)
__STATIC_FORCEINLINE q31_t riscv_nn_requantize_ns(const q31_t val, const q31_t multiplier, const q31_t shift)
{
    const long total_shift = 31 - shift;
    int64_t new_val = val * (int64_t)multiplier;
    int32_t result = new_val >> (total_shift - 1);
    result = (result + 1) >> 1;

    return result;
}

__STATIC_FORCEINLINE q31_t riscv_nn_requantize_s64(const q63_t val, const q31_t reduced_multiplier, const q31_t shift)
{
    const q63_t new_val = val * reduced_multiplier;

    q31_t result = new_val >> (14 - shift);
    result = (result + 1) >> 1;

    return result;
}

// Macros for shortening quantization functions' names and avoid long lines
#define MUL_SAT(a, b)  riscv_nn_sat_doubling_high_mult((a), (b))
#define MUL_POW2(a, b) riscv_nn_mult_by_power_of_two((a), (b))

#define DIV_POW2(a, b) riscv_nn_divide_by_power_of_two((a), (b))
#define DIV_POW2_V2(a, b) riscv_nn_divide_by_power_of_two_v2((a), (b))

#define EXP_ON_NEG(x)  riscv_nn_exp_on_negative_values((x))
#define ONE_OVER1(x)   riscv_nn_one_over_one_plus_x_for_x_in_0_1((x))

// this sub-function is used to make
//  - the allocated buffer size be a multiple of "align_byte"
//  - the pointers point to the address of aligned "align_byte"
// note. remember to make "align_byte" be power of 2
__STATIC_FORCEINLINE unsigned long nn_align(unsigned long src, unsigned long align_byte)
{
    // the src may be buffer size (in byte) or pointers
    src = (src + (align_byte - 1)) & ~(align_byte - 1);
    return src;
}

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
    const int32_t thresh = ((1 << (31 - exp)) - 1);
    int32_t result = val << exp;
    result = SELECT_USING_MASK(MASK_IF_NON_ZERO(val > thresh), Q31_MAX, result);
    result = SELECT_USING_MASK(MASK_IF_NON_ZERO(val < -thresh), Q31_MIN, result);
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

//----- sub-functions for integer tanh/sigmoid _begin-----

//----- sub-functions for softmax tanh/sigmoid _end -----

// Exponent polynomial coefficients
#define EXP_COE0        (float32_t)1.f
#define EXP_COE1        (float32_t)0.0416598916054f
#define EXP_COE2        (float32_t)0.500000596046f
#define EXP_COE3        (float32_t)0.0014122662833f
#define EXP_COE4        (float32_t)1.00000011921f
#define EXP_COE5        (float32_t)0.00833693705499f
#define EXP_COE6        (float32_t)0.166665703058f
#define EXP_COE7        (float32_t)0.000195780929062f

//--- const values for exp ---
#define LN2             (float32_t)0.6931471805f    /* ln(2) */
#define INV_LN2         (float32_t)1.4426950408f    /* 1/ln(2) */
#define EXP_F32_MAX     (float32_t)88.72200896539586f
#define EXP_F32_MIN     (float32_t)-87.33271909529616f
#define EXP_F16_MAX     (float16_t)11.0898f
#define EXP_F16_MIN     (float16_t)-9.7046f

//--- const values for gelu ---
#ifndef M_SQRT1_2
#define M_SQRT1_2       (float32_t)0.70710678118654752440   /* 1/sqrt(2) */
#endif
#define SQRT_2_D_PI     (M_2_SQRTPI * M_SQRT1_2)            /* sqrt( 2 / pi ) */
#define GELU_COE0       (float32_t)0.5f
#define GELU_COE1       (float32_t)0.044715f

//--- const velues for tanh ---
#define TANH_F32_MAX        (float32_t)10.f
#define TANH_F32_MIN        (float32_t)-10.f
#define TANH_F32_THR        (float32_t)5.e-3
#define TANH_F32_THR        (float32_t)5.e-3
#define CST_1               (float32_t)1.f
#define CST_2               (float32_t)2.f
#define CST_1_3             (float32_t)0.3333333f

//--- const velues for sigmoid ---
#define SIGMOID_MAX         (float32_t)10.f
#define SIGMOID_MIN         (float32_t)-10.f

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

//exponential related functions' prototype
extern float32_t exp_f32(float32_t x);
extern float32_t tanh_f32(float32_t x);
#ifdef __riscv_zfh
extern float16_t exp_f16(float16_t x);
extern float16_t tanh_f16(float16_t x);
#endif

// ACE related macros
#ifdef ENA_ACE_RVV
#define ace_exp(_RESULT, _INPUT)                        \
    __asm__ __volatile__("exp " _RESULT ", " _INPUT " \n");
#endif

//----- algorithn switches_begin -----
/*******************************************************************************
 * For some fp16 functions will be calculated with exp, we can enable this
 * switch to make the kernel calculations computed in 32-bit. By default, this
 * swtich is not enabled to get better performance.
 ******************************************************************************/
// #define ENA_KERNEL_FP32

/*******************************************************************************
 * For those functions using divisions, we could enable ENA_FAST_ALGO switch to
 * replace divisions with multiplying the reciprocal to get better performance.
 * Now ENA_FAST_ALGO switch will be enabled by default.
 ******************************************************************************/
#define ENA_FAST_ALGO

/*******************************************************************************
 * The algorithn controlling switches for the tiling algorithm.
 ******************************************************************************/
#define ENA_TILING
//----- algorithn switches_end -----

#ifdef  __cplusplus
}
#endif

#endif /* __INTERNAL_NN_MATH_H__ */
