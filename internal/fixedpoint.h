#ifndef _FIXEDPOINT_H_
#define _FIXEDPOINT_H_

// #include <assert.h>
#include <stdint.h>
#include <math.h>

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

// Part 1: Low-level integer-arithmetic primitives.
// The implementations here are generic implementations valid for
// scalar types (e.g. std::int32_t). Architecture-specific SIMD types
// (e.g. NEON int32x4_t) may be supported by providing
// specializations for them in separate files.
//
// The purpose of these primitives is two-fold:
//  - They will be used to implement higher-level fixed-point
//    abstractions, namely the FixedPoint class and its arithmetic
//    operators.
//  - They will be directly used to implement some more involved
//    fixed-point computations, e.g. the fixed-point implementation
//    of math functions such as tanh.

// Some compile-time traits around raw types to handle SIMD aspects:
// number of lanes, underlying scalar type.

// Returns a SIMD value duplicating a scalar value across all lanes.
static inline int16_t Dup16b(int16_t x) {
    return x;
}
static inline int32_t Dup32b(int32_t x) {
    return x;
}
// Plain bit-wise AND
static inline int16_t BitAnd16b(int16_t a, int16_t b) {
    return a & b;
}
static inline int32_t BitAnd32b(int32_t a, int32_t b) {
    return a & b;
}
// Plain bit-wise OR
static inline int16_t BitOr16b(int16_t a, int16_t b) {
    return a | b;
}
static inline int32_t BitOr32b(int32_t a, int32_t b) {
    return a | b;
}
// Plain bit-wise XOR
static inline int16_t BitXor16b(int16_t a, int16_t b) {
    return a ^ b;
}
static inline int32_t BitXor32b(int32_t a, int32_t b) {
    return a ^ b;
}
// Plain bit-wise NOT
static inline int16_t BitNot16b(int16_t a) {
    return ~a;
}
static inline int32_t BitNot32b(int32_t a) {
    return ~a;
}
// Integer addition. Not saturating. Overflow is undefined behavior.
static inline int16_t Add16b(int16_t a, int16_t b) {
    return a + b;
}
static inline int32_t Add32b(int32_t a, int32_t b) {
    return a + b;
}
// Integer multiplication. Not saturating. Overflow is undefined behavior.
static inline int16_t Mul16b(int16_t a, int16_t b) {
    return a * b;
}
static inline int32_t Mul32b(int32_t a, int32_t b) {
    return a * b;
}
// Integer subtraction. Not saturating. Overflow is undefined behavior.
static inline int16_t Sub16b(int16_t a, int16_t b) {
    return a - b;
}
static inline int32_t Sub32b(int32_t a, int32_t b) {
    return a - b;
}
// Integer unary negative. Not saturating. Overflow is undefined behavior.
static inline int16_t Neg16b(int16_t a) {
    return -a;
}
static inline int32_t Neg32b(int32_t a) {
    return -a;
}
// Integer arithmetic left-shift, equivalent to multiplying with a power of two.
// Negative values are OK. In case of overflow, no Undefined
// Behavior, but the results are implementation-defined (in practice,
// they currently are saturated, but we make no commitment to that). The idea
// is that the caller will want to implement the overflowing cases with
// saturation with compare-and-mask, so we don't care about the results
// in the overflow case, we just want to avoid undefined behavior.
//
// tIntegerType may be int32 or any narrower signed type.

static inline int16_t ShiftLeft16b(int16_t a, int16_t offset) {
    const int64_t wide_a = (int64_t)(a);
    const int64_t wide_shifted = wide_a * (1 << offset);
    //const auto min = numeric_limits<int16_t>::min();
    //const auto max = numeric_limits<int16_t>::max();
    const int16_t min = INT16_MIN;
    const int16_t max = INT16_MAX;
    return wide_shifted < min
    ? min
    : wide_shifted > max ? max
    : (int16_t)(wide_shifted);
}

// Integer arithmetic right-shift. Not rounding.
// Relying on implementation-defined, but in-practice-consistent,
// C++ compiler behavior.
static inline int16_t ShiftRight16b(int16_t a, int offset) {
    return a >> offset;
}
static inline int32_t ShiftRight32b(int32_t a, int offset) {
    return a >> offset;
}
// Each bit of the result is set to the corresponding bit of either then_val or
// else_val depending on whether the corresponding bit of if_mask is set.
// Equivalent to the VBSL instruction in ARM NEON.
static inline int16_t SelectUsingMask(int16_t if_mask, int16_t then_val,
                                      int16_t else_val) {
    return BitXor16b(BitAnd16b(if_mask, then_val), BitAnd16b(BitNot16b(if_mask), else_val));
}

// For each input scalar, the corresponding bits of the result are set if the
// input scalar is non-zero.
static inline int16_t MaskIfNonZero16b(int16_t a) {
    static int16_t zero = 0;
    return a ? BitNot16b(zero) : zero;
}
static inline int32_t MaskIfNonZero32b(int32_t a) {
    static int32_t zero = 0;
    return a ? BitNot32b(zero) : zero;
}
// For each input scalar, the corresponding bits of the result are set if the
// input scalar is zero.
static inline int16_t MaskIfZero(int16_t a) {
    return MaskIfNonZero16b(!a);
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars are equal.
static inline int16_t MaskIfEqual(int16_t a, int16_t b) {
    return MaskIfNonZero16b(a == b);
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars are not equal.
static inline int16_t MaskIfNotEqual(int16_t a, int16_t b) {
    return MaskIfNonZero16b(a != b);
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars a, b satisfy a > b.
static inline int16_t MaskIfGreaterThan16b(int16_t a, int16_t b) {
    return MaskIfNonZero16b(a > b);
}
static inline int32_t MaskIfGreaterThan32b(int32_t a, int32_t b) {
    return MaskIfNonZero32b(a > b);
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars a, b satisfy a >= b.
static inline int16_t MaskIfGreaterThanOrEqual16b(int16_t a, int16_t b) {
    return MaskIfNonZero16b(a >= b);
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars a, b satisfy a < b.
static inline int16_t MaskIfLessThan16b(int16_t a, int16_t b) {
    return MaskIfNonZero16b(a < b);
}
static inline int32_t MaskIfLessThan32b(int32_t a, int32_t b) {
    return MaskIfNonZero32b(a < b);
}
// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars a, b satisfy a <= b.
static inline int16_t MaskIfLessThanOrEqual16b(int16_t a, int16_t b) {
    return MaskIfNonZero16b(a <= b);
}

static inline int16_t RoundingHalfSum(int16_t a, int16_t b) {
    int32_t a32 = a;
    int32_t b32 = b;
    int32_t sum = a32 + b32;
    int32_t sign = sum >= 0 ? 1 : -1;
    return (int16_t)((sum + sign) / 2);
}

// So far this is only needed for int16.
static inline int16_t SaturatingAdd(int16_t a, int16_t b) {
    int32_t a32 = a;
    int32_t b32 = b;
    int32_t sum = a32 + b32;
    return (int16_t)(MIN((int32_t)(32767), MAX((int32_t)(-32768), sum)));
}

// Returns a+b, saturating if the integers are 16bit or narrower,
// otherwise just a plain addition.
static inline int16_t AddSaturatingIf16BitImpl_Run(int16_t a, int16_t b) {
    return SaturatingAdd(a, b);
}

static inline int16_t AddSaturatingIf16Bit(int16_t a, int16_t b) {
    return AddSaturatingIf16BitImpl_Run(a, b);
}

// Returns the integer that represents the product of two fixed-point
// numbers, interpreting all integers as fixed-point values in the
// interval [-1, 1), rounding to the nearest value, and saturating
// -1 * -1 to the maximum value (since 1 is not in the half-open
// interval [-1, 1)).
//
// [The explanation below specializes to std::int32_t for example purpose.]
//
// The mapping between IntegerType and the interval [-1, 1) is unique and
// implied by IntegerType, which is assumed to be signed. For example,
// for IntegerType==std::int32_t, the mapping is
//   real_value = integer_value / 2^31.
// So in this case, and leaving aside rounding and saturating, this
// function computes ((a / 2^31) * (b / 2^31)) * 2^31, which simplifies to
//   (a * b) / 2^31.
//
// The 'doubling' part in the name of this function comes from the fact that
// this operation is very close to a "multiply-high" operation, keeping only
// the top half bits, except that that would be effectively computing
//   (a * b) / 2^32,
// so here we are computing 2x that, since
//   1/2^31 = 2 * 1/2^32.
// The idea is to use all of the available 32 bits in the destination int32
// value.
//
// [End of the explanation specializing to int32.]
//
// This is equivalent to the VQRDMULH instruction in ARM NEON.
static inline int16_t SaturatingRoundingDoublingHighMul(int16_t a, int16_t b) {
    int16_t overflow = 0;
    overflow = a == b && a == INT16_MIN;
    int32_t a_32 = (int32_t)a;
    int32_t b_32 = (int32_t)b;
    int32_t ab_32 = a_32 * b_32;
    int16_t nudge = ab_32 >= 0 ? (1 << 14) : (1 - (1 << 14));
    int16_t ab_x2_high16 =
        (int16_t)((ab_32 + nudge) / (1 << 15));
    return overflow ? INT16_MAX : ab_x2_high16;
}

static inline int16_t SaturatingDoublingHighMul(int16_t a, int16_t b) {
    int16_t overflow = 0;
    overflow = a == b && a == INT16_MIN;
    int32_t a_32 = (int32_t)a;
    int32_t b_32 = (int32_t)b;
    int32_t ab_32 = a_32 * b_32;
    int16_t ab_x2_high16 =
        (int16_t)((ab_32) / (1 << 15));
    return overflow ? INT16_MAX : ab_x2_high16;
}

// Correctly-rounded-to-nearest division by a power-of-two.
// Also known as a rounding arithmetic right shift.
static inline int16_t RoundingDivideByPOT16b(int16_t x, int16_t exponent) {
    // assert(exponent >= 0);
    // assert(exponent <= 31);
    const int16_t mask = Dup16b((int16_t)((1ll << exponent) - 1));
    const int16_t zero = Dup16b((int16_t)0);
    const int16_t one = Dup16b((int16_t)1);
    const int16_t remainder = BitAnd16b(x, mask);
    const int16_t threshold =
        Add16b(ShiftRight16b(mask, 1), BitAnd16b(MaskIfLessThan16b(x, zero), one));
    return Add16b(ShiftRight16b(x, exponent),
                  BitAnd16b(MaskIfGreaterThan16b(remainder, threshold), one));
}

static inline int32_t RoundingDivideByPOT32b(int32_t x, int16_t exponent) {
    // assert(exponent >= 0);
    // assert(exponent <= 31);
    const int32_t mask = Dup32b((int32_t)((1ll << exponent) - 1));
    const int32_t zero = Dup32b((int32_t)0);
    const int32_t one = Dup32b((int32_t)1);
    const int32_t remainder = BitAnd32b(x, mask);
    const int32_t threshold =
        Add32b(ShiftRight32b(mask, 1), BitAnd32b(MaskIfLessThan32b(x, zero), one));
    return Add32b(ShiftRight32b(x, exponent),
                  BitAnd32b(MaskIfGreaterThan32b(remainder, threshold), one));
}

// Returns the product of a run-time integer value by a compile-time power
// of two, with either a positive exponent (equivalent to an arithmetic
// left shift, saturating) or a negative exponent (equivalent to an arithmetic
// right shift, rounding to nearest).
static inline int16_t SaturatingRoundingMultiplyByPOT(int16_t x, int16_t Exponent)
{
    int ExponentSign = (Exponent > 0 ? 1 : Exponent < 0 ? -1 : 0);
    if(ExponentSign == 1){
        const int16_t min = Dup16b(INT16_MIN);
        const int16_t max = Dup16b(INT16_MAX);
        const int ScalarIntegerTypeBits = 8 * sizeof(int16_t);

        const int32_t threshold =
            ((1 << (ScalarIntegerTypeBits - 1 - Exponent)) - 1);
        const int16_t positive_mask =
            MaskIfGreaterThan16b(x, Dup16b(threshold));
        const int16_t negative_mask =
            MaskIfLessThan16b(x, Dup16b(-threshold));

        int16_t result = ShiftLeft16b(x, Exponent);
        result = SelectUsingMask(positive_mask, max, result);
        result = SelectUsingMask(negative_mask, min, result);
        return result;
    }
    else if(ExponentSign == 0){
        return x;
    }
    else{
        return RoundingDivideByPOT16b(x, -Exponent);
    }
}

// Part 2: the FixedPoint class.

// A FixedPoint object represents a fixed-point value stored in the underlying
// integer type tRawType, if tRawType is a plain scalar integer type.
// Alternatively, tRawType may be a SIMD type (e.g. NEON int32x4_t) in which
// case a FixedPoint object represents a corresponding SIMD vector of fixed
// point values.
//
// tIntegerBits describes the range of the fixed-point format: if
// tIntegerBits == m then the range of representable values is the half-open
// interval [-2^m; 2^m) where the open boundary on the right side means that
// 2^m is not representable (how close the maximum representable value is to
// it, depends on bit-depth of tRawType).
//
// In "Q format notation",
//   https://en.wikipedia.org/wiki/Q_(number_format)
// we are describing the format
//   Qm.n
// where
//   m = tIntegerBits
// and
//   n = NumberOfBits(tRawType) - (m + 1)
// Note that the (m + 1) in the above line is because we adopt the convention
// that we count the integer bits exclusively of the sign bit; so (m + 1) is
// the total number of integer bits inclusive of the sign bit.
//
// Accordingly, the number of integral representable values in our range
//   [-2^m ; 2^m)
// is equal to 2^(m+1).

struct FixedPoint {
    int16_t i_;
    //typedef tRawType RawType;
    int kTotalBits;     // = 8 * sizeof(ScalarRawType);
    int kIntegerBits;   // = tIntegerBits;
    int kFractionalBits;// = kTotalBits - 1 - kIntegerBits;
};

typedef struct FixedPoint ScalarFixedPointType;
typedef int16_t ScalarRawType;
typedef int16_t RawType;

static void SetFixedPointBits(struct FixedPoint* ptr, int16_t IntBits){
    ptr->kIntegerBits = IntBits;
    ptr->kTotalBits = 16;
    ptr->kFractionalBits = ptr->kTotalBits - 1 - ptr->kIntegerBits;
}

static void raw(struct FixedPoint* ptr, RawType x) { ptr->i_ = x; }

/*
  static const ScalarRawType ScalarRawMin() {
    return INT16_MIN;
  }*/
static const ScalarRawType ScalarRawMax() {
    return INT16_MAX;
}

static struct FixedPoint FromRaw(RawType x, int IntBit) {
    struct FixedPoint retval;
    SetFixedPointBits(&retval, IntBit);
    raw(&retval, x);
    return retval;
}

static struct FixedPoint FromScalarRaw(ScalarRawType x, int IntBit) {
    struct FixedPoint retval;
    SetFixedPointBits(&retval, IntBit);
    raw(&retval, Dup16b(x));//assume int16_t
    return retval;
}

static struct FixedPoint ConstantPOT(int Exponent, int kFractionalBits) {
    int kOffset = kFractionalBits + Exponent;
    return FromScalarRaw(((ScalarRawType)1) << kOffset, (16 - 1 - kFractionalBits));
}

static struct FixedPoint Zero(int IntBit) { return FromScalarRaw(0, IntBit); }

#ifdef GCOV
__attribute__((noinline))
#endif
static struct FixedPoint One(struct FixedPoint* ptr) {
    return FromScalarRaw(
        ptr->kIntegerBits == 0
        ? ScalarRawMax()
        : (((ScalarRawType)1) << (ptr->kIntegerBits == 0 ? 0 : ptr->kFractionalBits)), ptr->kIntegerBits);
}

// Part 3: implementation of arithmetic operators for the
// FixedPoint class, and a few related functions.

// A FixedPoint multiplication is just a
// SaturatingRoundingDoublingHighMul operation on the underlying
// raw integer values. The IntegerBits simply add up, as is obvious
// from the fact that the range is [-2^IntegerBits, 2^IntegerBits).

static struct FixedPoint FixedPointMul(struct FixedPoint a, struct FixedPoint b){
    struct FixedPoint c;
    SetFixedPointBits(&c, a.kIntegerBits + b.kIntegerBits);
    c.i_ = SaturatingRoundingDoublingHighMul(a.i_, b.i_);
    return c;
}

// Tweaking IntegerBits gives exact multiplication by a power of two.
static struct FixedPoint ExactMulByPot(struct FixedPoint a, int16_t exponent) {
    struct FixedPoint c;
    SetFixedPointBits(&c, a.kIntegerBits + exponent);
    c.i_ = a.i_;
    return c;
}

// If we want to leave IntegerBits fixed, then multiplication
// by a power of two has to be saturating/rounding, not exact anymore.
static struct FixedPoint SaturatingRoundingMultiplyByPOT_ret_FP(struct FixedPoint a, int16_t exponent) {
    return FromRaw(SaturatingRoundingMultiplyByPOT(a.i_, exponent), a.kIntegerBits);
}

// Generic arithmetic operators.

static struct FixedPoint SelectUsingMask_ret_FP(RawType if_mask, struct FixedPoint then_val, struct FixedPoint else_val) {
    struct FixedPoint ret = FromRaw(SelectUsingMask(if_mask, then_val.i_, else_val.i_), then_val.kIntegerBits);
    SetFixedPointBits(&ret, then_val.kIntegerBits);
    return ret;
}

static struct FixedPoint AddSaturatingIf16Bit_ret_FP(struct FixedPoint a, struct FixedPoint b) {
    return FromRaw(AddSaturatingIf16Bit(a.i_, b.i_), a.kIntegerBits);
}

// Rescale changes the number of IntegerBits and updates the underlying
// raw integer value accordingly.
static struct FixedPoint Rescale(struct FixedPoint x_src, int16_t IntegerBitsDst) {
    int kExponent = x_src.kIntegerBits - IntegerBitsDst;
    struct FixedPoint result;
    SetFixedPointBits(&result, IntegerBitsDst);
    result.i_ = SaturatingRoundingMultiplyByPOT(x_src.i_, kExponent);
    return result;
}

// CheckedFixedPointConstant allows to specify fixed-point constants
// initialized as real numbers, in a way that does not compile floating-point
// arithmetic in production code, yet still checks agreement with the
// floating-point expressions when asserts are enabled.
//
// The raw integer value provided is always a int32, encoding a 32-bit
// fixed-point value, regardless of the actual Scalar type. This allows
// writing generic code that applies just as well to the 32-bit and 16-bit
// cases. In the 16-bit case, the raw integer value is internally
// rounding-shifted by 16 bits to the right.
static ScalarRawType RescaleConstantInitializer(int32_t int32_value) {
    typedef ScalarRawType ScalarRawType;
    static int ScalarTypeBits = 8 * sizeof(ScalarRawType);
    ScalarRawType ret = (ScalarRawType)(RoundingDivideByPOT32b(int32_value, 32 - ScalarTypeBits)); //assume 16b
    return ret;
}

#ifdef GEMMLOWP_ENABLE_FIXEDPOINT_CONSTANTS_CHECKS
template <typename FixedPointType>
FixedPointType CheckedFixedPointConstant(std::int32_t raw_value,
                                         double double_value) {
    const FixedPointType result = FixedPointType::FromScalarRaw(raw_value);
    // assert(result == FixedPointType::FromDouble(double_value));
    return result;
}
#define GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(FixedPointType,                    \
                                            ScalarRawInt32Value, DoubleValue)   \
    (gemmlowp::CheckedFixedPointConstant<FixedPointType>(                       \
        gemmlowp::RescaleConstantInitializer<FixedPointType>(                   \
            ScalarRawInt32Value),                                               \
            DoubleValue))

#else
#define GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(FixedPointType,                               \
                                             ScalarRawInt32Value/*, DoubleValue*/, IntBit) \
        (FromScalarRaw(RescaleConstantInitializer(ScalarRawInt32Value), IntBit))
#endif


// Implementation of exponential function.

// Returns exp(x) for x in [-1/4, 0).
static struct FixedPoint exp_on_interval_between_negative_one_quarter_and_0_excl(struct FixedPoint a) { //IntBit=0
    typedef struct FixedPoint F;
    const F constant_term =
        GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F, 1895147668, /*std::exp(-1.0 / 8.0),*/ a.kIntegerBits);
    const F constant_1_over_3 =
        GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F, 715827883, /*1.0 / 3.0,*/ a.kIntegerBits);
    // We're evaluating a Taylor expansion around -1/8, so we do the change of
    // variable: x = a + 1/8.
    // In fixed-point with 0 integer bits, 1/8 is represented by 1 << 28.
    F x, x2, x3, x4, x4_over_4, x4_over_24_plus_x3_over_6_plus_x2_over_2, temp;
    SetFixedPointBits(&x, 0);
    SetFixedPointBits(&x2, 0);
    SetFixedPointBits(&x3, 0);
    SetFixedPointBits(&x4, 0);
    SetFixedPointBits(&x4_over_4, 0);
    SetFixedPointBits(&x4_over_24_plus_x3_over_6_plus_x2_over_2, 0);
    SetFixedPointBits(&temp, 0);

    x.i_ = a.i_ + ConstantPOT(-3, a.kFractionalBits).i_;
    //x2.i_ = x.i_ * x.i_;
    x2.i_ = SaturatingRoundingDoublingHighMul(x.i_, x.i_);
    //x3.i_ = x2.i_ * x.i_;
    x3.i_ = SaturatingRoundingDoublingHighMul(x2.i_, x.i_);
    //x4.i_ = x2.i_ * x2.i_;
    x4.i_ = SaturatingRoundingDoublingHighMul(x2.i_, x2.i_);
    x4_over_4 = SaturatingRoundingMultiplyByPOT_ret_FP(x4, -2);

    //temp.i_ = (x4_over_4.i_ + x3.i_) * constant_1_over_3.i_ + x2.i_;
    temp.i_ = SaturatingRoundingDoublingHighMul((x4_over_4.i_ + x3.i_), constant_1_over_3.i_) + x2.i_;
    x4_over_24_plus_x3_over_6_plus_x2_over_2 = SaturatingRoundingMultiplyByPOT_ret_FP(temp, -1);

    temp.i_ = SaturatingRoundingDoublingHighMul(constant_term.i_, (x.i_ + x4_over_24_plus_x3_over_6_plus_x2_over_2.i_));
    return AddSaturatingIf16Bit_ret_FP(constant_term, temp);
}

static struct FixedPoint exp_on_negative_values(struct FixedPoint a) {//a: FixedPoint w/ non-zero IntBits.
    int kFractionalBits = a.kFractionalBits;
    int kIntegerBits = a.kIntegerBits;
    const struct FixedPoint kOneQuarter = ConstantPOT(-2, kFractionalBits);

    struct FixedPoint mask;
    SetFixedPointBits(&mask, kIntegerBits);
    mask.i_ = kOneQuarter.i_ - FromScalarRaw(1, kIntegerBits).i_;

    struct FixedPoint a_mod_quarter_minus_one_quarter;
    SetFixedPointBits(&a_mod_quarter_minus_one_quarter, kIntegerBits);
    a_mod_quarter_minus_one_quarter.i_ = BitAnd16b(a.i_, mask.i_) - kOneQuarter.i_; //assume 16b

    struct FixedPoint result = exp_on_interval_between_negative_one_quarter_and_0_excl( //IntBit=0.
                                   Rescale(a_mod_quarter_minus_one_quarter, 0));
    RawType remainder = a_mod_quarter_minus_one_quarter.i_ - a.i_;

#define GEMMLOWP_EXP_BARREL_SHIFTER(Exponent, FixedPointMultiplier)                         \
    typedef struct FixedPoint F;                                                            \
    if (kIntegerBits > Exponent) {                                                          \
        F kMultiplier = GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(                               \
            F, FixedPointMultiplier/*, exp(-pow(2.0, Exponent))*/, 0);                      \
        int kShiftAmount =                                                                  \
            kIntegerBits > Exponent ? kFractionalBits + Exponent : 0;                       \
        F Res_x_Multiplier;                                                                 \
        SetFixedPointBits(&Res_x_Multiplier, 0);                                            \
        Res_x_Multiplier.i_ = SaturatingRoundingDoublingHighMul(kMultiplier.i_, result.i_); \
        result = SelectUsingMask_ret_FP(                                                    \
            MaskIfNonZero16b(BitAnd16b(remainder, Dup16b(1 << kShiftAmount) )),             \
            Res_x_Multiplier, result);                                                      \
    }

    // Constants below are Q0 representations of negative exp fractionals:
    GEMMLOWP_EXP_BARREL_SHIFTER(-2, 1672461947);  // exp(-1/4)
    GEMMLOWP_EXP_BARREL_SHIFTER(-1, 1302514674);  // exp(-1/2)
    GEMMLOWP_EXP_BARREL_SHIFTER(+0, 790015084);   // exp(-1)
    GEMMLOWP_EXP_BARREL_SHIFTER(+1, 290630308);   // exp(-2)
    GEMMLOWP_EXP_BARREL_SHIFTER(+2, 39332535);    // exp(-4)
    GEMMLOWP_EXP_BARREL_SHIFTER(+3, 720401);      // exp(-8)
    GEMMLOWP_EXP_BARREL_SHIFTER(+4, 242);         // exp(-16)

    #undef GEMMLOWP_EXP_BARREL_SHIFTER

#ifndef GCOV
    int clampB = kIntegerBits > 5 ? 36 - kIntegerBits : 0;
    if (kIntegerBits > 5) {
        //IntBit same as a.
        struct FixedPoint clamp = FromScalarRaw(RescaleConstantInitializer(-(1 << clampB)), kIntegerBits);
        result = SelectUsingMask_ret_FP(MaskIfLessThan16b(a.i_, clamp.i_), Zero(kIntegerBits), result);
    }
#endif

    result = SelectUsingMask_ret_FP(MaskIfZero(a.i_), One(&result), result);
    return result;
}

// Implementation of tanh: (1 - exp(-2x)) / (1 + exp(-2x)).

// Returns (1 - x) / (1 + x) for x in (0, 1).
static struct FixedPoint one_minus_x_over_one_plus_x_for_x_in_0_1(struct FixedPoint a) {
    struct FixedPoint FP0, FP2;
    SetFixedPointBits(&FP0, 0);//F0
    SetFixedPointBits(&FP2, 2);//F2

    struct FixedPoint half_denominator;//F0
    SetFixedPointBits(&half_denominator, 0);
    half_denominator.i_ = RoundingHalfSum(a.i_, One(&FP0).i_);

    // Newton-Raphson division
    // https://en.wikipedia.org/wiki/Division_algorithm#Newton.E2.80.93Raphson_division
    // Refer to that page for the logic behind the 48/17 and 32/17 constants.
    struct FixedPoint constant_48_over_17;//F2
    SetFixedPointBits(&constant_48_over_17, 2);
    struct FixedPoint constant_neg_32_over_17;//F2
    SetFixedPointBits(&constant_neg_32_over_17, 2);
    struct FixedPoint x;
    SetFixedPointBits(&x, 2);

    constant_48_over_17 =
        GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F2, 1515870810, /*48.0 / 17.0,*/ constant_48_over_17.kIntegerBits);
    constant_neg_32_over_17 =
        GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F2, -1010580540, /*-32.0 / 17.0,*/ constant_neg_32_over_17.kIntegerBits);

    // x = constant_48_over_17 + half_denominator * constant_neg_32_over_17.
    x = AddSaturatingIf16Bit_ret_FP(constant_48_over_17, FixedPointMul(half_denominator , constant_neg_32_over_17));

    for (int i = 0; i < 3; i++) {
        struct FixedPoint half_denominator_times_x;//F2
        SetFixedPointBits(&half_denominator_times_x, 2);
        half_denominator_times_x.i_ = SaturatingRoundingDoublingHighMul(half_denominator.i_ , x.i_);

        struct FixedPoint one_minus_half_denominator_times_x;//F2
        SetFixedPointBits(&one_minus_half_denominator_times_x, 2);
        one_minus_half_denominator_times_x.i_ = One(&one_minus_half_denominator_times_x).i_ - half_denominator_times_x.i_;

        // x = x + Rescale<2>(x * one_minus_half_denominator_times_x).
        x = AddSaturatingIf16Bit_ret_FP(x, Rescale(FixedPointMul(x, one_minus_half_denominator_times_x), 2));
    }

    struct FixedPoint temp2;
    SetFixedPointBits(&temp2, 2);
    temp2.i_ = x.i_ - One(&x).i_;

    return Rescale(temp2, 0);
}

// Returns -tanh(x) for x < 0.
static struct FixedPoint neg_tanh_on_negative_values(struct FixedPoint FP){
    return one_minus_x_over_one_plus_x_for_x_in_0_1(
        exp_on_negative_values(ExactMulByPot(FP, 1)));
}

// Implementation of logistic function.

// Returns 1 / (1 + x) for x in (0, 1).
static struct FixedPoint one_over_one_plus_x_for_x_in_0_1(struct FixedPoint a) {
    struct FixedPoint FP0, FP2;
    SetFixedPointBits(&FP0, 0);//F0
    SetFixedPointBits(&FP2, 2);//F2

    struct FixedPoint half_denominator;//F0
    SetFixedPointBits(&half_denominator, 0);
    half_denominator.i_ = RoundingHalfSum(a.i_, One(&FP0).i_);

    // Newton-Raphson division
    // https://en.wikipedia.org/wiki/Division_algorithm#Newton.E2.80.93Raphson_division
    // Refer to that page for the logic behind the 48/17 and 32/17 constants.
    struct FixedPoint constant_48_over_17;//F2
    SetFixedPointBits(&constant_48_over_17, 2);
    struct FixedPoint constant_neg_32_over_17;//F2
    SetFixedPointBits(&constant_neg_32_over_17, 2);
    struct FixedPoint x;
    SetFixedPointBits(&x, 2);

    constant_48_over_17 =
        GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F2, 1515870810, /*48.0 / 17.0,*/ constant_48_over_17.kIntegerBits);
    constant_neg_32_over_17 =
        GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F2, -1010580540, /*-32.0 / 17.0,*/ constant_neg_32_over_17.kIntegerBits);

    // x = constant_48_over_17 + half_denominator * constant_neg_32_over_17.
    x = AddSaturatingIf16Bit_ret_FP(constant_48_over_17, FixedPointMul(half_denominator , constant_neg_32_over_17));

    for (int i = 0; i < 3; i++) {
        struct FixedPoint half_denominator_times_x;//F2
        SetFixedPointBits(&half_denominator_times_x, 2);
        half_denominator_times_x.i_ = SaturatingRoundingDoublingHighMul(half_denominator.i_ , x.i_);

        struct FixedPoint one_minus_half_denominator_times_x;//F2
        SetFixedPointBits(&one_minus_half_denominator_times_x, 2);
        one_minus_half_denominator_times_x.i_ = One(&one_minus_half_denominator_times_x).i_ - half_denominator_times_x.i_;

        // x = x + Rescale<2>(x * one_minus_half_denominator_times_x).
        x = AddSaturatingIf16Bit_ret_FP(x, Rescale(FixedPointMul(x, one_minus_half_denominator_times_x), 2));
    }

    return Rescale(ExactMulByPot(x, -1), 0);
}

// Returns logistic(x) = 1 / (1 + exp(-x)) for x > 0.
static struct FixedPoint logistic_on_positive_values(struct FixedPoint a) {
    a.i_ = -a.i_;
    return one_over_one_plus_x_for_x_in_0_1(exp_on_negative_values(a));
}

// Returns logistic(x) = 1 / (1 + exp(-x)) for any x.
static inline void logistic(const struct FixedPoint tInFP, struct FixedPoint* pOutFP) {
    const int InIntBit = tInFP.kIntegerBits;
    const int OutIntBit = pOutFP->kIntegerBits;
    struct FixedPoint abs_input           = FromRaw(0, InIntBit);
    struct FixedPoint neg_tInFP           = FromRaw(0, InIntBit);
    struct FixedPoint FP0                 = FromRaw(0, OutIntBit);
    struct FixedPoint result_if_positive  = FromRaw(0, OutIntBit);
    struct FixedPoint result_if_negative  = FromRaw(0, OutIntBit);
    struct FixedPoint one_half            = FromRaw(0, OutIntBit);

    RawType mask_if_positive = MaskIfGreaterThan16b(tInFP.i_, Zero(InIntBit).i_);
    RawType mask_if_zero = MaskIfZero(tInFP.i_);

    neg_tInFP.i_ = -tInFP.i_;
    abs_input = SelectUsingMask_ret_FP(mask_if_positive, tInFP, neg_tInFP);
    result_if_positive = logistic_on_positive_values(abs_input);
    result_if_negative.i_ = One(&FP0).i_ - result_if_positive.i_;

    one_half = GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(FixedPoint::ScalarFixedPointType, 1 << 30, /*0.5,*/ one_half.kIntegerBits);

    *pOutFP = SelectUsingMask_ret_FP(mask_if_zero, one_half,
                                     SelectUsingMask_ret_FP(mask_if_positive, result_if_positive, result_if_negative));
}

// Returns tanh(x) for any x.
static inline void tanh_s16(const struct FixedPoint tInFP, struct FixedPoint* pOutFP) {
    const int InIntBit = tInFP.kIntegerBits;
    const int OutIntBit = pOutFP->kIntegerBits;
    struct FixedPoint neg_tInFP   = FromRaw(0, InIntBit);
    struct FixedPoint n           = FromRaw(0, InIntBit);
    struct FixedPoint t           = FromRaw(0, OutIntBit);
    struct FixedPoint neg_t       = FromRaw(0, OutIntBit);
    RawType mask_if_negative = MaskIfLessThan16b(tInFP.i_, Zero(tInFP.kIntegerBits).i_);
    RawType mask_if_zero = MaskIfZero(tInFP.i_);
    neg_tInFP.i_ = -tInFP.i_;
    n = SelectUsingMask_ret_FP(mask_if_negative, tInFP, neg_tInFP);
    t = neg_tanh_on_negative_values(n);
    neg_t.i_ = -t.i_;
    *pOutFP = SelectUsingMask_ret_FP(mask_if_zero, Zero(t.kIntegerBits),
                                     SelectUsingMask_ret_FP(mask_if_negative, neg_t, t));
}
#endif
