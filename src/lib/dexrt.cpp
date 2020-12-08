// Copyright 2019 Google LLC
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include <float.h>
#include <inttypes.h>

#ifdef DEX_CUDA
#include <cuda.h>
#endif // DEX_CUDA

#define STRING_EXPANDED(string) #string
#define STRING(string) STRING_EXPANDED(string)

extern "C" {

char* malloc_dex(int64_t nbytes) {
  // XXX: Changes to this value might require additional changes to parameter attributes in LLVM
  static const int64_t alignment = 64;
  char *ptr;
  if (posix_memalign(reinterpret_cast<void**>(&ptr), alignment, nbytes)) {
    fprintf(stderr, "Failed to allocate %ld bytes", (long)nbytes);
    std::abort();
  }
  return ptr;
}

void free_dex(char* ptr) {
  free(ptr);
}

uint32_t rotate_left(uint32_t x, uint32_t d) {
  return (x << d) | (x >> (32 - d));
}

uint64_t apply_round(uint32_t x, uint32_t y, int rot) {
  uint64_t out;

  x = x + y;
  y = rotate_left(y, rot);
  y = x ^ y;

  out = (uint64_t) x;
  out = (out << 32) | y;
  return out;
}

uint64_t threefry2x32(uint64_t keypair, uint64_t count) {
  /* Based on jax's threefry_2x32 by Matt Johnson and Peter Hawkins */

  uint32_t k0;
  uint32_t k1;
  uint32_t k2;

  uint32_t x;
  uint32_t y;

  uint64_t out;
  int i;

  int rotations1[4] = {13, 15, 26, 6};
  int rotations2[4] = {17, 29, 16, 24};

  k0 = (uint32_t) (keypair >> 32);
  k1 = (uint32_t) keypair;
  k2 = k0 ^ k1 ^ 0x1BD11BDA;
  x = (uint32_t) (count >> 32);
  y = (uint32_t) count;

  x = x + k0;
  y = y + k1;


  for (i=0;i<4;i++) {
    count = apply_round(x, y, rotations1[i]);
    x = (uint32_t) (count >> 32);
    y = (uint32_t) count;
  }
  x = x + k1;
  y = y + k2 + 1;


  for (i=0;i<4;i++) {
    count = apply_round(x, y, rotations2[i]);
    x = (uint32_t) (count >> 32);
    y = (uint32_t) count;
  }
  x = x + k2;
  y = y + k0 + 2;

  for (i=0;i<4;i++) {
    count = apply_round(x, y, rotations1[i]);
    x = (uint32_t) (count >> 32);
    y = (uint32_t) count;
  }
  x = x + k0;
  y = y + k1 + 3;

  for (i=0;i<4;i++) {
    count = apply_round(x, y, rotations2[i]);
    x = (uint32_t) (count >> 32);
    y = (uint32_t) count;
  }
  x = x + k1;
  y = y + k2 + 4;

  for (i=0;i<4;i++) {
    count = apply_round(x, y, rotations1[i]);
    x = (uint32_t) (count >> 32);
    y = (uint32_t) count;
  }
  x = x + k2;
  y = y + k0 + 5;

  out = (uint64_t) x;
  out = (out << 32) | y;
  return out;
}

long randint(uint64_t keypair, long nmax) {
  return keypair % nmax; // TODO: correct this with rejection sampling or more bits
}

double randunif(uint64_t keypair) {
  /* Assumes 1023 offset and 52 mantissa bits and probably very platform-specific. */
  uint64_t mantissa_bits;
  uint64_t exponent_bits;
  uint64_t bits;

  mantissa_bits = keypair & ((((uint64_t) 1) << 52) - 1);
  exponent_bits = ((uint64_t) 1023) << 52;
  bits = mantissa_bits | exponent_bits;

  double out = *(double*)&bits;
  return out - 1;
}

void showInt32(char **resultPtr, int32_t x) {
  auto p = reinterpret_cast<char*>(malloc_dex(100));
  auto n = sprintf(p, "%" PRId32, x);
  auto result1Ptr = reinterpret_cast<int32_t*>(resultPtr[0]);
  auto result2Ptr = reinterpret_cast<char**>(  resultPtr[1]);
  *result1Ptr = n;
  *result2Ptr = p;
}

void showInt64(char **resultPtr, int64_t x) {
  auto p = reinterpret_cast<char*>(malloc_dex(100));
  auto n = sprintf(p, "%" PRId64, x);
  auto result1Ptr = reinterpret_cast<int32_t*>(resultPtr[0]);
  auto result2Ptr = reinterpret_cast<char**>(  resultPtr[1]);
  *result1Ptr = n;
  *result2Ptr = p;
}

// Float table
//
// The constant powers of 10 here represent pure fractions
// with a binary point at the far left. (Each number in
// this first table is implicitly divided by 2^64.)
//
// Table size: 320 bytes
//
// A 64-bit significand allows us to exactly represent
// powers of 10 up to 10^27.  For larger powers, the
// value here is rounded DOWN from the exact value.
// For those powers, the value here is less than the
// exact power of 10; adding one gives a value greater
// than the exact power of 10.
//
// For single-precision Float, we use these directly
// for positive powers of 10.  For negative powers of
// ten, we multiply a value here by 10^-40.
//
// For Double and Float80, we use the 28 exact values
// here to help reduce the size of those tables.
static const uint64_t powersOf10_Float[40] = {
    0x8000000000000000, // x 2^1 == 10^0 exactly
    0xa000000000000000, // x 2^4 == 10^1 exactly
    0xc800000000000000, // x 2^7 == 10^2 exactly
    0xfa00000000000000, // x 2^10 == 10^3 exactly
    0x9c40000000000000, // x 2^14 == 10^4 exactly
    0xc350000000000000, // x 2^17 == 10^5 exactly
    0xf424000000000000, // x 2^20 == 10^6 exactly
    0x9896800000000000, // x 2^24 == 10^7 exactly
    0xbebc200000000000, // x 2^27 == 10^8 exactly
    0xee6b280000000000, // x 2^30 == 10^9 exactly
    0x9502f90000000000, // x 2^34 == 10^10 exactly
    0xba43b74000000000, // x 2^37 == 10^11 exactly
    0xe8d4a51000000000, // x 2^40 == 10^12 exactly
    0x9184e72a00000000, // x 2^44 == 10^13 exactly
    0xb5e620f480000000, // x 2^47 == 10^14 exactly
    0xe35fa931a0000000, // x 2^50 == 10^15 exactly
    0x8e1bc9bf04000000, // x 2^54 == 10^16 exactly
    0xb1a2bc2ec5000000, // x 2^57 == 10^17 exactly
    0xde0b6b3a76400000, // x 2^60 == 10^18 exactly
    0x8ac7230489e80000, // x 2^64 == 10^19 exactly
    0xad78ebc5ac620000, // x 2^67 == 10^20 exactly
    0xd8d726b7177a8000, // x 2^70 == 10^21 exactly
    0x878678326eac9000, // x 2^74 == 10^22 exactly
    0xa968163f0a57b400, // x 2^77 == 10^23 exactly
    0xd3c21bcecceda100, // x 2^80 == 10^24 exactly
    0x84595161401484a0, // x 2^84 == 10^25 exactly
    0xa56fa5b99019a5c8, // x 2^87 == 10^26 exactly
    0xcecb8f27f4200f3a, // x 2^90 == 10^27 exactly
    0x813f3978f8940984, // x 2^94 ~= 10^28
    0xa18f07d736b90be5, // x 2^97 ~= 10^29
    0xc9f2c9cd04674ede, // x 2^100 ~= 10^30
    0xfc6f7c4045812296, // x 2^103 ~= 10^31
    0x9dc5ada82b70b59d, // x 2^107 ~= 10^32
    0xc5371912364ce305, // x 2^110 ~= 10^33
    0xf684df56c3e01bc6, // x 2^113 ~= 10^34
    0x9a130b963a6c115c, // x 2^117 ~= 10^35
    0xc097ce7bc90715b3, // x 2^120 ~= 10^36
    0xf0bdc21abb48db20, // x 2^123 ~= 10^37
    0x96769950b50d88f4, // x 2^127 ~= 10^38
    0xbc143fa4e250eb31, // x 2^130 ~= 10^39
};

//
// ------------  Arithmetic helpers ----------------
//

// The core algorithm relies heavily on fraction and fixed-point
// arithmetic with 64-bit, 128-bit, and 192-bit integer values. (For
// float, double, and float80, respectively.) They also need precise
// control over all rounding.
//
// Note that most arithmetic operations are the same for integers and
// fractions, so we can just use the normal integer operations in most
// places.  Multiplication however, is different for fixed-size
// fractions.  Integer multiplication preserves the low-order part and
// discards the high-order part (ignoring overflow).  Fraction
// multiplication preserves the high-order part and discards the
// low-order part (rounding).  So most of the arithmetic helpers here
// are for multiplication.

// Note: With 64-bit GCC and Clang, we get a noticable performance
// gain by using `__uint128_t`.  Otherwise, we have to break things
// down into 32-bit chunks so we don't overflow 64-bit temporaries.

// Multiply a 64-bit fraction by a 32-bit fraction, rounding down.
static uint64_t multiply64x32RoundingDown(uint64_t lhs, uint32_t rhs) {
    static const uint64_t mask32 = UINT32_MAX;
    uint64_t t = ((lhs & mask32) * rhs) >> 32;
    return t + (lhs >> 32) * rhs;
}

// Multiply a 64-bit fraction by a 32-bit fraction, rounding up.
static uint64_t multiply64x32RoundingUp(uint64_t lhs, uint32_t rhs) {
    static const uint64_t mask32 = UINT32_MAX;
    uint64_t t = (((lhs & mask32) * rhs) + mask32) >> 32;
    return t + (lhs >> 32) * rhs;
}

// Multiply a 64-bit fraction by a 64-bit fraction, rounding down.
static uint64_t multiply64x64RoundingDown(uint64_t lhs, uint64_t rhs) {
#if HAVE_UINT128_T
    __uint128_t full = (__uint128_t)lhs * rhs;
    return (uint64_t)(full >> 64);
#else
    static const uint64_t mask32 = UINT32_MAX;
    uint64_t t = (lhs & mask32) * (rhs & mask32);
    t >>= 32;
    uint64_t a = (lhs >> 32) * (rhs & mask32);
    uint64_t b = (lhs & mask32) * (rhs >> 32);
    // Useful: If w,x,y,z are all 32-bit values, then:
    // w * x + y + z
    //   <= (2^64 - 2^33 + 1) + (2^32 - 1) + (2^32 - 1)
    //   <= 2^64 - 1
    //
    // That is, a product of two 32-bit values plus two more 32-bit
    // values can't overflow 64 bits.  (But "three more" can, so be
    // careful!)
    //
    // Here: t + a + (b & mask32) <= 2^64 - 1
    t += a + (b & mask32);
    t >>= 32;
    t += (b >> 32);
    return t + (lhs >> 32) * (rhs >> 32);
#endif
}

// Multiply a 64-bit fraction by a 64-bit fraction, rounding up.
static uint64_t multiply64x64RoundingUp(uint64_t lhs, uint64_t rhs) {
#if HAVE_UINT128_T
    static const __uint128_t roundingUpBias = ((__uint128_t)1 << 64) - 1;
    __uint128_t full = (__uint128_t)lhs * rhs;
    return (uint64_t)((full + roundingUpBias) >> 64);
#else
    static const uint64_t mask32 = UINT32_MAX;
    uint64_t t = (lhs & mask32) * (rhs & mask32);
    t = (t + mask32) >> 32;
    uint64_t a = (lhs >> 32) * (rhs & mask32);
    uint64_t b = (lhs & mask32) * (rhs >> 32);
    t += (a & mask32) + (b & mask32) + mask32;
    t >>= 32;
    t += (a >> 32) + (b >> 32);
    return t + (lhs >> 32) * (rhs >> 32);
#endif
}

// The power-of-10 tables do not directly store the associated binary
// exponent.  That's because the binary exponent is a simple linear
// function of the decimal power (and vice versa), so it's just as
// fast (and uses much less memory) to compute it:

// The binary exponent corresponding to a particular power of 10.
// This matches the power-of-10 tables across the full range of Float80.
static int binaryExponentFor10ToThe(int p) {
    return (int)(((((int64_t)p) * 55732705) >> 24) + 1);
}

// A decimal exponent that approximates a particular binary power.
static int decimalExponentFor2ToThe(int e) {
    return (int)(((int64_t)e * 20201781) >> 26);
}

// Given a power `p`, this returns three values:
// * 64-bit fractions `lower` and `upper`
// * integer `exponent`
//
// The returned values satisty the following:
// ```
//    lower * 2^exponent <= 10^p <= upper * 2^exponent
// ```
//
// Note: Max(*upper - *lower) = 3
static void intervalContainingPowerOf10_Float(int p, uint64_t *lower, uint64_t *upper, int *exponent) {
    if (p < 0) {
        uint64_t base = powersOf10_Float[p + 40];
        int baseExponent = binaryExponentFor10ToThe(p + 40);
        uint64_t tenToTheMinus40 = 0x8b61313bbabce2c6; // x 2^-132 ~= 10^-40
        *upper = multiply64x64RoundingUp(base + 1, tenToTheMinus40 + 1);
        *lower = multiply64x64RoundingDown(base, tenToTheMinus40);
        *exponent = baseExponent - 132;
    } else {
        uint64_t base = powersOf10_Float[p];
        *upper = base + 1;
        *lower = base;
        *exponent = binaryExponentFor10ToThe(p);
    }
}

// Format into decimal form: "123456789000.0", "1234.5678", "0.0000001234"
// Returns number of bytes of `dest` actually used, or zero if
// provided buffer is too small.
size_t swift_format_decimal(char *dest, size_t length,
    bool negative, const int8_t *digits, int digit_count, int exponent)
{
    // Largest buffer we could possibly need:
    size_t maximum_size =
        digit_count // All the digits
        + (exponent > 0 ? exponent : -exponent) // Max # of extra zeros
        + 4; // Max # of other items
    if (length < maximum_size) {
        // We only do the detailed check if the size is borderline.
        if (exponent <= 0) { // "0.0000001234"
            size_t actual_size =
                (negative ? 1 : 0) // Leading minus
                + 2 // Leading "0."
                + (-exponent) // Leading zeros after decimal point
                + digit_count
                + 1; // Trailing zero byte
            if (length < actual_size) {
                if (length > 0) {
                    dest[0] = 0;
                }
                return 0;
            }
        } else if (exponent < digit_count) { // "123.45"
            size_t actual_size =
                (negative ? 1 : 0) // Leading minus
                + digit_count
                + 1 // embedded decimal point
                + 1; // Trailing zero byte
            if (length < actual_size) {
                if (length > 0) {
                    dest[0] = 0;
                }
                return 0;
            }
        } else { // "12345000.0"
            size_t actual_size =
                (negative ? 1 : 0) // Leading minus
                + digit_count
                + (exponent - digit_count) // trailing zeros
                + 2 // ".0" to mark this as floating point
                + 1; // Trailing zero byte
            if (length < actual_size) {
                if (length > 0) {
                    dest[0] = 0;
                }
                return 0;
            }
        }
    }

    char *p = dest;
    if (negative) {
        *p++ = '-';
    }

    if (exponent <= 0) {
        *p++ = '0';
        *p++ = '.';
        while (exponent < 0) {
            *p++ = '0';
            exponent += 1;
        }
        for (int i = 0; i < digit_count; ++i) {
            *p++ = digits[i] + '0';
        }
    } else if (exponent < digit_count) {
        for (int i = 0; i < digit_count; ++i) {
            if (exponent == 0) {
                *p++ = '.';
            }
            *p++ = digits[i] + '0';
            exponent -= 1;
        }
    } else {
        for (int i = 0; i < digit_count; ++i) {
            *p++ = digits[i] + '0';
            exponent -= 1;
        }
        while (exponent > 0) {
            *p++ = '0';
            exponent -= 1;
        }
        *p++ = '.';
        *p++ = '0';
    }
    *p = '\0';
    return p - dest;
}

static size_t swift_format_constant(char *dest, size_t length, const char *s) {
    const size_t l = strlen(s);
    if (length <= l) {
        return 0;
    }
    strcpy(dest, s);
    return l;
}

static uint32_t bitPatternForFloat(float f) {
    union { float f; uint32_t u; } converter;
    converter.f = f;
    return converter.u;
}

int swift_decompose_float(float f,
    int8_t *digits, size_t digits_length, int *decimalExponent)
{
    static const int significandBitCount = FLT_MANT_DIG - 1;
    static const uint32_t significandMask
        = ((uint32_t)1 << significandBitCount) - 1;
    static const int exponentBitCount = 8;
    static const int exponentMask = (1 << exponentBitCount) - 1;
    // See comments in swift_decompose_double
    static const int64_t exponentBias = (1 << (exponentBitCount - 1)) - 2; // 125

    // Step 0: Deconstruct the target number
    // Note: this strongly assumes IEEE 754 binary32 format
    uint32_t raw = bitPatternForFloat(f);
    int exponentBitPattern = (raw >> significandBitCount) & exponentMask;
    uint32_t significandBitPattern = raw & significandMask;

    // Step 1: Handle the various input cases:
    int binaryExponent;
    uint32_t significand;
    if (digits_length < 9) {
        // Ensure we have space for 9 digits
        return 0;
    } else if (exponentBitPattern == exponentMask) { // NaN or Infinity
        // Return no digits
        return 0;
    } else if (exponentBitPattern == 0) {
        if (significandBitPattern == 0) { // Zero
            // Return one zero digit and decimalExponent = 0.
            digits[0] = 0;
            *decimalExponent = 0;
            return 1;
        } else { // Subnormal
            binaryExponent = 1 - exponentBias;
            significand = significandBitPattern << (32 - significandBitCount - 1);
        }
    } else { // normal
        binaryExponent = exponentBitPattern - exponentBias;
        uint32_t hiddenBit = (uint32_t)1 << (uint32_t)significandBitCount;
        uint32_t fullSignificand = significandBitPattern + hiddenBit;
        significand = fullSignificand << (32 - significandBitCount - 1);
    }

    // Step 2: Determine the exact unscaled target interval
    static const uint32_t halfUlp = (uint32_t)1 << (32 - significandBitCount - 2);
    uint32_t upperMidpointExact = significand + halfUlp;

    int isBoundary = significandBitPattern == 0;
    static const uint32_t quarterUlp = halfUlp >> 1;
    uint32_t lowerMidpointExact
        = significand - (isBoundary ? quarterUlp : halfUlp);

    // Step 3: Estimate the base 10 exponent
    int base10Exponent = decimalExponentFor2ToThe(binaryExponent);

    // Step 4: Compute a power-of-10 scale factor
    uint64_t powerOfTenRoundedDown = 0;
    uint64_t powerOfTenRoundedUp = 0;
    int powerOfTenExponent = 0;
    intervalContainingPowerOf10_Float(-base10Exponent,
                                      &powerOfTenRoundedDown,
                                      &powerOfTenRoundedUp,
                                      &powerOfTenExponent);
    const int extraBits = binaryExponent + powerOfTenExponent;

    // Step 5: Scale the interval (with rounding)
    static const int integerBits = 5;
    const int shift = integerBits - extraBits;
    const int roundUpBias = (1 << shift) - 1;
    static const int fractionBits = 64 - integerBits;
    uint64_t u, l;
    if (significandBitPattern & 1) {
        // Narrow the interval (odd significand)
        uint64_t u1 = multiply64x32RoundingDown(powerOfTenRoundedDown,
                                                upperMidpointExact);
        u = u1 >> shift; // Rounding down

        uint64_t l1 = multiply64x32RoundingUp(powerOfTenRoundedUp,
                                              lowerMidpointExact);
        l = (l1 + roundUpBias) >> shift; // Rounding Up
    } else {
        // Widen the interval (even significand)
        uint64_t u1 = multiply64x32RoundingUp(powerOfTenRoundedUp,
                                              upperMidpointExact);
        u = (u1 + roundUpBias) >> shift; // Rounding Up

        uint64_t l1 = multiply64x32RoundingDown(powerOfTenRoundedDown,
                                                lowerMidpointExact);
        l = l1 >> shift; // Rounding down
    }

    // Step 6: Align first digit, adjust exponent
    // In particular, this prunes leading zeros from subnormals
    static const uint64_t fixedPointOne = (uint64_t)1 << fractionBits;
    static const uint64_t fixedPointMask = fixedPointOne - 1;
    uint64_t t = u;
    uint64_t delta = u - l;
    int exponent = base10Exponent + 1;

    while (t < fixedPointOne) {
        exponent -= 1;
        delta *= 10;
        t *= 10;
    }

    // Step 7: Generate digits
    int8_t *digit_p = digits;
    int nextDigit = (int)(t >> fractionBits);
    t &= fixedPointMask;

    // Generate one digit at a time...
    while (t > delta) {
        *digit_p++ = nextDigit;
        delta *= 10;
        t *= 10;
        nextDigit = (int)(t >> fractionBits);
        t &= fixedPointMask;
    }

    // Adjust the final digit to be closer to the original value
    if (delta > t + fixedPointOne) {
        uint64_t skew;
        if (isBoundary) {
            skew = delta - delta / 3 - t;
        } else {
            skew = delta / 2 - t;
        }
        uint64_t one = (uint64_t)(1) << (64 - integerBits);
        uint64_t lastAccurateBit = 1ULL << 24;
        uint64_t fractionMask = (one - 1) & ~(lastAccurateBit - 1);
        uint64_t oneHalf = one >> 1;
        if (((skew + (lastAccurateBit >> 1)) & fractionMask) == oneHalf) {
            // If the skew is exactly integer + 1/2, round the last
            // digit even after adjustment
            int adjust = (int)(skew >> (64 - integerBits));
            nextDigit = (nextDigit - adjust) & ~1;
        } else {
            // Else round to nearest...
            int adjust = (int)((skew + oneHalf) >> (64 - integerBits));
            nextDigit = (nextDigit - adjust);
        }
    }
    *digit_p++ = nextDigit;

    *decimalExponent = exponent;
    return digit_p - digits;
}

size_t swift_format_float(float d, char *dest, size_t length)
{
    if (!isfinite(d)) {
        if (isinf(d)) {
            // Infinity
            if (signbit(d)) {
                return swift_format_constant(dest, length, "-inf");
            } else {
                return swift_format_constant(dest, length, "inf");
            }
        } else {
            // NaN
            static const int significandBitCount = 23;
            uint32_t raw = bitPatternForFloat(d);
            const char *sign = signbit(d) ? "-" : "";
            const char *signaling = ((raw >> (significandBitCount - 1)) & 1) ? "" : "s";
            uint32_t payload = raw & ((1L << (significandBitCount - 2)) - 1);
            char buff[32];
            if (payload != 0) {
                snprintf(buff, sizeof(buff), "%s%snan(0x%x)",
                         sign, signaling, payload);
            } else {
                snprintf(buff, sizeof(buff), "%s%snan",
                         sign, signaling);
            }
            return swift_format_constant(dest, length, buff);
        }
    }

    // zero
    if (d == 0.0) {
        if (signbit(d)) {
            return swift_format_constant(dest, length, "-0.0");
        } else {
            return swift_format_constant(dest, length, "0.0");
        }
    }

    // Decimal numeric formatting
    int decimalExponent;
    int8_t digits[9];
    int digitCount =
        swift_decompose_float(d, digits, sizeof(digits), &decimalExponent);
    // People use float to model integers <= 2^24, so we use that
    // as a cutoff for decimal vs. exponential format.
    if (decimalExponent < -3 || fabsf(d) > 0x1.0p24F) {
        return swift_format_decimal(dest, length, signbit(d),
                 digits, digitCount, decimalExponent);
/*
        return swift_format_exponential(dest, length, signbit(d),
                 digits, digitCount, decimalExponent);
*/
    } else {
        return swift_format_decimal(dest, length, signbit(d),
                 digits, digitCount, decimalExponent);
    }
}

void showFloat32(char **resultPtr, float x) {
  auto p = reinterpret_cast<char*>(malloc_dex(100));
  // printf("Hello\n");
  // auto n = swift_format_float(x, p, 100);
  auto n = snprintf(p, 100, "%.9g", x);
  // auto n = sprintf(p, "%." STRING(FLT_DECIMAL_DIG) "g", x);
  auto result1Ptr = reinterpret_cast<int32_t*>(resultPtr[0]);
  auto result2Ptr = reinterpret_cast<char**>(  resultPtr[1]);
  *result1Ptr = n;
  *result2Ptr = p;
}

void showFloat64(char **resultPtr, double x) {
  auto p = reinterpret_cast<char*>(malloc_dex(100));
  // auto n = sprintf(p, "%.16g", x);
  // auto n = sprintf(p, "%." STRING(DBL_DECIMAL_DIG) "g", x);
  // auto n = sprintf(p, "%." "17" "g", x);
  auto n = snprintf(p, 100, "%.17g", x);
  auto result1Ptr = reinterpret_cast<int32_t*>(resultPtr[0]);
  auto result2Ptr = reinterpret_cast<char**>(  resultPtr[1]);
  *result1Ptr = n;
  *result2Ptr = p;
}

#ifdef DEX_CUDA

} // extern "C"

template<typename ...Args>
using driver_func = CUresult(*)(Args...);

template<typename ...Args1, typename ...Args2>
void dex_check(const char* fname, driver_func<Args1...> f, Args2... args) {
  auto result = f(args...);
  if (result != 0) {
    const char* error_name = nullptr;
    const char* error_msg = nullptr;
    cuGetErrorName(result, &error_name);
    cuGetErrorString(result, &error_msg);
    if (!error_name) error_name = "unknown error";
    if (!error_msg) error_msg = "Unknown error";
    printf("CUDA driver error at %s (%s): %s\n", fname, error_name, error_msg);
    std::abort();
  }
}

#define CHECK(f, ...) dex_check(#f, f, __VA_ARGS__)

extern "C" {

void load_cuda_array(void* host_ptr, void* device_ptr, int64_t bytes) {
  CHECK(cuMemcpyDtoH, host_ptr, reinterpret_cast<CUdeviceptr>(device_ptr), bytes);
}

void dex_cuMemcpyDtoH(int64_t bytes, char* device_ptr, char* host_ptr) {
  CHECK(cuMemcpyDtoH, host_ptr, reinterpret_cast<CUdeviceptr>(device_ptr), bytes);
}

void dex_cuMemcpyHtoD(int64_t bytes, char* device_ptr, char* host_ptr) {
  CHECK(cuMemcpyHtoD, reinterpret_cast<CUdeviceptr>(device_ptr), host_ptr, bytes);
}

void dex_queryParallelismCUDA(const char* kernel_func, int64_t iters,
                              int32_t* numWorkgroups, int32_t* workgroupSize) {
  if (iters == 0) {
    *numWorkgroups = 0;
    *workgroupSize = 0;
    return;
  }
  // TODO: Use the occupancy calculator, or at least use a fixed number of blocks?
  const int64_t fixedWgSize = 1024;
  *workgroupSize = fixedWgSize;
  *numWorkgroups = std::min((iters + fixedWgSize - 1) / fixedWgSize, fixedWgSize);
}

void dex_loadKernelCUDA(const char* kernel_text, char** module_storage, char** kernel_storage) {
  if (*kernel_storage) { return; }
  CUmodule *module = reinterpret_cast<CUmodule*>(module_storage);
  CHECK(cuModuleLoadData, module, kernel_text);
  CUfunction *kernel = reinterpret_cast<CUfunction*>(kernel_storage);
  CHECK(cuModuleGetFunction, kernel, *module, "kernel");
}

void dex_unloadKernelCUDA(char** module_storage, char** kernel_storage) {
  CUmodule *module = reinterpret_cast<CUmodule*>(module_storage);
  CUfunction *kernel = reinterpret_cast<CUfunction*>(kernel_storage);
  CHECK(cuModuleUnload, *module);
  *module = nullptr;
  *kernel = nullptr;
}

void dex_cuLaunchKernel(char* kernel_func, int64_t iters, char** args) {
  if (iters == 0) return;
  CUfunction kernel = reinterpret_cast<CUfunction>(kernel_func);
  int32_t block_dim_x, grid_dim_x;
  dex_queryParallelismCUDA(kernel_func, iters, &grid_dim_x, &block_dim_x);
  CHECK(cuLaunchKernel, kernel,
                        grid_dim_x, 1, 1,               // grid size
                        block_dim_x, 1, 1,              // block size
                        0,                              // shared memory
                        CU_STREAM_LEGACY,               // stream
                        reinterpret_cast<void**>(args), // kernel arguments
                        nullptr);
}

char* dex_cuMemAlloc(int64_t size) {
  if (size == 0) return nullptr;
  CUdeviceptr ptr;
  CHECK(cuMemAlloc, &ptr, size);
  return reinterpret_cast<char*>(ptr);
}

void dex_cuMemFree(char* ptr) {
  if (!ptr) return;
  CHECK(cuMemFree, reinterpret_cast<CUdeviceptr>(ptr));
}

void dex_synchronizeCUDA() {
  CHECK(cuStreamSynchronize, CU_STREAM_LEGACY);
}

void dex_ensure_has_cuda_context() {
  CHECK(cuInit, 0);
  CUcontext ctx;
  CHECK(cuCtxGetCurrent, &ctx);
  if (!ctx) {
    CUdevice dev;
    CHECK(cuDeviceGet, &dev, 0);
    CHECK(cuDevicePrimaryCtxRetain, &ctx, dev);
    CHECK(cuCtxPushCurrent, ctx);
  }
}

#undef CHECK

#endif // DEX_CUDA

int32_t dex_queryParallelismMC(int64_t iters) {
  int32_t nthreads = std::thread::hardware_concurrency();
  if (iters < nthreads) {
    nthreads = iters;
  }
  return nthreads;
}

void dex_launchKernelMC(char *function_ptr, int64_t size, char **args) {
  auto function = reinterpret_cast<void (*)(int32_t, int32_t, char**)>(function_ptr);
  int32_t nthreads = dex_queryParallelismMC(size);
  std::vector<std::thread> threads(nthreads);
  for (int32_t tid = 0; tid < nthreads; ++tid) {
    threads[tid] = std::thread([function, args, tid, nthreads]() {
      function(tid, nthreads, args);
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

} // end extern "C"
