/******************************************************************************
 *
 * @brief   Approximate AdAM multiplier (Mitchell logarithmic multiplication
 *          with adaptive, fault-tolerant MSB protection).
 * @author  Michael Markov
 *
 *******************************************************************************/
#ifndef ADAMMULTIPLIER_HPP
#define ADAMMULTIPLIER_HPP

#include <ap_int.h>
#include <hls_stream.h>

class ap_resource_adam {};

namespace adam_hls {

// Returns absolute value for x
template <typename T>
static inline T abs_value(T x) {
#pragma HLS inline
	return (x < 0) ? T(-x) : x;
}

// Returns MSB for x
template <typename T>
static inline unsigned leading_one_position(T x) {
#pragma HLS inline
	for (int i = T::width - 1; i >= 0; --i) {
#pragma HLS unroll
		if (x[i]) {
			return static_cast<unsigned>(i);
		}
	}
	return 0;
}



// ---------------------------------------------------------------------------
// Mitchell logarithmic-multiplication helpers
// ---------------------------------------------------------------------------

// Compute f = (x mod 2^lod) / 2^lod to normalize x into a fixed-point Mitchell mantissa of "frac_width" bits
// x: Operand to normalize
// lod (leading-one-detector position): Index of x's MSB
// frac_width: Precision for representing fractions throughout
// Example math: f=(11%2^3)/2^3=3/8=0.375
// Example real world: x=1011, lod=3, frac_width=2 -> below=1011&0111=0011 -> f=0011>>(3-2)=0001 -> due to frac_width read as 0.01 in binary -> 0*(1/2)+1*(1/4)=0.25 -> approximately 0.375
template <typename T>
static inline unsigned long long mitchell_fraction(T x, unsigned lod, unsigned frac_width) {
#pragma HLS inline
	if (lod == 0) {
		// No bits below the leading one: fraction is exactly 0.y
		return 0ULL;
	}
	// Create mask from everything below lod-1 downto 0 and mask x
	unsigned long long below = static_cast<unsigned long long>(x) & ((1ULL << lod) - 1ULL); // below = x mod 2^lod / 2^lod

	// Resizes "below" from its natural lod-bit width to a frac_width-bit field
	if (lod >= frac_width) {
		// More mantissa bits available than we keep: drop the extra LSBs.
		return below >> (lod - frac_width);
	} else {
		// Fewer mantissa bits than the field width: left-align, zero-pad LSBs.
		return below << (frac_width - lod);
	}
}

// The smaller the combined result, the more mantissa
// bits are considered significant enough to protect/redundantly compute.
// max_lod: max(lod_a, lod_b)
// mantissa_width: width T of the (already truncated) mantissa field
static inline unsigned protection_schedule(unsigned max_lod, unsigned mantissa_width) {
#pragma HLS inline
	unsigned protect;
	if (max_lod >= 5) {
		protect = 2;
	} else if (max_lod == 4) {
		protect = 3;
	} else { // max_lod <= 3: comprehensive protection
		protect = mantissa_width;
	}
	return (protect > mantissa_width) ? mantissa_width : protect;
}

// Function adds a width amount of bits from x and y with carry-over. Returns a sum of x and y with the length of width
// x: summand 1
// y: summand 2
// width: Tells how many bits to process
// carry_in: initial carry fed into bit 0
static inline unsigned long long ripple_add(unsigned long long x, unsigned long long y, unsigned width, bool carry_in) {
#pragma HLS inline off // Keep it its own hardware block
	unsigned long long result = 0;
	bool carry = carry_in;

	// Sum bit by bit until width is reached
	for (unsigned i = 0; i < width; ++i) {
#pragma HLS unroll // Do not implement as loop in hardware
		bool bx = (x >> i) & 1ULL;
		bool by = (y >> i) & 1ULL;
		bool sum_bit = bx ^ by ^ carry;
		carry = (bx && by) || (bx && carry) || (by && carry);
		result |= (static_cast<unsigned long long>(sum_bit) << i);
	}
	return result;
}

// Bitwise majority vote across three replicas
static inline unsigned long long majority_vote(unsigned long long r0, unsigned long long r1,
                                                unsigned long long r2, unsigned width) {
#pragma HLS inline
	unsigned long long result = 0;
	for (unsigned i = 0; i < width; ++i) {
#pragma HLS unroll
		bool b0 = (r0 >> i) & 1ULL;
		bool b1 = (r1 >> i) & 1ULL;
		bool b2 = (r2 >> i) & 1ULL;
		bool maj = (b0 && b1) || (b1 && b2) || (b0 && b2);
		result |= (static_cast<unsigned long long>(maj) << i);
	}
	return result;
}

// TMR-protected addition of the two k (LOD/exponent) values, with an
// incoming carry from the mantissa addition. Three independent adds + majority vote,
// mirroring the fully-triplicated k-adder
static inline unsigned long long tmr_k_add(unsigned long long lod_a, unsigned long long lod_b,
                                            unsigned width, bool carry_in) {
#pragma HLS inline
	unsigned long long rep0 = ripple_add(lod_a, lod_b, width, carry_in);
	unsigned long long rep1 = ripple_add(lod_a, lod_b, width, carry_in);
	unsigned long long rep2 = ripple_add(lod_a, lod_b, width, carry_in);
	return majority_vote(rep0, rep1, rep2, width);
}

// ---------------------------------------------------------------------------
// AdAM Multiplier function
// ---------------------------------------------------------------------------

// Function replaces standard a*b with an approximate computation which turns multiplication into addition
// at the cost of some accuracy, while adding an adaptive, LOD-driven partial-redundancy check + zero-on-mismatch
// mitigation over the mantissa's protected MSBs and full TMR + majority voting over the k (exponent) adder
// log2(a*b)=log2(a)+log2(b)
template <typename TA, typename TB>
static inline auto adam_multiplier(TA a, TB b) -> decltype(a * b) {
#pragma HLS inline
	typedef decltype(a * b) product_t; // Widen bit width of return type

	// Return 0
	if ((a == 0) || (b == 0)) {
		return product_t(0);
	}

	// Determine result polarity
	const bool neg_a = (a < 0);
	const bool neg_b = (b < 0);
	const bool result_negative = (neg_a != neg_b);

	// Ensure result is less than 64-bit and determine
	constexpr unsigned WA = TA::width;
	constexpr unsigned WB = TB::width;
	constexpr unsigned WMAX = (WA > WB) ? WA : WB; // Determine highest width
	constexpr unsigned F = (WMAX > 1) ? (WMAX - 1) : 1; // Mitchell mantissa width
	constexpr unsigned DROP = (F > 2) ? 2 : (F > 1 ? 1 : 0);
	constexpr unsigned T = (F > DROP) ? (F - DROP) : 1; // Truncated mantissa width used by the adaptive adder
	constexpr unsigned K_WIDTH = WA + WB; // k-adder width
	static_assert(T + K_WIDTH < 63,
		"adam_multiplier: operand widths too large for the 64-bit logarithmic-"
		"domain scratch arithmetic used here; widen the scratch type if you "
		"need bigger operands.");

	// Determine absolute values for summands
	TA abs_a = abs_value(a);
	TB abs_b = abs_value(b);

	// Determine the MSB for summands
	unsigned lod_a = leading_one_position(abs_a);
	unsigned lod_b = leading_one_position(abs_b);
	unsigned max_lod = (lod_a > lod_b) ? lod_a : lod_b; // k value of the biggest operand

	unsigned long long frac_a = mitchell_fraction(abs_a, lod_a, T);
	unsigned long long frac_b = mitchell_fraction(abs_b, lod_b, T);

	// Path A: T-bit add with carry-out
	unsigned long long frac_mask = (1ULL << T) - 1ULL;
	unsigned long long frac_sum_full = ripple_add(frac_a, frac_b, T + 1, false);
	bool carry_into_k = (frac_sum_full >> T) & 1ULL; // Mitchell mantissa overflow -> bump exponent
	unsigned long long frac_primary = frac_sum_full & frac_mask;

	// Adaptive protection: only the MSBs of mantissa are redundantly computed/compared (protected)
	// The remaining (unprotected) LSBs are accepted from Path A as-is
	unsigned protected_bits = protection_schedule(max_lod, T);
	unsigned unprotected_bits = T - protected_bits;
	unsigned long long secondary_mask = (protected_bits >= 64) ? ~0ULL : ((1ULL << protected_bits) - 1ULL);
	unsigned long long protect_mask = secondary_mask << unprotected_bits; // Mask of protected bits

	// Determine protected fraction bits
	unsigned long long top_primary = frac_primary & protect_mask;

	// Path B: narrow redundant addition over only the protected mantissa region
	// Only the MSB portion is duplicated, not the full sum
	unsigned long long protected_frac_a = frac_a >> unprotected_bits;
	unsigned long long protected_frac_b = frac_b >> unprotected_bits;

	// Determine unprotected fraction
	unsigned long long unprotected_frac_a = frac_a & ((1ULL << unprotected_bits) - 1ULL);
	unsigned long long unprotected_frac_b = frac_b & ((1ULL << unprotected_bits) - 1ULL);
	
	// Determine the carry that ripples from the unprotected LSBs into the protected region
	unsigned long long unprotected_sum = ripple_add(unprotected_frac_a, unprotected_frac_b, unprotected_bits + 1, false);
	bool carry_into_protected = (unprotected_sum >> unprotected_bits) & 1ULL;
	
	// Calculate secondary top sum
	// +1 guard bit to observe carry-out of the protected region itself
	unsigned long long secondary_top_sum = ripple_add(protected_frac_a, protected_frac_b, protected_bits + 1, carry_into_protected);
	unsigned long long top_secondary = (secondary_top_sum & secondary_mask) << unprotected_bits;

	// Default to Path A
	unsigned long long frac_corrected = frac_primary;

	// Compare 2 different path by only the protected fraction bits
	if (top_primary != top_secondary) { // Fault detected in the protected MSBs
		// Carry over agreeing bits and zero disagreeing bits rather than trusting either path blindly
		unsigned long long disagreement = top_primary ^ top_secondary;
		frac_corrected = frac_primary & ~disagreement;
	}

	// TMR-protected, independent of mantissa
	unsigned long long k_sum = tmr_k_add(lod_a, lod_b, K_WIDTH, carry_into_k);
	unsigned sum_int = static_cast<unsigned>(k_sum);

	// Anti-log (delog) reconstruction: 2^sum_int * (1 + frac/2^T).
	product_t magnitude;
	if (sum_int >= T) {
		magnitude = (product_t(1) << sum_int) | product_t(frac_corrected << (sum_int - T));
	} else {
		magnitude = (product_t(1) << sum_int) | product_t(frac_corrected >> (T - sum_int));
	}

	return result_negative ? product_t(-magnitude) : magnitude;
}



// ---------------------------------------------------------------------------
// Wrapper functions
// ---------------------------------------------------------------------------

template <typename TC, typename TD>
auto mul(TC const &c, TD const &d, ap_resource_adam const&) -> decltype(c * d) {
#pragma HLS inline
	return adam_multiplier(c, d);
}

} // namespace adam_hls

using adam_hls::adam_multiplier;

#endif