/******************************************************************************
 *
 * @brief   Validation top-level module for adammultiplier component.
 * @author  Michael Markov
 *
 *******************************************************************************/
#include "adammultiplier.hpp"
#include <ap_int.h>
#include <iostream>
#include <cstdlib>

// Top-level function exposed for csim / synthesis sanity-checking.
ap_int<16> adam_multiplier_top(ap_int<8> a, ap_int<8> b) {
    return adam_hls::adam_multiplier(a, b);
}

#ifdef ADAM_MULTIPLIER_TESTBENCH
int main() {
    int mismatches = 0;
    int max_abs_error = 0;

    for (int a = -128; a < 128; ++a) {
        for (int b = -128; b < 128; ++b) {
            ap_int<8> ap_a = a;
            ap_int<8> ap_b = b;
            ap_int<16> approx = adam_multiplier_top(ap_a, ap_b);
            int exact = a * b;
            int error = std::abs(static_cast<int>(approx) - exact);
            max_abs_error = std::max(max_abs_error, error);

            // Sanity checks that should always hold regardless of approximation:
            if ((a == 0 || b == 0) && approx != 0) {
                std::cerr << "FAIL zero case: a=" << a << " b=" << b
                          << " got=" << approx << "\n";
                ++mismatches;
            }
            bool sign_ok = (exact >= 0) == (approx >= 0) || exact == 0 || approx == 0;
            if (!sign_ok) {
                std::cerr << "FAIL sign mismatch: a=" << a << " b=" << b
                          << " exact=" << exact << " approx=" << approx << "\n";
                ++mismatches;
            }
        }
    }

    std::cout << "Max abs error over exhaustive 8x8 sweep: " << max_abs_error << "\n";
    std::cout << "Mismatches (zero/sign sanity failures): " << mismatches << "\n";
    return mismatches == 0 ? 0 : 1;
}
#endif