#include "../include/types.h"
#include "../include/operators.h"

#include <math.h>
#include <stdlib.h>


// Class A: always unitary


SocStatus soc_op_phase_shift(SocShape* shape, double theta) {
    if (shape == NULL) {
        return SOC_ERR_NULL_POINTER;
    }
    SocComplex phase = cos(theta) + I * sin(theta);
    for (int i = 0; i < SOC_COEFF_LENGTH; i++) {
        shape->coeffs[i] *= phase;
    }
    return SOC_OK;
}


// Class B: conditionally unitary


SocStatus soc_op_freq_shift(const SocShape* src, SocShape* dst,
                            int m, int mode) {
    if (src == NULL || dst == NULL) {
        return SOC_ERR_NULL_POINTER;
    }
    if (mode != SOC_SHIFT_TRUNCATE && mode != SOC_SHIFT_WRAP) {
        return SOC_ERR_INVALID_MODE;
    }

    for (int idx = 0; idx < SOC_COEFF_LENGTH; idx++) {
        int k = idx - SOC_N;
        int src_k = k - m;

        if (mode == SOC_SHIFT_TRUNCATE) {
            if (src_k < -SOC_N || src_k > SOC_N) {
                dst->coeffs[idx] = 0.0 + 0.0 * I;
            } else {
                dst->coeffs[idx] = src->coeffs[src_k + SOC_N];
            }
        } else {  // SOC_SHIFT_WRAP
            src_k = ((src_k + SOC_N) % SOC_COEFF_LENGTH) - SOC_N;
            if (src_k < -SOC_N) src_k += SOC_COEFF_LENGTH;
            dst->coeffs[idx] = src->coeffs[src_k + SOC_N];
        }
    }
    return SOC_OK;
}


// Class C: never unitary once active


SocStatus soc_op_spectral_filter(SocShape* shape, const double* weights) {
    if (shape == NULL || weights == NULL) {
        return SOC_ERR_NULL_POINTER;
    }
    for (int i = 0; i < SOC_COEFF_LENGTH; i++) {
        shape->coeffs[i] *= weights[i];
    }
    return SOC_OK;
}

SocStatus soc_op_uniform_gain(SocShape* shape, double gain) {
    if (shape == NULL) {
        return SOC_ERR_NULL_POINTER;
    }
    for (int i = 0; i < SOC_COEFF_LENGTH; i++) {
        shape->coeffs[i] *= gain;
    }
    return SOC_OK;
}

SocStatus soc_op_lowpass(SocShape* shape, int cutoff) {
    if (shape == NULL) {
        return SOC_ERR_NULL_POINTER;
    }
    if (cutoff < 0 || cutoff > SOC_N) {
        return SOC_ERR_INVALID_INDEX;
    }
    for (int idx = 0; idx < SOC_COEFF_LENGTH; idx++) {
        int k = idx - SOC_N;
        if (abs(k) > cutoff) {
            shape->coeffs[idx] = 0.0 + 0.0 * I;
        }
    }
    return SOC_OK;
}

SocStatus soc_op_highpass(SocShape* shape, int cutoff) {
    if (shape == NULL) {
        return SOC_ERR_NULL_POINTER;
    }
    if (cutoff < 0 || cutoff > SOC_N) {
        return SOC_ERR_INVALID_INDEX;
    }
    for (int idx = 0; idx < SOC_COEFF_LENGTH; idx++) {
        int k = idx - SOC_N;
        if (abs(k) <= cutoff) {
            shape->coeffs[idx] = 0.0 + 0.0 * I;
        }
    }
    return SOC_OK;
}

SocStatus soc_op_dc_boost(SocShape* shape, double boost) {
    if (shape == NULL) {
        return SOC_ERR_NULL_POINTER;
    }
    shape->coeffs[SOC_N] *= boost;  // k=0 at flat index N
    return SOC_OK;
}

SocStatus soc_op_attenuate(SocShape* shape, double distance,
                           double alpha, double noise_std) {
    if (shape == NULL) {
        return SOC_ERR_NULL_POINTER;
    }
    if (distance < 0.0 || alpha < 0.0 || noise_std < 0.0) {
        return SOC_ERR_INVALID_INDEX;
    }

    for (int idx = 0; idx < SOC_COEFF_LENGTH; idx++) {
        int k = idx - SOC_N;
        double atten = exp(-alpha * fabs((double)k) * distance);
        shape->coeffs[idx] *= atten;

        if (noise_std > 0.0) {
            // Box-Muller for Gaussian noise
            double u1 = (double)rand() / RAND_MAX;
            double u2 = (double)rand() / RAND_MAX;
            if (u1 < 1e-12) u1 = 1e-12;
            double mag = sqrt(-2.0 * log(u1));
            double real = mag * cos(2.0 * SOC_PI * u2) * noise_std;
            double imag = mag * sin(2.0 * SOC_PI * u2) * noise_std;
            shape->coeffs[idx] += real + I * imag;
        }
    }
    return SOC_OK;
}

SocStatus soc_op_power_clamp(SocShape* shape, double p_max) {
    if (shape == NULL) {
        return SOC_ERR_NULL_POINTER;
    }
    if (p_max <= 0.0) {
        return SOC_ERR_INVALID_INDEX;
    }

    double p = 0.0;
    for (int i = 0; i < SOC_COEFF_LENGTH; i++) {
        double mag = cabs(shape->coeffs[i]);
        p += mag * mag;
    }

    if (p > p_max) {
        double scale = sqrt(p_max / p);
        for (int i = 0; i < SOC_COEFF_LENGTH; i++) {
            shape->coeffs[i] *= scale;
        }
    }
    return SOC_OK;
}
