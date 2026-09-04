#ifndef OPERATORS_H
#define OPERATORS_H

#include "types.h"


// Operator reversibility classes (spec §3)
//   Class A: always unitary (norm-preserving, invertible)
//   Class B: conditionally unitary (lossless iff no energy crosses +-N)
//   Class C: never unitary once active (discards or injects energy)


// Frequency shift boundary modes
#define SOC_SHIFT_TRUNCATE 0   // energy past +-N is discarded (lossy)
#define SOC_SHIFT_WRAP     1   // energy past +-N wraps around (lossless)


// Class A: always unitary


// R_theta: C_out[k] = C_in[k] * exp(i * theta)
// Diagonal, unit-modulus, invertible by phase_shift(-theta).
SocStatus soc_op_phase_shift(SocShape* shape, double theta);


// Class B: conditionally unitary


// H_m: C_out[k] = C_in[k - m]
// mode = SOC_SHIFT_TRUNCATE: energy past +-N is lost.
// mode = SOC_SHIFT_WRAP: energy wraps circularly, always unitary.
SocStatus soc_op_freq_shift(const SocShape* src, SocShape* dst,
                            int m, int mode);


// Class C: never unitary once active


// W: C_out[k] = w_k * C_in[k]
// weights: array of SOC_COEFF_LENGTH real values.
// Pure gain change if all weights equal, otherwise spectral filtering.
SocStatus soc_op_spectral_filter(SocShape* shape, const double* weights);

// Uniform gain: all w_k = gain. Convenience wrapper.
SocStatus soc_op_uniform_gain(SocShape* shape, double gain);

// Low-pass filter: keep |k| <= cutoff, zero otherwise.
SocStatus soc_op_lowpass(SocShape* shape, int cutoff);

// High-pass filter: keep |k| > cutoff, zero otherwise.
SocStatus soc_op_highpass(SocShape* shape, int cutoff);

// DC boost: scale only the k=0 term by boost factor.
// Test probe for the decoupled-term leak (spec §5).
SocStatus soc_op_dc_boost(SocShape* shape, double boost);

// Exponential frequency-dependent attenuation + optional additive noise.
// C_out[k] = C_in[k] * exp(-alpha * |k| * distance) + noise
SocStatus soc_op_attenuate(SocShape* shape, double distance,
                           double alpha, double noise_std);

// Power clamp: scale down uniformly if total energy exceeds p_max.
// Identity if energy <= p_max.
SocStatus soc_op_power_clamp(SocShape* shape, double p_max);

#endif
