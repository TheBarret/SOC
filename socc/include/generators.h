#ifndef SOC_GENERATORS_H
#define SOC_GENERATORS_H

#include "types.h"


// Shape generators: produce SocShape objects from geometric primitives.
//
// curve_to_coeffs is here because it is the bridge between spatial domain
// (z(t) samples) and the Fourier coefficient domain (SocShape).


// Transform: sampled curve -> coefficient vector.
//
// z_samples: array of n complex values, z(t) at equally spaced t in [0,1).
// shape: output shape with coefficients C_k for k = -N..N.
//
// Uses normalized DFT: C_k = (1/n) * sum_t z(t) * exp(-i 2pi k t).
// Coefficients beyond +-N are discarded (band-limiting).
SocStatus soc_curve_to_coeffs(const SocComplex* z_samples, int n,
                              SocShape* shape);

// Unit circle: z(t) = exp(i * 2pi * t)
SocStatus soc_gen_circle(SocShape* shape, int n_samples);

// Regular polygon with given number of sides, circumradius 1.
SocStatus soc_gen_polygon(SocShape* shape, int sides, int n_samples);

// Star shape with given number of points and inner radius ratio.
SocStatus soc_gen_star(SocShape* shape, int points, double inner_ratio,
                       int n_samples);

#endif
