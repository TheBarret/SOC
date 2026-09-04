#ifndef SHAPE_H
#define SHAPE_H

#include "types.h"


// Shape lifecycle


// Initialize a shape from a coefficient array.
// coeffs must point to SOC_COEFF_LENGTH contiguous SocComplex values.
// The array is copied into the struct.
SocStatus soc_shape_init(SocShape* shape, const SocComplex* coeffs);

// Initialize a shape with all coefficients set to zero.
SocStatus soc_shape_zero(SocShape* shape);

// Copy src into dst. Both must be valid shapes.
SocStatus soc_shape_copy(const SocShape* src, SocShape* dst);


// Coefficient access


// Get coefficient by harmonic index k (range -N..N).
// Returns SOC_ERR_INVALID_INDEX if k is out of range.
SocStatus soc_shape_get(const SocShape* shape, SocHarmonic k, SocComplex* out);

// Set coefficient by harmonic index k (range -N..N).
// Returns SOC_ERR_INVALID_INDEX if k is out of range.
SocStatus soc_shape_set(SocShape* shape, SocHarmonic k, SocComplex value);

// Get coefficient by flat index (range 0..2N).
SocStatus soc_shape_get_flat(const SocShape* shape, SocIndex idx, SocComplex* out);

// Set coefficient by flat index (range 0..2N).
SocStatus soc_shape_set_flat(SocShape* shape, SocIndex idx, SocComplex value);


// Derived quantities


// Total energy: sum |C_k|^2 over all k, including k=0.
double soc_shape_norm_full(const SocShape* shape);

// Shape energy: sum |C_k|^2 over k != 0.
double soc_shape_norm_shape(const SocShape* shape);

// Get the C0 term (position/bias).
SocComplex soc_shape_c0(const SocShape* shape);


// Reconstruction


// Reconstruct z(t) = sum_k C_k * exp(i * 2pi * k * t) at sample points t.
// t: array of sample positions in [0,1), length n.
// out: array of length n, filled with reconstructed complex values.
SocStatus soc_shape_reconstruct(const SocShape* shape,
                                const double* t,
                                int n,
                                SocComplex* out);


// Indexing helpers


// Map harmonic index k (-N..N) to flat index (0..2N).
// Returns -1 if k is out of range.
int soc_k_to_index(SocHarmonic k);

// Map flat index (0..2N) to harmonic index k (-N..N).
// Returns 0 for invalid index (caller should validate range separately).
SocHarmonic soc_index_to_k(SocIndex idx);

#endif
