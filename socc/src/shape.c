#include "../include/types.h"
#include "../include/shape.h"

#include <math.h>
#include <string.h>


// Lifecycle


SocStatus soc_shape_init(SocShape* shape, const SocComplex* coeffs) {
    if (shape == NULL || coeffs == NULL) {
        return SOC_ERR_NULL_POINTER;
    }
    memcpy(shape->coeffs, coeffs, sizeof(SocComplex) * SOC_COEFF_LENGTH);
    return SOC_OK;
}

SocStatus soc_shape_zero(SocShape* shape) {
    if (shape == NULL) {
        return SOC_ERR_NULL_POINTER;
    }
    memset(shape->coeffs, 0, sizeof(SocComplex) * SOC_COEFF_LENGTH);
    return SOC_OK;
}

SocStatus soc_shape_copy(const SocShape* src, SocShape* dst) {
    if (src == NULL || dst == NULL) {
        return SOC_ERR_NULL_POINTER;
    }
    memcpy(dst->coeffs, src->coeffs, sizeof(SocComplex) * SOC_COEFF_LENGTH);
    return SOC_OK;
}


// Coefficient access


SocStatus soc_shape_get(const SocShape* shape, SocHarmonic k, SocComplex* out) {
    if (shape == NULL || out == NULL) {
        return SOC_ERR_NULL_POINTER;
    }
    int idx = soc_k_to_index(k);
    if (idx < 0) {
        return SOC_ERR_INVALID_INDEX;
    }
    *out = shape->coeffs[idx];
    return SOC_OK;
}

SocStatus soc_shape_set(SocShape* shape, SocHarmonic k, SocComplex value) {
    if (shape == NULL) {
        return SOC_ERR_NULL_POINTER;
    }
    int idx = soc_k_to_index(k);
    if (idx < 0) {
        return SOC_ERR_INVALID_INDEX;
    }
    shape->coeffs[idx] = value;
    return SOC_OK;
}

SocStatus soc_shape_get_flat(const SocShape* shape, SocIndex idx, SocComplex* out) {
    if (shape == NULL || out == NULL) {
        return SOC_ERR_NULL_POINTER;
    }
    if (idx < 0 || idx >= SOC_COEFF_LENGTH) {
        return SOC_ERR_INVALID_INDEX;
    }
    *out = shape->coeffs[idx];
    return SOC_OK;
}

SocStatus soc_shape_set_flat(SocShape* shape, SocIndex idx, SocComplex value) {
    if (shape == NULL) {
        return SOC_ERR_NULL_POINTER;
    }
    if (idx < 0 || idx >= SOC_COEFF_LENGTH) {
        return SOC_ERR_INVALID_INDEX;
    }
    shape->coeffs[idx] = value;
    return SOC_OK;
}


// Derived quantities


double soc_shape_norm_full(const SocShape* shape) {
    if (shape == NULL) {
        return 0.0;
    }
    double sum = 0.0;
    for (int i = 0; i < SOC_COEFF_LENGTH; i++) {
        double mag = cabs(shape->coeffs[i]);
        sum += mag * mag;
    }
    return sum;
}

double soc_shape_norm_shape(const SocShape* shape) {
    if (shape == NULL) {
        return 0.0;
    }
    double sum = 0.0;
    for (int i = 0; i < SOC_COEFF_LENGTH; i++) {
        if (i == SOC_N) continue;  // skip k=0
        double mag = cabs(shape->coeffs[i]);
        sum += mag * mag;
    }
    return sum;
}

SocComplex soc_shape_c0(const SocShape* shape) {
    if (shape == NULL) {
        return 0.0 + 0.0 * I;
    }
    return shape->coeffs[SOC_N];  // k=0 sits at flat index N
}


// Reconstruction


SocStatus soc_shape_reconstruct(const SocShape* shape,
                                const double* t,
                                int n,
                                SocComplex* out) {
    if (shape == NULL || t == NULL || out == NULL) {
        return SOC_ERR_NULL_POINTER;
    }
    if (n <= 0) {
        return SOC_ERR_INVALID_INDEX;
    }

    for (int i = 0; i < n; i++) {
        SocComplex sum = 0.0 + 0.0 * I;
        for (int idx = 0; idx < SOC_COEFF_LENGTH; idx++) {
            int k = idx - SOC_N;
            double phase = 2.0 * SOC_PI * (double)k * t[i];
            SocComplex e = cos(phase) + I * sin(phase);
            sum += shape->coeffs[idx] * e;
        }
        out[i] = sum;
    }
    return SOC_OK;
}


// Indexing helpers


int soc_k_to_index(SocHarmonic k) {
    if (k < -SOC_N || k > SOC_N) {
        return -1;
    }
    return k + SOC_N;
}

SocHarmonic soc_index_to_k(SocIndex idx) {
    if (idx < 0 || idx >= SOC_COEFF_LENGTH) {
        return 0;  // Caller should validate range separately
    }
    return idx - SOC_N;
}
