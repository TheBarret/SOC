#include "../include/types.h"
#include "../include/generators.h"

#include <math.h>
#include <stdlib.h>


// curve_to_coeffs


SocStatus soc_curve_to_coeffs(const SocComplex* z_samples, int n,
                              SocShape* shape) {
    if (z_samples == NULL || shape == NULL) {
        return SOC_ERR_NULL_POINTER;
    }
    if (n <= 0) {
        return SOC_ERR_INVALID_INDEX;
    }

    // Initialize output to zero
    for (int idx = 0; idx < SOC_COEFF_LENGTH; idx++) {
        shape->coeffs[idx] = 0.0 + 0.0 * I;
    }

    // Direct DFT: C_k = (1/n) * sum_{j=0}^{n-1} z_j * exp(-i 2pi k j / n)
    for (int idx = 0; idx < SOC_COEFF_LENGTH; idx++) {
        int k = idx - SOC_N;
        SocComplex sum = 0.0 + 0.0 * I;

        for (int j = 0; j < n; j++) {
            double angle = -2.0 * SOC_PI * (double)k * (double)j / (double)n;
            SocComplex e = cos(angle) + I * sin(angle);
            sum += z_samples[j] * e;
        }

        shape->coeffs[idx] = sum / (double)n;
    }

    return SOC_OK;
}


// gen_circle


SocStatus soc_gen_circle(SocShape* shape, int n_samples) {
    if (shape == NULL) {
        return SOC_ERR_NULL_POINTER;
    }
    if (n_samples <= 0) {
        return SOC_ERR_INVALID_INDEX;
    }

    SocComplex* z = (SocComplex*)malloc(sizeof(SocComplex) * n_samples);
    if (z == NULL) {
        return SOC_ERR_NULL_POINTER;
    }

    for (int j = 0; j < n_samples; j++) {
        double t = (double)j / (double)n_samples;
        double angle = 2.0 * SOC_PI * t;
        z[j] = cos(angle) + I * sin(angle);
    }

    SocStatus status = soc_curve_to_coeffs(z, n_samples, shape);
    free(z);
    return status;
}


// gen_polygon


SocStatus soc_gen_polygon(SocShape* shape, int sides, int n_samples) {
    if (shape == NULL) {
        return SOC_ERR_NULL_POINTER;
    }
    if (sides < 3 || n_samples <= 0) {
        return SOC_ERR_INVALID_INDEX;
    }

    SocComplex* z = (SocComplex*)malloc(sizeof(SocComplex) * n_samples);
    if (z == NULL) {
        return SOC_ERR_NULL_POINTER;
    }

    double seg = 2.0 * SOC_PI / (double)sides;
    double cos_half_seg = cos(seg / 2.0);

    for (int j = 0; j < n_samples; j++) {
        double t = (double)j / (double)n_samples;
        double theta = 2.0 * SOC_PI * t;
        double phi = fmod(theta, seg) - seg / 2.0;
        double r = cos_half_seg / cos(phi);
        z[j] = r * (cos(theta) + I * sin(theta));
    }

    SocStatus status = soc_curve_to_coeffs(z, n_samples, shape);
    free(z);
    return status;
}


// gen_star


SocStatus soc_gen_star(SocShape* shape, int points, double inner_ratio,
                       int n_samples) {
    if (shape == NULL) {
        return SOC_ERR_NULL_POINTER;
    }
    if (points < 2 || n_samples <= 0) {
        return SOC_ERR_INVALID_INDEX;
    }
    if (inner_ratio <= 0.0 || inner_ratio >= 1.0) {
        return SOC_ERR_INVALID_INDEX;
    }

    SocComplex* z = (SocComplex*)malloc(sizeof(SocComplex) * n_samples);
    if (z == NULL) {
        return SOC_ERR_NULL_POINTER;
    }

    double seg = SOC_PI / (double)points;

    for (int j = 0; j < n_samples; j++) {
        double t = (double)j / (double)n_samples;
        double theta = 2.0 * SOC_PI * t;
        double phi = fmod(theta, 2.0 * seg);
        double tri = fabs(phi / seg - 1.0);
        double r = inner_ratio + (1.0 - inner_ratio) * (1.0 - tri);
        z[j] = r * (cos(theta) + I * sin(theta));
    }

    SocStatus status = soc_curve_to_coeffs(z, n_samples, shape);
    free(z);
    return status;
}
