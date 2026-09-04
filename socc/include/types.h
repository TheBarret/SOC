// types.h

// Todo: centralize config based parameters


#ifndef TYPES_H
#define TYPES_H

#include <complex.h>

// Dimensional parameters
#define SOC_N 8
#define SOC_COEFF_LENGTH (2 * SOC_N + 1)

// Scalar type
typedef double complex SocComplex;

// Index types
typedef int SocHarmonic;   // k in [-N, N]
typedef int SocIndex;      // flat index in [0, 2N]

// Core data types
typedef struct {
    SocComplex coeffs[SOC_COEFF_LENGTH];
} SocShape;

typedef struct {
    int n;
    SocComplex* z;
} SocCurve;

typedef struct {
    double energy_before;
    double energy_after;
    double energy_delta;
    int    norm_preserved;
} SocEnergyAudit;

// Status codes
typedef enum {
    SOC_OK = 0,
    SOC_ERR_NULL_POINTER,
    SOC_ERR_INVALID_INDEX,
    SOC_ERR_INVALID_MODE,
    SOC_ERR_ENERGY_ZERO,
} SocStatus;

#endif

#ifndef SOC_PI
#define SOC_PI 3.14159265358979323846
#endif
