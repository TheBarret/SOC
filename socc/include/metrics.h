#ifndef METRICS_H
#define METRICS_H

#include "types.h"


// Metrics and readouts (spec §2, §3, §5)
//
// eta operates on the shape subspace (k != 0). C0 is excluded by design.
// The decoupled-term leak (§5) is a property of combining eta with a norm
// that includes C0. Both readout formulas are provided explicitly.


// Shape similarity metric (spec §2).
//
// eta(A, B) = |<A, B>_shape| / (||A||_shape * ||B||_shape)
//
// Invariant to: global phase rotation, uniform gain, C0 modification.
// Range: [0, 1]. 1.0 = identical shape geometry.
//
// Returns 0.0 if either shape has zero energy in the shape subspace.
double soc_eta(const SocShape* a, const SocShape* b);

// Scalar readout with match gate (spec §3).
//
// formula: 0 = spec (full norm, contains §5 leak)
//          1 = fixed (shape norm, closes leak)
//
// threshold: eta below this returns 0.0.
// gamma: scalar gain.
double soc_y_rx(const SocShape* received, const SocShape* target,
                int formula, double threshold, double gamma);

// Energy audit: track energy flow across an operator (spec §3, §4C).
SocStatus soc_energy_audit(const SocShape* before, const SocShape* after,
                           SocEnergyAudit* audit);

// Convenience constants for y_rx formula parameter
#define SOC_YRX_FORMULA_SPEC  0
#define SOC_YRX_FORMULA_FIXED 1

#endif
