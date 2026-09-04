#include "../include/types.h"
#include "../include/metrics.h"

#include <math.h>
#include <stddef.h>

// eta


double soc_eta(const SocShape* a, const SocShape* b) {
    if (a == NULL || b == NULL) {
        return 0.0;
    }

    double norm_a_sq = 0.0;
    double norm_b_sq = 0.0;
    SocComplex inner = 0.0 + 0.0 * I;

    for (int idx = 0; idx < SOC_COEFF_LENGTH; idx++) {
        if (idx == SOC_N) continue;  // skip k=0

        double mag_a = cabs(a->coeffs[idx]);
        double mag_b = cabs(b->coeffs[idx]);
        norm_a_sq += mag_a * mag_a;
        norm_b_sq += mag_b * mag_b;

        // inner = sum a_k * conj(b_k)
        inner += a->coeffs[idx] * conj(b->coeffs[idx]);
    }

    double norm_a = sqrt(norm_a_sq);
    double norm_b = sqrt(norm_b_sq);
    double denom = norm_a * norm_b;

    if (denom < 1e-12) {
        return 0.0;
    }

    return cabs(inner) / denom;
}


// y_rx


double soc_y_rx(const SocShape* received, const SocShape* target,
                int formula, double threshold, double gamma) {
    if (received == NULL || target == NULL) {
        return 0.0;
    }
    if (formula != SOC_YRX_FORMULA_SPEC && formula != SOC_YRX_FORMULA_FIXED) {
        return 0.0;
    }

    double e = soc_eta(received, target);
    if (e < threshold) {
        return 0.0;
    }

    double norm_sq;
    if (formula == SOC_YRX_FORMULA_SPEC) {
        // Full norm, includes C0 — the §5 leak
        norm_sq = 0.0;
        for (int i = 0; i < SOC_COEFF_LENGTH; i++) {
            double mag = cabs(received->coeffs[i]);
            norm_sq += mag * mag;
        }
    } else {
        // Shape norm, excludes C0 — the fix
        norm_sq = 0.0;
        for (int i = 0; i < SOC_COEFF_LENGTH; i++) {
            if (i == SOC_N) continue;
            double mag = cabs(received->coeffs[i]);
            norm_sq += mag * mag;
        }
    }

    return gamma * norm_sq * e;
}


// energy_audit


SocStatus soc_energy_audit(const SocShape* before, const SocShape* after,
                           SocEnergyAudit* audit) {
    if (before == NULL || after == NULL || audit == NULL) {
        return SOC_ERR_NULL_POINTER;
    }

    double e_before = 0.0;
    double e_after = 0.0;

    for (int i = 0; i < SOC_COEFF_LENGTH; i++) {
        double mag_b = cabs(before->coeffs[i]);
        double mag_a = cabs(after->coeffs[i]);
        e_before += mag_b * mag_b;
        e_after += mag_a * mag_a;
    }

    audit->energy_before = e_before;
    audit->energy_after = e_after;
    audit->energy_delta = e_after - e_before;
    audit->norm_preserved = (fabs(audit->energy_delta) < 1e-12) ? 1 : 0;

    return SOC_OK;
}
