package com.barthel.nstvar.domain.model;

import java.math.BigDecimal;

/**
 * Result of the Leontief propagation capturing indirect risk values.
 *
 * @param sector the sector evaluated
 * @param region the region of the sector
 * @param aiiType the impact type
 * @param scenario the scenario context
 * @param indirectRisk the resulting indirect risk value
 */
public record IndirectRiskResult(
        Sector sector,
        Region region,
        AiiType aiiType,
        Scenario scenario,
        BigDecimal indirectRisk) {

    public IndirectRiskResult {
        if (indirectRisk == null) {
            throw new IllegalArgumentException("Indirect risk value is required");
        }
    }
}
