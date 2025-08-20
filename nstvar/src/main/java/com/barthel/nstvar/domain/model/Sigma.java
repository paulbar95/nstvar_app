package com.barthel.nstvar.domain.model;

import java.math.BigDecimal;

/**
 * Represents an erosion shock (relative production loss) for a given
 * combination of AII type, region and scenario.
 *
 * @param aiiType the type of impact
 * @param region  the affected region
 * @param scenario the scenario context
 * @param value  the relative loss (>= -1)
 */
public record Sigma(AiiType aiiType, Region region, Scenario scenario, BigDecimal value) {
    public Sigma {
        if (value == null || value.compareTo(BigDecimal.valueOf(-1)) < 0) {
            throw new IllegalArgumentException("Sigma must not be less than -1");
        }
    }
}
