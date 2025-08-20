package com.barthel.nstvar.domain.model;

import java.math.BigDecimal;

/**
 * Direct exposure representing DS multiplied by sigma.
 *
 * @param sector the economic sector
 * @param sigma the shock value
 * @param value the direct exposure amount
 */
public record DirectExposure(Sector sector, Sigma sigma, BigDecimal value) {
    public DirectExposure {
        if (value == null) {
            throw new IllegalArgumentException("Value is required");
        }
    }
}
