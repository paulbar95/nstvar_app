package com.barthel.nstvar.application.port.out;

import com.barthel.nstvar.domain.model.*;

import java.math.BigDecimal;

/**
 * Port for obtaining a global threshold for an AII type and scenario.
 */
public interface FetchGlobalThresholdPort {
    /**
     * Fetch the threshold used to normalise AII values.
     *
     * @param type the AII type
     * @param scenario the scenario context
     * @return the threshold value
     */
    BigDecimal fetchThreshold(AiiType type, Scenario scenario);
}
