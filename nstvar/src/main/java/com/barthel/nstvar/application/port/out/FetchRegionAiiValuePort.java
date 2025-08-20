package com.barthel.nstvar.application.port.out;

import com.barthel.nstvar.domain.model.*;

import java.math.BigDecimal;

/**
 * Port for fetching the regional AII value for a region and scenario.
 */
public interface FetchRegionAiiValuePort {
    /**
     * Retrieve the AII value for the given combination of type, region and scenario.
     *
     * @param type the AII type
     * @param region the region
     * @param scenario the scenario context
     * @return the retrieved value
     */
    BigDecimal fetchRegionalValue(AiiType type, Region region, Scenario scenario);
}
