package com.barthel.nstvar.application.port.in;

import com.barthel.nstvar.domain.model.*;

/**
 * Use case for computing the sigma value for a given AII type, region and scenario.
 */
public interface ComputeSigmaUseCase {
    /**
     * Computes the sigma for the given input parameters.
     *
     * @param aiiType the impact type
     * @param region the region
     * @param scenario the scenario context
     * @return the resulting {@link Sigma}
     */
    Sigma computeSigma(AiiType aiiType, Region region, Scenario scenario);

    /**
     * Whether this implementation supports the given AII type.
     *
     * @param aiiType the type to check
     * @return true if supported
     */
    boolean supports(AiiType aiiType);
}
