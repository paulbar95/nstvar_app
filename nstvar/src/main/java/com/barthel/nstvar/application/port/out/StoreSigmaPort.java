package com.barthel.nstvar.application.port.out;

import com.barthel.nstvar.domain.model.Sigma;

/**
 * Port for storing computed sigma values.
 */
public interface StoreSigmaPort {
    /**
     * Persist the given sigma value.
     *
     * @param sigma the sigma to store
     */
    void store(Sigma sigma);
}
