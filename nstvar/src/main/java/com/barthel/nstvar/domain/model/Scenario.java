package com.barthel.nstvar.domain.model;

/**
 * Socioeconomic or climate scenario descriptor (e.g. SSP or RCP name).
 *
 * @param name scenario identifier
 */
public record Scenario(String name) {
    public Scenario {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("Scenario must not be blank");
        }
    }
}
