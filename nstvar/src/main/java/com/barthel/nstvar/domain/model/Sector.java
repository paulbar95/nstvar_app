package com.barthel.nstvar.domain.model;

/**
 * Sector descriptor. In the future this might reference a NACE code.
 *
 * @param name human-readable sector name
 */
public record Sector(String name) {
    public Sector {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("Sector name is required");
        }
    }
}
