package com.barthel.nstvar.domain.model;

/**
 * Two-letter region code, currently using ISO country codes.
 *
 * @param isoCode the ISO 3166-1 alpha-2 code
 */
public record Region(String isoCode) {
    public Region {
        if (isoCode == null || isoCode.length() != 2) {
            throw new IllegalArgumentException("ISO code must be a 2-letter country code");
        }
    }
}
