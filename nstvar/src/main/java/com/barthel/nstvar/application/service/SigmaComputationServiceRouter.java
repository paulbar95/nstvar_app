package com.barthel.nstvar.application.service;

import com.barthel.nstvar.application.port.in.ComputeSigmaUseCase;
import com.barthel.nstvar.domain.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.context.annotation.Primary;

import java.util.List;

/**
 * Facade routing requests to the appropriate sigma computation service.
 */
@Service
@RequiredArgsConstructor
@Primary
public class SigmaComputationServiceRouter implements ComputeSigmaUseCase {

    private final List<ComputeSigmaUseCase> implementations;

    @Override
    public Sigma computeSigma(AiiType aiiType, Region region, Scenario scenario) {
        return implementations.stream()
                .filter(i -> i.supports(aiiType))
                .findFirst()
                .orElseThrow(() -> new UnsupportedOperationException("No handler for AII: " + aiiType))
                .computeSigma(aiiType, region, scenario);
    }

    @Override
    public boolean supports(AiiType aiiType) {
        // Never called directly
        return false;
    }
}
