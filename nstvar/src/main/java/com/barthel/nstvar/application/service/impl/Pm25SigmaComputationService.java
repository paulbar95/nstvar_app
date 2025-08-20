package com.barthel.nstvar.application.service.impl;

import com.barthel.nstvar.application.port.in.ComputeSigmaUseCase;
import com.barthel.nstvar.application.port.out.*;
import com.barthel.nstvar.domain.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;

/**
 * Sigma computation implementation for the PM2.5 AII type.
 */
@Service
@RequiredArgsConstructor
public class Pm25SigmaComputationService implements ComputeSigmaUseCase {

    private final FetchRegionAiiValuePort fetchRegionAiiValuePort;
    private final FetchGlobalThresholdPort fetchGlobalThresholdPort;
    private final StoreSigmaPort storeSigmaPort;

    @Override
    public Sigma computeSigma(AiiType type, Region region, Scenario scenario) {
        BigDecimal regionValue = fetchRegionAiiValuePort.fetchRegionalValue(type, region, scenario);
        BigDecimal threshold = fetchGlobalThresholdPort.fetchThreshold(type, scenario);
        BigDecimal sigmaValue = regionValue.divide(threshold, 6, BigDecimal.ROUND_HALF_UP);

        Sigma sigma = new Sigma(type, region, scenario, sigmaValue);
        storeSigmaPort.store(sigma);
        return sigma;
    }

    @Override
    public boolean supports(AiiType aiiType) {
        return aiiType == AiiType.PM25;
    }
}
