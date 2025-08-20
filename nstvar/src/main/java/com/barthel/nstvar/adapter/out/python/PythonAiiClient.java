package com.barthel.nstvar.adapter.out.python;

import com.barthel.nstvar.application.port.out.FetchGlobalThresholdPort;
import com.barthel.nstvar.application.port.out.FetchRegionAiiValuePort;
import com.barthel.nstvar.domain.model.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;

import java.math.BigDecimal;

/**
 * WebClient based adapter fetching AII values from the Python service.
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class PythonAiiClient implements FetchRegionAiiValuePort, FetchGlobalThresholdPort {

    private final WebClient webClient = WebClient.builder()
            .baseUrl("http://localhost:8000/api/pm25")
            .build();

    @Override
    public BigDecimal fetchRegionalValue(AiiType type, Region region, Scenario scenario) {
        log.info("Fetching regional PM2.5 value for region={}, scenario={}", region.isoCode(), scenario.name());
        return webClient.get()
                .uri(uriBuilder -> uriBuilder
                        .path("/region")
                        .queryParam("region", region.isoCode())
                        .queryParam("scenario", scenario.name())
                        .build())
                .retrieve()
                .bodyToMono(ValueResponse.class)
                .block()
                .value();
    }

    @Override
    public BigDecimal fetchThreshold(AiiType type, Scenario scenario) {
        log.info("Fetching PM2.5 threshold for scenario={}", scenario.name());
        return webClient.get()
                .uri(uriBuilder -> uriBuilder
                        .path("/threshold")
                        .queryParam("scenario", scenario.name())
                        .build())
                .retrieve()
                .bodyToMono(ValueResponse.class)
                .block()
                .value();
    }

    private record ValueResponse(BigDecimal value) {}
}
