package com.barthel.nstvar.adapter.in.web.controller;

import com.barthel.nstvar.adapter.in.web.dto.SigmaResponseDto;
import com.barthel.nstvar.application.port.in.ComputeSigmaUseCase;
import com.barthel.nstvar.domain.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/sigma")
@RequiredArgsConstructor
public class SigmaController {

    private final ComputeSigmaUseCase computeSigmaUseCase;

    @GetMapping
    public SigmaResponseDto computeSigma(
            @RequestParam AiiType aiiType,
            @RequestParam String region,
            @RequestParam String scenario
    ) {
        Sigma result = computeSigmaUseCase.computeSigma(
                aiiType,
                new Region(region),
                new Scenario(scenario)
        );

        return new SigmaResponseDto(
                aiiType.name(),
                region,
                scenario,
                result.value()
        );
    }
}
