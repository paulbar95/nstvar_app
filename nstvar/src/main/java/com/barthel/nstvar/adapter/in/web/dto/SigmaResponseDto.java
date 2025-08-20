package com.barthel.nstvar.adapter.in.web.dto;

import java.math.BigDecimal;

public record SigmaResponseDto(String aiiType, String region, String scenario, BigDecimal sigma) {}
