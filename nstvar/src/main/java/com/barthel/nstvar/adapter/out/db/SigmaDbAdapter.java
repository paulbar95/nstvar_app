package com.barthel.nstvar.adapter.out.db;

import com.barthel.nstvar.adapter.out.db.entity.SigmaEntity;
import com.barthel.nstvar.adapter.out.db.repository.SigmaRepository;
import com.barthel.nstvar.application.port.out.StoreSigmaPort;
import com.barthel.nstvar.domain.model.Sigma;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class SigmaDbAdapter implements StoreSigmaPort {

    private final SigmaRepository repository;

    @Override
    public void store(Sigma sigma) {
        SigmaEntity entity = SigmaEntity.builder()
                .aiiType(sigma.aiiType().name())
                .region(sigma.region().isoCode())
                .scenario(sigma.scenario().name())
                .value(sigma.value())
                .build();
        repository.save(entity);
    }
}
