package com.barthel.nstvar.adapter.out.db.entity;

import jakarta.persistence.*;
import lombok.*;

import java.math.BigDecimal;

@Entity
@Table(name = "sigma_values")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SigmaEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String aiiType;
    private String region;
    private String scenario;

    @Column(nullable = false, precision = 10, scale = 6)
    private BigDecimal value;
}
