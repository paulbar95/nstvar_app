package com.barthel.nstvar;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import com.barthel.nstvar.application.port.out.FetchGlobalThresholdPort;
import com.barthel.nstvar.application.port.out.FetchRegionAiiValuePort;
import com.barthel.nstvar.application.port.out.StoreSigmaPort;

@SpringBootTest(properties = {
        "spring.autoconfigure.exclude=org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration"
})
class NstvarApplicationTests {

    @MockBean
    private FetchRegionAiiValuePort fetchRegionAiiValuePort;

    @MockBean
    private FetchGlobalThresholdPort fetchGlobalThresholdPort;

    @MockBean
    private StoreSigmaPort storeSigmaPort;

	@Test
	void contextLoads() {
	}

}
