package fraud.api;

import com.intuit.karate.junit5.Karate;

class FraudApiTest {
    @Karate.Test
    Karate gatewayContracts() {
        return Karate.run("gateway").relativeTo(getClass());
    }
}
