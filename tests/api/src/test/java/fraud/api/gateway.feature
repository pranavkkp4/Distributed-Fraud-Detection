Feature: Gateway HTTP contract

  Background:
    * url baseUrl
    * def authorization = 'Bearer ' + authToken
    * def unknownEntity = 'contract-missing-feature'
    * def validPayload = { entity_id: '#(unknownEntity)', amount: 12.50 }

  Scenario: health endpoint reports JSON readiness for basic liveness
    Given path 'healthz'
    When method get
    Then status 200
    And match response == { status: 'ok' }

  Scenario: scoring requires a bearer token when gateway auth is enabled
    Given path 'v1', 'score'
    And request validPayload
    When method post
    Then status 401
    And match response == { error: 'unauthorized' }

  Scenario: authenticated malformed payload is rejected before inference
    Given path 'v1', 'score'
    And header Authorization = authorization
    And request { entity_id: '', amount: -1, unexpected: true }
    When method post
    Then status 400
    And match response == { error: 'invalid request' }

  Scenario: authenticated materialized features reach the worker without fallback
    Given path 'v1', 'score'
    And header Authorization = authorization
    And header X-Request-ID = 'karate-materialized-success'
    And request { entity_id: '#(materializedEntity)', amount: 12.50 }
    When method post
    Then status 200
    And match header X-Request-ID == 'karate-materialized-success'
    And match response contains { request_id: 'karate-materialized-success', score: '#number', confidence: '#number', decision: '#string', model: 'deterministic-cpu-v1', fallback: false, fallback_reason: '' }
    And match response.explanation == '#[]'
    And assert response.explanation.length > 0
    And assert response.explanation.length <= 3
    And match response.calibration_version == '#string'
    And match response.policy_version == '#string'
    And match response.feature_version == 'rolling-features-v1-32'

  Scenario: supplied request IDs are preserved in response metadata and payload
    * def traceId = 'karate-contract-request-id'
    Given path 'v1', 'score'
    And header Authorization = authorization
    And header X-Request-ID = traceId
    And request validPayload
    When method post
    Then status 503
    And match header X-Request-ID == traceId
    And match response contains { request_id: '#string', score: '#number', confidence: '#number', decision: 'deny', model: 'fail-closed', fallback: true, fallback_reason: 'feature_unavailable' }
    And match response.explanation == '#[]'
    And match response.calibration_version == '#string'
    And match response.policy_version == '#string'
    And match response.feature_version == '#string'
    And match response.request_id == traceId

  Scenario: missing features use a deterministic fail-closed response
    Given path 'v1', 'score'
    And header Authorization = authorization
    And header X-Request-ID = 'karate-fallback-one'
    And request validPayload
    When method post
    Then status 503
    And def firstFallback = response
    Given path 'v1', 'score'
    And header Authorization = authorization
    And header X-Request-ID = 'karate-fallback-two'
    And request validPayload
    When method post
    Then status 503
    And match response.score == firstFallback.score
    And match response.decision == firstFallback.decision
    And match response.model == firstFallback.model
    And match response.fallback == firstFallback.fallback
    And match response.fallback_reason == firstFallback.fallback_reason
    And match response.confidence == firstFallback.confidence
