function fn() {
  var config = {};
  // System properties work in CI: mvn test -DbaseUrl=http://gateway:8080 -DauthToken=...
  config.baseUrl = karate.properties['baseUrl'] || 'http://127.0.0.1:8080';
  config.authToken = karate.properties['authToken'] || 'local-development-token';
  config.materializedEntity = karate.properties['materializedEntity'] || 'demo-account';
  return config;
}
