#include "fraud_engine/execution_engine.hpp"
#include <cuda_runtime.h>

namespace fraud_engine {
DeviceCapabilities cuda_device_capabilities() {
  DeviceCapabilities result{};
  result.cuda_compiled = true;
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return result;
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return result;
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) return result;
  result.cuda_available = true;
  result.compute_capability_major = properties.major;
  result.compute_capability_minor = properties.minor;
  return result;
}
}  // namespace fraud_engine
