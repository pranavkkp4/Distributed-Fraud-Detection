#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace fraud::daemon {
struct CutlassStatus final { bool available; std::string_view message; };
CutlassStatus cutlass_int8_gemm(const std::int8_t* a, const std::int8_t* b, std::int32_t* c,
                                int m, int n, int k, void* stream) noexcept;
}  // namespace fraud::daemon
