#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>  // For AT_ERROR and c10::toString

#define AT_DISPATCH_FLOATING_TYPES_HALF_BFLOAT16(TYPE, NAME, ...)            \
  [&] {                                                                      \
    const auto _dispatch_type = TYPE;                                        \
    switch (static_cast<int>(_dispatch_type)) {                              \
      case static_cast<int>(at::ScalarType::Float): {                        \
        using scalar_t = float;                                              \
        return __VA_ARGS__();                                                \
      }                                                                      \
      case static_cast<int>(at::ScalarType::Half): {                         \
        using scalar_t = at::Half;                                             \
        return __VA_ARGS__();                                                \
      }                                                                      \
      case static_cast<int>(at::ScalarType::BFloat16): {                     \
        using scalar_t = at::BFloat16;                                         \
        return __VA_ARGS__();                                                \
      }                                                                      \
      default:                                                               \
        AT_ERROR(#NAME, " is not implemented for '", c10::toString(TYPE), "'"); \
    }                                                                        \
  }()