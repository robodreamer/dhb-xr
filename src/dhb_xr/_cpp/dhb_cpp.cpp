/**
 * Minimal C++ extension for dhb_xr (nanobind).
 * Placeholder for future performance-critical encode/decode.
 */
#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(_dhb_xr_cpp, m) {
  m.def(
      "cpp_version", []() { return 1; },
      "Return C++ extension version (1 = placeholder).");
  m.def(
      "add", [](double a, double b) { return a + b; }, nb::arg("a"),
      nb::arg("b"), "Placeholder: add two doubles.");
}
