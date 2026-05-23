/* SPDX-License-Identifier: Apache-2.0
 *
 * Public types for the all-C++ GDTW port.
 * Mirrors fields from python/gdtw/gdtw.py::__init__ as concrete C++ types.
 */
#pragma once

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <variant>
#include <vector>

namespace gdtw {

enum class Penalty { L1, L2, HUBER, CUSTOM };

using PenaltyFn = std::function<double(double)>;

/* SignalSpec mirrors the three Python forms (function / array / (array, t)).
 *   - SamplesScalar: 1-D contiguous samples, evenly spaced over [0,1]
 *   - SamplesMulti : 2-D row-major (T, d), evenly spaced over [0,1]
 *   - SamplesWithT : (samples, t) — `samples` is either scalar or multi
 *   - Callable     : f(t) -> scalar or f(t) -> vector
 */
struct SignalSamples {
    std::vector<double> data;  // length T*d, row-major
    std::size_t d = 1;         // channel count
};

struct SignalSamplesWithT {
    SignalSamples samples;
    std::vector<double> t;     // length T
};

using SignalCallableScalar = std::function<double(double)>;
using SignalCallableVector = std::function<std::vector<double>(double)>;

using SignalSpec = std::variant<
    std::monostate,
    SignalSamples,
    SignalSamplesWithT,
    SignalCallableScalar,
    SignalCallableVector
>;

/* All knobs from gdtw.py::__init__. Defaults match Python bit-for-bit. */
struct GDTWParams {
    SignalSpec x{};
    SignalSpec y{};
    std::vector<double> t{};    // empty == "not set" (we'll build linspace(0,1,N))

    double lambda_cum   = 1.0;
    double lambda_inst  = 0.1;
    Penalty loss        = Penalty::L2;
    Penalty r_cum       = Penalty::L2;
    Penalty r_inst      = Penalty::L2;
    double huber_delta  = 1.0;
    PenaltyFn custom_loss{};   // used iff loss == CUSTOM

    bool symmetric      = false;

    int N               = -1;   // -1 == auto
    int N_default       = 300;
    int M               = -1;   // -1 == auto
    int M_max           = 300;
    double eta          = 0.15;

    double s_min        = 1e-8;
    double s_max        = 1e8;
    double s_beta       = 0.0;
    bool BC_start_stop  = true;

    int max_iters       = 10;
    double epsilon_abs  = 1e-1;
    double epsilon_rel  = 1e-2;

    bool scale_signals  = true;
    int verbose         = 0;
};

struct GDTWResult {
    std::vector<double> t;
    std::vector<double> tau;
    std::vector<double> x;     // flattened (N*d), row-major
    std::vector<double> y;     // flattened (N*d), row-major
    std::vector<double> x_hat; // x_f(tau) flattened (N*d), row-major
    std::vector<int> path;
    double f_tau = 0.0;
    int iterations = 0;
    int N = 0;
    int M = 0;
    int d = 1;
};

class GDTWError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

}  // namespace gdtw
