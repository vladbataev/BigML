#include "factor.h"
#include <iostream>
#include "F.h"
#include "W.h"
#include "X.h"

using namespace Eigen;

std::tuple<CachedWTransform, Factorization> Init(const MatrixXd& Y,
                                                 const Regularizer& opts,
                                                 size_t lat_dim) {
    return {CachedWTransform(opts.lags),
            {.W = MatrixXd::Random(lat_dim, opts.lags.size()),
             .F = MatrixXd::Random(lat_dim, Y.rows()),
             .X = MatrixXd::Random(lat_dim, Y.cols())}};
}

double Loss(const MatrixXd& Y, const MatrixXb& omega, const Regularizer& opts,
            const Factorization& result) {
    double easy = ((Y - result.F.transpose() * result.X)
                       .cwiseProduct(omega.cast<double>()))
                      .squaredNorm() +
                  opts.lambdaF * result.F.squaredNorm() +
                  opts.lambdaW * result.W.squaredNorm() +
                  opts.lambdaX * opts.nu / 2 * result.X.squaredNorm();
    double x_part = 0;
    auto T = Y.cols();
    auto L = *std::max_element(opts.lags.begin(), opts.lags.end());
    for (int r = 0; r < result.X.rows(); ++r) {
        for (int t = L; t < T; ++t) {
            if (omega(r, t)) {
                auto tmp = result.X(r, t);
                for (int l = 0; l < opts.lags.size(); ++l) {
                    tmp -= result.W(r, l) * result.X(r, t - opts.lags[l]);
                }
                x_part += tmp * tmp * opts.lambdaX / 2;
            }
        }
    }
    return easy + x_part;
}

void Step(const MatrixXd& Y, const MatrixXb& omega, const Regularizer& opts,
          Factorization& result, CachedWTransform& Wt, bool verify) {
    auto print = [&](auto msg) {
        if (verify) {
            std::cerr << "loss after " << msg << " : "
                      << Loss(Y, omega, opts, result) << std::endl;
        }
    };
    result.W = OptimizeByW(result.X, opts.lags, opts.lambdaX, opts.lambdaW);
    print("W");
    result.F = OptimizeByF(Y, result.X, opts.lambdaF);
    print("F");
    OptimizeByX(Y, omega, result.F, result.X, Wt, result.W, opts.nu,
                opts.lambdaX, verify);
    print("X");
}

Factorization Factorize(MatrixXd Y, MatrixXb omega, Regularizer opts,
                        size_t lat_dim, size_t steps, bool verbose) {
    auto [Wt, result] = Init(Y, opts, lat_dim);

    if (verbose) {
        std::cerr << "Loss: " << Loss(Y, omega, opts, result) << "\n";
    }
    for (size_t i = 0; i < steps; i++) {
        Step(Y, omega, opts, result, Wt, false);
        if (verbose) {
            std::cerr << "Loss after" << i
                      << "th iteration: " << Loss(Y, omega, opts, result)
                      << "\n";
        }
    }
    return result;
}
