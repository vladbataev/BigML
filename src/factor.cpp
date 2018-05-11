#include "W.h"
#include "X.h"
#include "F_optimize.h"
#include "factor.h"
#include <iostream>

using namespace Eigen;

std::tuple<CachedWTransform, Factorization> Init(const MatrixXd& Y, const Regularizer& opts, size_t lat_dim) {
    return {
        CachedWTransform(opts.lags),
        {
            .W = MatrixXd::Random(lat_dim, opts.lags.size()),
            .F = MatrixXd::Random(lat_dim, Y.rows()),
            .X = MatrixXd::Random(lat_dim, Y.cols())
        }
    };
}

double Loss(const MatrixXd& Y, const MatrixXb& sigma, const Regularizer& opts, const Factorization& result) {
    auto easy = ((Y - result.F.transpose() * result.X).cwiseProduct(sigma.cast<double>())).squaredNorm() + opts.lambdaF * result.F.squaredNorm() +
            opts.lambdaW * result.W.squaredNorm() + opts.lambdaX * opts.nu / 2 * result.X.squaredNorm();
    auto x_part = 0;
    auto T = Y.cols();
    auto L = *std::max_element(opts.lags.begin(), opts.lags.end());
    for (int r = 0; r < result.X.rows(); ++r) {
        for (int t = L; t < T; ++t) {
            auto tmp = result.X(r, t);
            for (int l = 0 ; l < opts.lags.size(); ++l) {
                tmp -= result.W(r,  l) * result.X(r, t - opts.lags[l]);
            }
            x_part += tmp * tmp * opts.lambdaX / 2;
        }
    }
    return easy + x_part;
}


void Step(const MatrixXd& Y, const MatrixXb& sigma, const Regularizer& opts, Factorization& result, CachedWTransform& Wt, double tol) {
    #ifndef NDEBUG
        auto before = Loss(Y, sigma, opts, result);
        std::cout << "Loss before: " << before << "\n";
    #endif
    result.F = OptimizeByF(Y, result.X, opts.lambdaF);
    #ifndef NDEBUG
        auto after =  Loss(Y, sigma, opts, result);
        std::cout << "Loss after F: " << after << "\n";
        assert(after < before);
        before = after;
    #endif
    optimize_X(Y, sigma, result.F, result.X, Wt, result.W, opts.nu, opts.lambdaX, tol);
    #ifndef NDEBUG
        after =  Loss(Y, sigma, opts, result);
        std::cout << "Loss after X: " << after << "\n";
        assert(after < before);
        before = after;
    #endif
}

Factorization Factorize(MatrixXd Y, MatrixXb sigma, Regularizer opts, size_t lat_dim, size_t steps, double tol) {
    auto [Wt, result] = Init(Y, opts, lat_dim);

    for (size_t i = 0; i < steps; i++) {
        Step(Y, sigma, opts, result, Wt, tol);
    }
    return result;
}
