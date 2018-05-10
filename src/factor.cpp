#include "W.h"
#include "X.h"
#include "F_optimize.h"
#include "factor.h"

using namespace Eigen;

std::tuple<CachedWTransform, Factorization> Init(const MatrixXd& Y, const Regularizer& opts, size_t lat_dim) {
    return {
        CachedWTransform(opts.lags),
        {
            .W = MatrixXd(Y.cols(), opts.lags.size()),
            .F = MatrixXd(lat_dim, Y.rows()),
            .X = MatrixXd(lat_dim, Y.cols())
        }
    };
}

void Step(const MatrixXd& Y, const Regularizer& opts, Factorization& result, CachedWTransform& Wt) {
    result.F = OptimizeByF(Y, result.X, opts.lambdaF);
    result.W = OptimizeByW(result.X, opts.lags, opts.lambdaX, opts.lambdaW);
    optimize_X(Y, result.F, result.X, Wt, result.W, opts.lambdaX);
}

Factorization Factorize(MatrixXd Y, Regularizer opts, size_t lat_dim, size_t steps) {
    auto [Wt, result] = Init(Y, opts, lat_dim);

    for (size_t i = 0; i < steps; i++) {
        Step(Y, opts, result, Wt);
    }
    return result;
}
