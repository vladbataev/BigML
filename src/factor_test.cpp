#include <gtest/gtest.h>

#include <cstdlib>

#include "factor.h"
#include "X.h"

using namespace Eigen;

TEST(Factor, LossDecreases) {
    srand(1231541);

    std::vector<int> lags{1, 5, 10};
    MatrixXd Y = MatrixXd::Random(50, 50);
    MatrixXb sigma = Eigen::MatrixXb::Ones(Y.rows(), Y.cols());
    Regularizer opts{lags, 1.0, 1.0, 1.0};
    auto lat_dim = 2;
    auto steps = 5;

    auto [Wt, result] = Init(Y, opts, lat_dim);
    auto loss = Loss(Y, sigma, opts, result);

    for (size_t i = 0; i < 3; i++) {
        Step(Y, sigma, opts, result, Wt, true);
        auto after = Loss(Y, sigma, opts, result);
        EXPECT_LE(after + 1e-4, loss);
        loss = after;
    }
}
