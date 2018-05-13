#include <gtest/gtest.h>

#include <cstdlib>

#include "X.h"
#include "factor.h"

using namespace Eigen;

TEST(Factor, LossDecreases) {
    srand(1231541);

    std::vector<int> lags{1, 5, 10};
    MatrixXd Y = MatrixXd::Random(50, 50);
    MatrixXb omega = Eigen::MatrixXb::Ones(Y.rows(), Y.cols());
    Regularizer opts{lags, 1.0, 1.0, 1.0, 1.0};
    auto lat_dim = 2;
    auto steps = 5;

    auto [Wt, result] = Init(Y, opts, lat_dim);
    auto loss = Loss(Y, omega, opts, result);

    for (size_t i = 0; i < 3; i++) {
        Step(Y, omega, opts, result, Wt, true);
        auto after = Loss(Y, omega, opts, result);
        EXPECT_LE(after + 1e-4, loss);
        loss = after;
    }
}

TEST(Factor, MissingValues) {
    srand(1231541);

    std::vector<int> lags{1, 5, 10};
    MatrixXd Y = MatrixXd::Random(100, 100);
    MatrixXb omega = Eigen::MatrixXb::Ones(Y.rows(), Y.cols());
    Regularizer opts{lags, 1.0, 1.0, 1.0, 1.0};
    auto lat_dim = 5;
    auto steps = 100;

    auto [Wt, result] = Init(Y, opts, lat_dim);
    auto loss = Loss(Y, omega, opts, result);

    for (int i = 0; i < Y.cols(); i++) {
        for (int j = 0; j < Y.rows() * 0.15; j++) {
            omega(i, abs(rand())%Y.rows()) = false;
        }
    }

    for (size_t i = 0; i < 3; i++) {
        Step(Y, omega, opts, result, Wt, true);
        auto after = Loss(Y, omega, opts, result);
        EXPECT_LE(after + 1e-4, loss);
        loss = after;
    }
}
