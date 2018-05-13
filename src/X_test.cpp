#include <gtest/gtest.h>

#include "X.h"

using namespace Eigen;

TEST(X_component, PositiveDefinite) {
    MatrixXd Y(2, 7);
    Y.setZero();
    MatrixXd F(3, 2);
    F << 13, 14, 15, 16, 17, 18;

    MatrixXd X(3, 7);
    X.setZero();

    auto t = CachedWTransform({2, 4});

    MatrixXd W(3, 2);
    W << 1, 2, 5, 6, 9, 10;

    auto s = MatrixXb::Ones(Y.rows(), Y.cols());

    OptimizeByX(Y, s, F, X, t, W, 1, 1, true);
}

TEST(X_component, Loss) {
    srand(1231541);
    std::vector<int> lags{1, 5, 10};
    MatrixXd Y = MatrixXd::Random(300, 3000);
    MatrixXb omega = Eigen::MatrixXb::Ones(Y.rows(), Y.cols());
    Regularizer opts{lags, 1.0, 1.0, 1.0, 1.0};
    auto lat_dim = 50;

    for (int i = 10; i < Y.rows(); i++) {
        for (int j = 0; j < 10; j++) {
            Y.row(i) += double(rand() % 20 + 200)/ 20 * Y.row(j);
        }
    }

    auto [Wt, result] = Init(Y, opts, lat_dim);

    for (int i = 0; i < Y.cols(); i++) {
        for (int j = 0; j < Y.rows() * 0.15; j++) {
            omega(abs(rand())%Y.rows(), i) = false;
        }
    }

    auto before = Loss(Y, omega, opts, result);
    OptimizeByX(Y, omega, result.F, result.X, Wt, result.W, opts.eta, opts.lambdaX);
    auto after = Loss(Y, omega, opts, result);
    EXPECT_LE(after + 1e-4, before);
}
