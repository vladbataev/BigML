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
