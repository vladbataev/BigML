#include <gtest/gtest.h>

#include "X.h"

using namespace Eigen;

TEST(X_component, Smoke) {
    MatrixXd Y(2, 7); Y.setZero();
    MatrixXd F(3, 2); F.setZero();
    MatrixXd X(3, 7); X.setZero();

    auto t = CachedWTransform({3, 4, 5, 7});

    MatrixXd W(3, 4); W.setZero();

    optimize_X(Y, F, X, t, W, 1);
}

TEST(X_component, PositiveDefinite) {
}

TEST(X_component, Optimization) {
}
