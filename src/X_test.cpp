#include <gtest/gtest.h>

#include "X.h"

using namespace Eigen;

TEST(X_component, Smoke) {
    MatrixXd Y(2, 7); Y.setZero();
    MatrixXd F(3, 2);
    F <<
        13, 14,
        15, 16,
        17, 18;

    MatrixXd X(3, 7); X.setZero();

    auto t = CachedWTransform({2, 4});

    MatrixXd W(3, 2);
    W <<
        1, 2,
        5, 6,
        9, 10;

    optimize_X(Y, F, X, t, W, 1, 1, 1e-6);
}
