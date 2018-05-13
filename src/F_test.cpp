#include "F.h"
#include "X.h"
#include "factor.h"

#include <gtest/gtest.h>

using namespace Eigen;

TEST(F_component, WOMissing) {
    srand(1231541);
    std::vector<int> lags{1, 5, 10};
    MatrixXd Y = MatrixXd::Random(300, 3000);
    MatrixXb omega = Eigen::MatrixXb::Ones(Y.rows(), Y.cols());
    Regularizer opts{lags, 1.0, 1.0, 1.0, 1.0};
    auto lat_dim = 10;

    for (int i = 10; i < Y.rows(); i++) {
        for (int j = 0; j < 10; j++) {
            Y.row(i) += double(rand() % 20 + 200)/ 20 * Y.row(j);
        }
    }

    auto [Wt, result] = Init(Y, opts, lat_dim);

    EXPECT_LE((OptimizeByFALS(Y, result.X, omega, 3.0) - OptimizeByF(Y, result.X, 3.0)).squaredNorm(), 1e-6);
}

TEST(F_component, Loss) {
    srand(1231541);
    std::vector<int> lags{1, 5, 10};
    MatrixXd Y = MatrixXd::Random(300, 3000);
    MatrixXb omega = Eigen::MatrixXb::Ones(Y.rows(), Y.cols());
    Regularizer opts{lags, 1.0, 1.0, 1.0, 1.0};
    auto lat_dim = 10;

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
    auto F = OptimizeByFALS(Y, result.X, omega, 1.0);
    result.F = F;
    auto after = Loss(Y, omega, opts, result);
    EXPECT_LE(after + 1e-4, before);
}
