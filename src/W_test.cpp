#include "W.h"
#include "X.h"
#include "factor.h"
#include "F.h"

#include <gtest/gtest.h>


using namespace Eigen;


TEST(W_component, Shape) {
    long T = 512;
    long k = 4;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(k, T);
    std::vector<int> lags;
    long L = 30;
    for (long i = 0; i < L; ++i) {
        lags.push_back(i);
    }
    auto result = OptimizeByW(X, lags, 0.5, 2);
}

TEST(W_component, Loss) {
    srand(1231541);
    std::vector<int> lags{1, 5, 10};
    MatrixXd Y = MatrixXd::Random(300, 3000);
    MatrixXb omega = Eigen::MatrixXb::Ones(Y.rows(), Y.cols());
    Regularizer opts{lags, 1.0, 1.0, 1.0, 1.0};
    auto lat_dim = 10;

    for (int i = 10; i < Y.rows(); i++) {
        for (int j = 0; j < 10; j++) {
            Y.row(i) += double(rand() % 20 + 200) / 20 * Y.row(j);
        }
    }

    auto [Wt, result] = Init(Y, opts, lat_dim);

    for (int i = 0; i < Y.cols(); i++) {
        for (int j = 0; j < Y.rows() * 0.15; j++) {
            omega(abs(rand())%Y.rows(), i) = false;
        }
    }

    auto before = Loss(Y, omega, opts, result);
    auto W = OptimizeByW(Y, lags, 1.0, 1.0);
    result.W = W;
    auto after = Loss(Y, omega, opts, result);
    EXPECT_LE(after + 1e-4, before);
}
