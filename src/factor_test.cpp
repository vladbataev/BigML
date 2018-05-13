#include <gtest/gtest.h>

#include <cstdlib>

#include "X.h"
#include "W.h"
#include "F.h"
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

    for (size_t i = 0; i < 30; i++) {
        auto check = [&](auto msg) {
            auto after = Loss(Y, omega, opts, result);
            EXPECT_LE(after + 1e-4, loss) << "i=" << i << " " << msg;
            loss = after;
        };
        result.W = OptimizeByW(result.X, opts.lags, opts.lambdaX, opts.lambdaW);
        check("W");
        result.F = OptimizeByFALS(Y, result.X, omega, opts.lambdaF);
        check("F");
        OptimizeByX(Y, omega, result.F, result.X, Wt, result.W, opts.eta,
                    opts.lambdaX, false);
        check("X");
    }
}

TEST(Factor, MissingValues) {
    srand(1231541);

    std::vector<int> lags{1, 5, 10};
    MatrixXd Y = MatrixXd::Random(300, 3000);
    MatrixXb omega = Eigen::MatrixXb::Ones(Y.rows(), Y.cols());
    Regularizer opts{lags, 1.0, 1.0, 1.0, 1.0};
    auto lat_dim = 10;
    auto steps = 100;

    for (int i = 10; i < Y.rows(); i++) {
        for (int j = 0; j < 10; j++) {
            Y.row(i) += double(rand() % 20 + 200)/ 20 * Y.row(j);
        }
    }

    auto [Wt, result] = Init(Y, opts, lat_dim);
    auto loss = Loss(Y, omega, opts, result);

    for (int i = 0; i < Y.cols(); i++) {
        for (int j = 0; j < Y.rows() * 0.15; j++) {
            omega(abs(rand())%Y.rows(), i) = false;
        }
    }

    for (size_t i = 0; i < 30; i++) {
        auto check = [&](auto msg) {
            auto after = Loss(Y, omega, opts, result);
            EXPECT_LE(after + 1e-4, loss) << "i=" << i << " " << msg;
            loss = after;
        };
        result.W = OptimizeByW(result.X, opts.lags, opts.lambdaW, opts.lambdaX);
        check("W");
        result.F = OptimizeByFALS(Y, result.X, omega, opts.lambdaF);
        check("F");
        OptimizeByX(Y, omega, result.F, result.X, Wt, result.W, opts.eta,
                    opts.lambdaX, false);
        check("X");
    }
}
