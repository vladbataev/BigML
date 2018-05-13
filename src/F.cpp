#include "F.h"
#include <Eigen/Dense>
#include <iostream>

Eigen::MatrixXd OptimizeByF(const Eigen::MatrixXd& Y, const Eigen::MatrixXd& X,
                            double lambda) {
    Eigen::BDCSVD<Eigen::MatrixXd> svd(
        X.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::VectorXd singularValues = svd.singularValues();
    Eigen::VectorXd properValues(singularValues);

    for (int i = 0; i < properValues.rows(); ++i) {
        double now = singularValues(i);
        properValues(i) = now / (now * now + lambda);
    }

    Eigen::DiagonalMatrix<double, Eigen::Dynamic> middle_d(properValues);
    Eigen::MatrixXd middle(middle_d);

    Eigen::MatrixXd proceededMatrix = V * middle * U.transpose();

    return proceededMatrix * Y.transpose();
}


Eigen::MatrixXd OptimizeByFALS(const Eigen::MatrixXd& Y, const Eigen::MatrixXd& X,
                               const Eigen::MatrixXb& omega, double lambda, int num_iterations) {
    auto n = Y.rows();
    auto T = Y.cols();
    auto k = X.rows();
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(n, k);

    for (int j = 0; j < num_iterations; ++j) {
        for (int u = 0; u < n; ++u) {
            Eigen::MatrixXd cum_sum = Eigen::MatrixXd::Identity(k, k) * lambda;
            Eigen::VectorXd mult = Eigen::VectorXd::Zero(k);
            for (int i = 0; i < k; ++i) {
                if (omega(u, i)) {
                    cum_sum += X.col(i) * X.col(i).transpose();
                    mult += Y(u, i) * X.col(i);
                }
            }
            F.row(u) = cum_sum.inverse() * mult;
        }
    }
    return F.transpose();
}
