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
                               const Eigen::MatrixXb& omega, double lambda) {
    auto n = Y.rows();
    auto T = Y.cols();
    auto k = X.rows();
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(n, k);

    for (int u = 0; u < n; ++u) {
        auto nullify = [&](auto X) {
            return (X.array().rowwise() * omega.cast<double>().row(u).array()).matrix();
        };
        F.row(u) = (Eigen::MatrixXd::Identity(k, k) * lambda + nullify(X) * X.transpose()).inverse() *
           X * (Y.cwiseProduct(omega.cast<double>())).row(u).transpose();
    }
    return F.transpose();
}
