#pragma once
#include <Eigen/Dense>

namespace Eigen {
    using MatrixXb = Matrix<bool, Dynamic, Dynamic>;
}

Eigen::MatrixXd OptimizeByF(const Eigen::MatrixXd& Y, const Eigen::MatrixXd& X,
                            double lambda);

Eigen::MatrixXd OptimizeByFALS(const Eigen::MatrixXd& Y, const Eigen::MatrixXd& X,
                               const Eigen::MatrixXb& omega, double lambda);
