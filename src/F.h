#pragma once
#include <Eigen/Dense>

Eigen::MatrixXd OptimizeByF(const Eigen::MatrixXd& Y, const Eigen::MatrixXd& X,
                            double lambda);
