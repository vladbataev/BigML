#pragma once
#include <Eigen/Dense>

Eigen::MatrixXd OptimizeByF(Eigen::MatrixXd Y, Eigen::MatrixXd X, double lambda);
