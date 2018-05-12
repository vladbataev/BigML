#pragma once

#include <Eigen/Dense>
#include "factor.h"

Eigen::MatrixXd Predict(const Factorization& factor,
                        const std::vector<int>& lags, long t_start, long t_end);

double RMSE(const Eigen::MatrixXd& true_, const Eigen::MatrixXd& prediction,
            const Eigen::MatrixXb& test_omega);
double ND(const Eigen::MatrixXd& true_, const Eigen::MatrixXd& prediction,
          const Eigen::MatrixXb& test_omega);