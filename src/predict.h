#pragma once

#include "factor.h"
#include <Eigen/Dense>


Eigen::VectorXd Predict(const Factorization& factor, const std::vector<int>& lags);
