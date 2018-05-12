#pragma once

#include "factor.h"
#include <Eigen/Dense>


Eigen::MatrixXd Predict(
        const Factorization& factor,
        const std::vector<int>& lags,
        long t_start, long t_end
);
