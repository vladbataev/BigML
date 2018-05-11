#ifndef PROJECT_PREDICT_H
#define PROJECT_PREDICT_H

#include "factor.h"
#include <Eigen/Dense>


Eigen::VectorXd Predict(const Factorization& factor, const std::vector<int>& lags);

#endif //PROJECT_PREDICT_H
