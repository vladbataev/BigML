#include "predict.h"
#include <iostream>

using namespace Eigen;

MatrixXd Predict(const Factorization& factor, const std::vector<int>& lags,
                  long t_start, long t_end) {
    int n = factor.F.rows();
    int lat_dim = factor.F.rows();
    int T = factor.X.cols();
    MatrixXd predicted_X = MatrixXd::Zero(lat_dim, t_end);

    for (int t = T; t < t_end; ++t) {
        for (int i = 0; i < lags.size(); ++i) {
            predicted_X.col(t - T)  += factor.W.col(i).asDiagonal() * factor.X.col(t - lags[i]);
        }
    }
    MatrixXd prediction(n, t_end);
    prediction = factor.F.transpose() * predicted_X.block(0, t_start, lat_dim, t_end - t_start);
    return prediction;
}
