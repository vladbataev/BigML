#include "predict.h"
#include <iostream>
#include <math.h>

using namespace Eigen;

MatrixXd Predict(const Factorization& factor, const std::vector<int>& lags,
                  long t_start, long t_end) {
    int n = factor.F.rows();
    int lat_dim = factor.F.rows();
    int T = factor.X.cols();
    MatrixXd predicted_X = MatrixXd::Zero(lat_dim, t_end - T);

    for (int t = T; t < t_end; ++t) {
        for (int i = 0; i < lags.size(); ++i) {
            VectorXd X_col = VectorXd::Zero(lat_dim);
            if (t - lags[i] < T) {
                X_col = factor.X.col(t - lags[i]);
            } else {
                X_col = predicted_X.col(t - T - lags[i]);
            }
            predicted_X.col(t - T)  += factor.W.col(i).asDiagonal() * X_col;
        }
    }
    MatrixXd prediction(n, t_end);
    prediction = factor.F.transpose() * predicted_X.block(0, t_start - T, lat_dim, t_end - t_start);
    return prediction;
}

double RMSE(const Eigen::MatrixXd& true_, const Eigen::MatrixXd& prediction, const Eigen::MatrixXb& test_omega) {
    size_t nnz = test_omega.cast<int>().sum();
    double up = sqrt((true_ - prediction).squaredNorm() / nnz);
    double down = true_.cwiseAbs().sum() / nnz;
    return up / down;
}

double ND(const Eigen::MatrixXd& true_, const Eigen::MatrixXd& prediction, const Eigen::MatrixXb& test_omega) {
    size_t nnz = test_omega.cast<int>().sum();
    double up = (true_ - prediction).cwiseAbs().sum() / nnz;
    double down = true_.cwiseAbs().sum() / nnz;
    return up / down;
}
