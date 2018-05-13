#include "W.h"
#include <iostream>

using namespace Eigen;

MatrixXd ConstructFeatureMatrix(const VectorXd& row,
                                const std::vector<int>& lags, long L) {
    long T = row.size();
    long m = L;
    long num_lags = lags.size();
    MatrixXd feature_matrix = MatrixXd::Zero(T - m, num_lags);
    for (long t = m; t < T; ++t) {
        for (long j = 0; j < num_lags; ++j) {
            feature_matrix(t - m, j) = row(t - lags[j]);
        }
    }
    return feature_matrix;
}

MatrixXd OptimizeByW(const MatrixXd& X, const std::vector<int>& lags,
                     double lambda_W, double lambda_X) {
    long L = *std::max_element(lags.begin(), lags.end());
    long m = L;
    long k = X.rows();
    long T = X.cols();
    long num_lags = lags.size();

    MatrixXd optimal_W = MatrixXd::Zero(k, num_lags);
    for (long r = 0; r < k; ++r) {
        Matrix feature_matrix = ConstructFeatureMatrix(X.row(r), lags, L);
        VectorXd target = VectorXd::Zero(T - m);
        for (long t = m; t < T; ++t) {
            target[t - m] = X(r, t);
        }
        LLT<MatrixXd> llt;
        llt.compute(feature_matrix.transpose() * feature_matrix +
                    2 * lambda_W / lambda_X *
                        MatrixXd::Identity(num_lags, num_lags));
        VectorXd w_r = llt.solve(feature_matrix.transpose() * target);
        optimal_W.row(r) = w_r;
    }
    return optimal_W;
}
