#include <iostream>
#include <Eigen/Dense>
#include <vector>

using Matrix = Eigen::MatrixXd;

Matrix ConstructFeatureMatrix(const Eigen::VectorXd& row, const std::vector<long>& lags, long L) {
    long T = row.size();
    long m = L + 1;
    long num_lags = lags.size();
    Matrix feature_matrix = Matrix::Zero(T - m, num_lags);
    for (long t = m; t < T; ++t) {
        for (long j = 0; j < num_lags; ++j) {
            feature_matrix(t - m, j) = row(m - lags[j]);
        }
    }
    return feature_matrix;
}

Matrix OptimizeByW(const Matrix& X, const std::vector<long>& lags, double lambda_W, double lambda_X) {
    long L = *std::max_element(lags.begin(), lags.end());
    long m = L + 1;
    long k = X.rows();
    long T = X.cols();
    long num_lags = lags.size();

    Matrix optimal_W = Matrix::Zero(k, num_lags);
    for (long r = 0; r < k; ++r) {
        Matrix feature_matrix = ConstructFeatureMatrix(X.row(r), lags, L);
        Eigen::VectorXd target = Eigen::VectorXd::Zero(T - m);
        for (long t = m; t < T; ++t) {
            target[t - m] = X(r, t);
        }
        Eigen::LLT<Matrix> llt;
        llt.compute(feature_matrix.transpose() * feature_matrix +
                            lambda_W / lambda_X * Matrix::Identity(num_lags ,num_lags));
        Eigen::VectorXd w_r = llt.solve(feature_matrix.transpose() * target);
        optimal_W.row(r) = w_r;
    }
    return optimal_W;
}
