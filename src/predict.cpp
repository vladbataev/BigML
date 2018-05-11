#include "predict.h"
#include <iostream>

using namespace Eigen;

VectorXd Predict(const Factorization& factor, const std::vector<int>& lags) {
    int n = factor.F.rows();
    int lat_dim = factor.F.rows();
    int T = factor.X.cols();
    VectorXd prediction(n);
    VectorXd predicted_X(lat_dim);
    std::cout << "Num of rows in X: " << predicted_X.rows() << "\n";
    for (int i = 0; i < lags.size(); ++i) {
        auto W_l = factor.W.col(i).asDiagonal();
        auto x_l = factor.X.col(T - lags[i]);
        predicted_X += W_l * x_l;
    }
    return factor.F.transpose() * predicted_X;
}
