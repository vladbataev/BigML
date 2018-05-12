#include "W.h"

#include <gtest/gtest.h>


TEST(W_component, Shape) {
    long T = 512;
    long k = 4;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(k, T);
    std::vector<int> lags;
    long L = 30;
    for (long i = 0; i < L; ++i) {
        lags.push_back(i);
    }
    auto result = OptimizeByW(X, lags, 0.5, 2);
}
