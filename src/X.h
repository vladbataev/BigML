#pragma once

#include "factor.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <vector>
#include <set>

class CachedWTransform {
public:
    CachedWTransform(std::vector<int>);

    Eigen::SparseMatrix<double> operator() (size_t T, const Eigen::VectorXd& w) const;

private:
    std::vector<std::tuple<int, int, int, int>> diffs;
    std::vector<int> lags;
    int m;
};

void optimize_X(
    const Eigen::MatrixXd& Y,
    const Eigen::MatrixXb& Sigma,
    const Eigen::MatrixXd& F,
    Eigen::MatrixXd& X,
    const CachedWTransform& transform,
    const Eigen::MatrixXd& W,
    double nu,
    double lambdaX,
    double tolerance=1e-6
);
