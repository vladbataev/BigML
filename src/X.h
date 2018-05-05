#pragma once

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
    const Eigen::MatrixXd& F,
    Eigen::MatrixXd& X,
    const CachedWTransform& transform,
    const Eigen::MatrixXd& W,
    double nu
);
