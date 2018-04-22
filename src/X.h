#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#include <vector>
#include <set>

class CachedWTransform {
public:
    CachedWTransform(std::vector<int>);

    Eigen::SparseMatrix<double> operator() (size_t T, const Eigen::VectorXd& w) const;

private:
    std::vector<std::pair<int, int>> diffs;
    std::vector<int> lags;
    std::set<int> diff_set;
    int m;
};

Eigen::MatrixXd optimize_x(
    const Eigen::MatrixXd& Y,
    const Eigen::MatrixXd& F,
    const CachedWTransform& transform,
    const std::vector<Eigen::MatrixXd>& W,
    double nu
);
