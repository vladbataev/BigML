#pragma once

#include "factor.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <set>
#include <vector>

class CachedWTransform {
   public:
    CachedWTransform(std::vector<int>);

    Eigen::SparseMatrix<double> operator()(size_t T,
                                           const Eigen::VectorXd& w) const;

   private:
    std::vector<std::tuple<int, int, int, int>> diffs_;
    std::vector<int> lags_;
    int m_;
};

void OptimizeByX(const Eigen::MatrixXd& Y, const Eigen::MatrixXb& omega,
                 const Eigen::MatrixXd& F, Eigen::MatrixXd& X,
                 const CachedWTransform& transform, const Eigen::MatrixXd& W,
                 double eta, double lambdaX, bool verify = false);
