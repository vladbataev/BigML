#pragma once

#include <Eigen/Dense>
#include <vector>
#include <fstream>

struct Regularizer {
    std::vector<int> lags;
    double lambdaW;
    double lambdaX;
    double lambdaF;
    double eta;
};

struct Factorization {
    Eigen::MatrixXd W;
    Eigen::MatrixXd F;
    Eigen::MatrixXd X;
};

namespace Eigen {
using MatrixXb = Matrix<bool, Dynamic, Dynamic>;
}

std::tuple<class CachedWTransform, Factorization> Init(const Eigen::MatrixXd& Y,
                                                       const Regularizer& opts,
                                                       size_t lat_dim);

void Step(const Eigen::MatrixXd& Y, const Eigen::MatrixXb& omega,
          const Regularizer& opts, Factorization& result, CachedWTransform& Wt,
          std::ofstream & logs_file, bool logs_enabled = false,
          bool verbose = false, bool verify = false);

double Loss(const Eigen::MatrixXd& Y, const Eigen::MatrixXb& omega,
            const Regularizer& opts, const Factorization& result);

Factorization Factorize(Eigen::MatrixXd Y, Eigen::MatrixXb omega,
                        Regularizer opts, size_t lat_dim, size_t steps,
                        std::ofstream &logs_file,
                        bool verbose = false, bool logs_enabled = false);
