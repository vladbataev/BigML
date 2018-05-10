#include <vector>
#include <Eigen/Dense>

struct Regularizer {
    std::vector<int> lags;
    double lambdaW;
};

struct Factorization {
};

Factorization Factorize(Eigen::MatrixXd X, Regularizer opts, size_t steps);
