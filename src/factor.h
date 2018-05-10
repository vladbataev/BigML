#include <vector>
#include <Eigen/Dense>

struct Regularizer {
    std::vector<int> lags;
    double lambdaW;
    double lambdaX;
    double lambdaF;
    double nu;
};

struct Factorization {
    Eigen::MatrixXd W;
    Eigen::MatrixXd F;
    Eigen::MatrixXd X;
};

std::tuple<class CachedWTransform, Factorization> Init(const Eigen::MatrixXd& Y, const Regularizer& opts, size_t lat_dim);
void Step(const Eigen::MatrixXd& Y, const Regularizer& opts, Factorization& result, CachedWTransform& Wt, double tol=1e-6);

Factorization Factorize(Eigen::MatrixXd Y, Regularizer opts, size_t lat_dim, size_t steps, double tol=1e-6);
