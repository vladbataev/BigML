#include <Eigen/Dense>
#include <vector>

Eigen::MatrixXd OptimizeByW(const Eigen::MatrixXd& X, const std::vector<long>& lags, double lambda_W, double lambda_X);
