#include "X.h"

#include <eigen3/Eigen/Sparse>

#include <algorithm>
#include <set>

using namespace Eigen;
using namespace std;

CachedWTransform::CachedWTransform(std::vector<int> lags)
    : lags(lags)
{
    m = 0;
    for (auto l: lags) {
        m = max(l, m);
        for (auto r: lags) {
            diff_set.insert(abs(l - r));
        }
    }
    auto has = [&](int l) {
        auto it = std::lower_bound(lags.begin(), lags.end(), l);
        return (it != lags.end() && *it == l);
    };

    for (auto d: diff_set) {
        for (auto l: lags) {
            if (has(l - d)) {
                diffs.push_back({l, d});
            }
        }
    }
}

Eigen::SparseMatrix<double> CachedWTransform::operator() (size_t T, const VectorXd& w) const {
    SparseMatrix<double> Lh(T, T);

    SparseMatrix<double> D;
    for (auto l: lags) {
        for (size_t i = m; i + l <= T; i++) {
            Lh.coeffRef(i, i) += w[l] * w[l] / 2;
        }
    }

    for (auto [l, d]: diffs) {
        for (size_t t = m; t < T; t++) {
            auto temp = w[l] * w[l - d];
            Lh.coeffRef(t, t + d) -= temp;
            Lh.coeffRef(t, t) += temp;
            Lh.coeffRef(t + d, t + d) += temp;
        }
    }
    return Lh;
}

MatrixXd optimize_x(
    const MatrixXd& Y,
    const MatrixXd& F,
    const CachedWTransform& transform,
    MatrixXd& X,
    const MatrixXd& W,
    double nu)
{
    auto T = Y.cols();
    auto n = Y.rows();
    auto k = F.cols();
    for (int i = 0; i < k; i++) {
        MatrixXd mY = Y; 
        auto Lh = transform(T, W.row(i));
        Lh *= 2 * nu;
        for (int j = 0; j < k; j++) {
            if (j != i) {
                for (int l = 0; l < T; l++) {
                    mY(i, l) -= F(l, j) * X(j, l);
                }
            }

        }
    }
}
