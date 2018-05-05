#include "X.h"
#include "conjugated_gradients.h"

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

void optimize_X(
    const MatrixXd& Y,
    const MatrixXd& F,
    MatrixXd& X,
    const CachedWTransform& transform,
    const MatrixXd& W,
    double nu)
{
    auto T = Y.cols();
    auto k = F.cols();
    for (int i = 0; i < k; i++) {
        MatrixXd mY = Y - F.transpose() * X; 
        SparseMatrix<double> It(T, T);
        It.setIdentity();
        auto Lh = transform(T, W.row(i)) * nu + F.row(i).squaredNorm() * It;
        mY += F.row(i).transpose() * X.row(i);
        X.row(i) = ConjugatedGradient(Lh, Y.transpose() * F.row(i));
    }
}
