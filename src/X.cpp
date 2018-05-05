#include "X.h"
#include "conjugated_gradients.h"

#include <Eigen/Sparse>

#include <cassert>

#include <algorithm>
#include <set>

using namespace Eigen;
using namespace std;

CachedWTransform::CachedWTransform(std::vector<int> lags)
    : lags(lags)
{
    m = 0;
    map<int, int> lag_num;
    std::set<int> diff_set;
    for (int i = 0; i < lags.size(); i++) {
        m = max(lags[i], m);
        lag_num[lags[i]] = i;
    }
    for (auto l: lags) {
        for (auto r: lags) {
            diff_set.insert(abs(l - r));
        }
    }
    for (auto d: diff_set) {
        for (int i = 0; i < lags.size(); i++) {
            auto it = lag_num.find(lags[i] - d);
            if (it != lag_num.end()) {
                diffs.push_back({it->second, d, i, lags[i]});
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

    for (auto [lmd_i, d, l_i, l]: diffs) {
        for (size_t t = m; t + l < T; t++) {
            auto temp = w[l_i] * w[lmd_i];
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
        SparseMatrix<double> It(T, T); It.setIdentity();
        SparseMatrix<double> Lh = transform(T, W.row(i)) * nu + F.row(i).squaredNorm() * It;
        mY += F.row(i).transpose() * X.row(i);
        X.row(i) = ConjugatedGradient(Lh, Y.transpose() * F.row(i).transpose());
    }
}
