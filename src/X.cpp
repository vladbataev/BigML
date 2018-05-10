#include "X.h"
#include "conjugated_gradients.h"

#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>

#include <cassert>

#include <iostream>
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
    double wsum = 0;
    for (int i = 0; i < lags.size(); i++) {
        wsum += w[i];
    }
    for (int i = 0; i < lags.size(); i++) {
        for (int j = m; j + lags[i] < T; j++) {
            Lh.coeffRef(j, j) += wsum * w[i] / 2;
        }
    }

    for (auto [lmd_i, d, l_i, l]: diffs) {
        for (size_t t = m; t + l < T; t++) {
            auto temp = 2 * w[l_i] * w[lmd_i];
            Lh.coeffRef(t, t + d) -= temp;
            Lh.coeffRef(t, t) += temp;
            Lh.coeffRef(t + d, t + d) += temp;
        }
    }

#ifndef NDEBUG
    std::cout <<"\nLh: \n" << Lh << "\n";
#endif
    return Lh;
}

void optimize_X(
    const MatrixXd& Y,
    const MatrixXd& F,
    MatrixXd& X,
    const CachedWTransform& transform,
    const MatrixXd& W,
    double nu,
    double lambdaX,
    double tolerance)
{
    auto T = Y.cols();
    auto k = F.rows();
    for (int i = 0; i < k; i++) {
        auto f_norm = F.row(i).squaredNorm();
        if (f_norm < tolerance) {
            continue;
        }
        MatrixXd mY = Y - F.transpose() * X + F.row(i).transpose() * X.row(i);

        SparseMatrix<double> It(T, T); It.setIdentity();
        SparseMatrix<double> Lh = (transform(T, W.row(i)) + (f_norm + nu/2) * It) * lambdaX;

#ifndef NDEBUG
        assert(F != MatrixXd::Zero(F.rows(), F.cols()));
        assert(W != MatrixXd::Zero(W.rows(), W.cols()));
        Eigen::LLT<MatrixXd> llt(Lh); // compute the Cholesky decomposition of A
        auto evals = MatrixXd(Lh).eigenvalues();
        std::cout << "\nEigen values: \n" << evals << std::endl;
        assert(llt.info() != Eigen::NumericalIssue);
#endif

        X.row(i) = ConjugatedGradient(Lh, mY.transpose() * F.row(i).transpose());
    }
}
