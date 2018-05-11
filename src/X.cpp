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
    double wsum = -1;
    for (int i = 0; i < lags.size(); i++) {
        wsum += w[i];
    }
    for (int j = 0; j < T; j++) {
        Lh.coeffRef(j, j) -= wsum;
    }
    for (int i = 0; i < lags.size(); i++) {
        for (int j = max(0, m - lags[i]); j + lags[i] < T; j++) {
            Lh.coeffRef(j, j) += wsum * w[i] / 2;
        }
    }

    for (auto [lmd_i, d, l_i, l]: diffs) {
        for (size_t t = max(0, m - l); t + l < T; t++) {
            auto temp = w[l_i] * w[lmd_i];
            Lh.coeffRef(t, t + d) -= temp;
            Lh.coeffRef(t + d, t) -= temp;
            Lh.coeffRef(t, t) += temp;
            Lh.coeffRef(t + d, t + d) += temp;
        }
    }

    return Lh;
}

void optimize_X(
    const MatrixXd& Y,
    const Eigen::MatrixXb& Sigma,
    const MatrixXd& F,
    MatrixXd& X,
    const CachedWTransform& transform,
    const MatrixXd& W,
    double nu,
    double lambdaX,
    bool verify)
{
    auto T = Y.cols();
    auto k = F.rows();
    auto n = F.cols();
    SparseMatrix<double> It(T, T); It.setIdentity();

    for (int i = 0; i < k; i++) {
        MatrixXd mY = Y - F.transpose() * X + F.row(i).transpose() * X.row(i);

        SparseMatrix<double> B(T, T);
        for (int l = 0; l < F.cols(); l++) {
            for (int j = 0; j < T; j++) {
                if (Sigma(l, j)) {
                    B.coeffRef(j, j) += F(i, l) * F(i, l);
                }
            }
        }

        SparseMatrix<double> M = (transform(T, W.row(i)) + (nu/2) * It) * lambdaX + B;

        if (verify) {
            assert(F != MatrixXd::Zero(F.rows(), F.cols()));
            assert(W != MatrixXd::Zero(W.rows(), W.cols()));
            Eigen::LLT<MatrixXd> llt(M);
            auto evals = MatrixXd(M).eigenvalues();
            std::cerr << "\nEigen values: \n" << evals << std::endl;
            assert(llt.info() != Eigen::NumericalIssue);
        }

        Eigen::ConjugateGradient<SparseMatrix<double>, Lower|Upper> cg;
        cg.compute(M);
        X.row(i) = cg.solve(mY.transpose() * F.row(i).transpose());
        //X.row(i) = ConjugatedGradient(M, mY.transpose() * F.row(i).transpose());
    }
}
