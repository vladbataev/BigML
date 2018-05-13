#include "X.h"

#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>

#include <cassert>

#include <algorithm>
#include <iostream>
#include <set>

using namespace Eigen;
using namespace std;

CachedWTransform::CachedWTransform(vector<int> lags) : lags_(lags) {
    m_ = 0;
    map<int, int> lag_num;
    set<int> diff_set;
    for (int i = 0; i < lags.size(); i++) {
        m_ = max(lags[i], m_);
        lag_num[lags[i]] = i;
    }
    for (auto l : lags) {
        for (auto r : lags) {
            diff_set.insert(abs(l - r));
        }
    }
    for (auto d : diff_set) {
        for (int i = 0; i < lags.size(); i++) {
            auto it = lag_num.find(lags[i] - d);
            if (it != lag_num.end()) {
                diffs_.push_back({it->second, d, i, lags[i]});
            }
        }
    }
}

SparseMatrix<double> CachedWTransform::operator()(size_t T,
                                                  const VectorXd& w) const {
    VectorXd D = VectorXd::Zero(T);

    double wsum = -1;
    for (int i = 0; i < lags_.size(); i++) {
        wsum += w[i];
    }
    for (int j = 0; j < T; j++) {
        D(j) -= wsum;
    }
    for (int i = 0; i < lags_.size(); i++) {
        for (int j = max(0, m_ - lags_[i]); j + lags_[i] < T; j++) {
            D(j) += wsum * w[i] / 2;
        }
    }

    vector<Triplet<double>> triplets;
    for (auto [lmd_i, d, l_i, l] : diffs_) {
        for (size_t t = max(0, m_ - l); t + l < T; t++) {
            auto temp = w[l_i] * w[lmd_i];
            triplets.push_back({t, t + d, -temp});
            triplets.push_back({t + d, t, -temp});
            D(t) += temp;
            D(t + d) += temp;
        }
    }

    SparseMatrix<double> Lh(T, T);

    for (int i = 0; i < T; i++) {
        triplets.push_back({i, i, D(i)});
    }
    Lh.setFromTriplets(triplets.begin(), triplets.end());
    return Lh;
}

void OptimizeByX(const MatrixXd& Y, const MatrixXb& omega, const MatrixXd& F,
                 MatrixXd& X, const CachedWTransform& transform,
                 const MatrixXd& W, double nu, double lambdaX, bool verify) {
    auto T = Y.cols();
    auto k = F.rows();
    auto n = F.cols();

    for (int i = 0; i < k; i++) {
        MatrixXd mY = Y - F.transpose() * X + F.row(i).transpose() * X.row(i);

        VectorXd B = VectorXd::Zero(T);
        for (int l = 0; l < F.cols(); l++) {
            for (int j = 0; j < T; j++) {
                if (omega(l, j)) {
                    B(j) += F(i, l) * F(i, l);
                }
            }
        }

        SparseMatrix<double> M = transform(T, W.row(i)) * lambdaX;
        M += (nu / 2) * lambdaX * VectorXd::Ones(T).asDiagonal();
        M += B.asDiagonal();

        if (verify) {
            assert(F != MatrixXd::Zero(F.rows(), F.cols()));
            assert(W != MatrixXd::Zero(W.rows(), W.cols()));
            LLT<MatrixXd> llt(M);
            auto evals = MatrixXd(M).eigenvalues();
            cerr << "\nEigen values: \n" << evals << endl;
            assert(llt.info() != NumericalIssue);
        }

        ConjugateGradient<SparseMatrix<double>, Lower, DiagonalPreconditioner<double>> cg;
        cg.setMaxIterations(k);
        cg.compute(M);
        X.row(i) = cg.solve(mY.transpose() * F.row(i).transpose());
    }
}
