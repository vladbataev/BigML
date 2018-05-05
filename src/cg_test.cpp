#include <gtest/gtest.h>

#include "conjugated_gradients.h"

TEST(Cg, Cg) {
    int max_order = 15;
    for (int n = 1; n <= max_order; ++n) {
        Eigen::MatrixXd initial_dense = Eigen::MatrixXd::Random(n,n);
        initial_dense = initial_dense * (Eigen::Transpose<Eigen::MatrixXd>(initial_dense));
        Eigen::SparseMatrix<double> A = initial_dense.sparseView();
        //Eigen::SparseMatrix<double> A(n, n);
        //A = initial.dot(initial.transpose());
        //A = initial;
        Eigen::VectorXd b = Eigen::VectorXd::Random(n);
        //std::cout << b.rows();
        std::cout << "testing on matrix with size " << n << ":\n";
        ConjugatedGradient(A, b, true);
        std::cout << "\n\n\n";
    }
}

