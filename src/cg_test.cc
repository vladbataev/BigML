#include "gtest/gtest.h"
#include "conjugated_gradients.h"
#include <iostream>

namespace {
TEST(ConjugatedGradient, zero_grad_at_the_end) {
    int MAX_ORDER = 50;
    double epsilon = 1e-8;
    int N_REPETITIONS = 10;
    for (int repetition_num = 0; repetition_num < N_REPETITIONS; ++repetition_num) {
        for (int n = 1; n <= MAX_ORDER; ++n) {
            Eigen::MatrixXd initial_dense = Eigen::MatrixXd::Random(n,n);
            initial_dense = initial_dense * (Eigen::Transpose<Eigen::MatrixXd>(initial_dense));
            for (int r = 0; r < initial_dense.rows(); ++ r) {
                initial_dense(r, r) += 1.0; //for strictly positive definition
            }

            Eigen::SparseMatrix<double> A = initial_dense.sparseView();
            Eigen::VectorXd b = Eigen::VectorXd::Random(n);
            Eigen::VectorXd result = ConjugatedGradient(A, b);
            Eigen::VectorXd grad = A * result - b;
            double average_swing = grad.dot(grad) / n;
            ASSERT_TRUE(average_swing < epsilon) << average_swing << ' ' << n;
        }
    }
}

TEST(ConjugatedGradient, matching_with_default) {
    int MAX_ORDER = 50;
    int N_REPETITIONS = 10;
    double epsilon = 1e-8;
    for (int repetition_num = 0; repetition_num < N_REPETITIONS; ++repetition_num) {
        for (int n = 1; n <= MAX_ORDER; ++n) {
            Eigen::MatrixXd initial_dense = Eigen::MatrixXd::Random(n,n);
            initial_dense = initial_dense * (Eigen::Transpose<Eigen::MatrixXd>(initial_dense));
            for (int r = 0; r < initial_dense.rows(); ++ r) {
                initial_dense(r, r) += 1.0; //for strictly positive definition
            }
            Eigen::SparseMatrix<double> A = initial_dense.sparseView();
            Eigen::VectorXd b = Eigen::VectorXd::Random(n);

            Eigen::VectorXd result = ConjugatedGradient(A, b);
            double function_value = 0.5 * result.dot(A * result) - b.dot(result);

            Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> cg;
            cg.compute(A);
            Eigen::VectorXd default_result = cg.solve(b);
            double default_function_value = 0.5 * default_result.dot(A * default_result) - b.dot(default_result);
            double delta = fabs(function_value - default_function_value);

            ASSERT_TRUE(delta < epsilon) << delta << ' ' << n;

        }
    }
}
}
