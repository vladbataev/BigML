#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

Eigen::VectorXd ConjugatedGradient(Eigen::SparseMatrix<double> A, Eigen::VectorXd b, bool check = false) {
    //std::cout << b.rows();
    int n = b.rows();

    Eigen::VectorXd current(n);
    for (int i = 0; i < n; ++i) {
        current[i] = 0.0;
    }

    Eigen::VectorXd r_now = b - A * current;
    Eigen::VectorXd p_now = r_now;
    Eigen::VectorXd r_previous(n);
    Eigen::VectorXd grad(n);
    for (int i = 0; i < n; ++i) {
        double up = r_now.dot(r_now);
        double down = (A * p_now).dot(p_now);
        double alpha = up / down;

        current = current + alpha * p_now;

        r_previous = r_now;
        r_now = r_now - alpha * A * p_now;

        double beta = r_now.dot(r_now) / (r_previous.dot(r_previous));
        p_now = r_now + beta * p_now;
        if (check) {
            grad = A * current - b;
            std::cout << "gradient norm: " <<  grad.dot(grad) << "  ";
            std::cout << "function value: " << 0.5 * current.dot(A * current) - b.dot(current) << '\n';
        }

    }

    return current;
}

void test_conjugated(int max_order = 15) {
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
