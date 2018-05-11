#include <iostream>
#include "conjugated_gradients.h"

Eigen::VectorXd ConjugatedGradient(Eigen::SparseMatrix<double> A, Eigen::VectorXd b, double epsilon, bool check) {
    int n = b.rows();

    Eigen::VectorXd current(n);
    for (int i = 0; i < n; ++i) {
        current[i] = 0.0;
    }

    Eigen::VectorXd r_now = b - A * current - epsilon * current;
    Eigen::VectorXd p_now = r_now;
    Eigen::VectorXd r_previous(n);
    Eigen::VectorXd grad(n);
    for (int i = 0; i < n; ++i) {
        double up = r_now.dot(r_now);
        double down = (A * p_now).dot(p_now) + epsilon * p_now.dot(p_now);
        double alpha = up / down;

        current = current + alpha * p_now;

        r_previous = r_now;
        r_now = r_now - alpha * A * p_now - alpha * epsilon * p_now;

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
