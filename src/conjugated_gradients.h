#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

Eigen::VectorXd ConjugatedGradient(Eigen::SparseMatrix<double> A, Eigen::VectorXd b, double epsilon = 1e-10, bool check = false);
