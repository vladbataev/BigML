#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

Eigen::VectorXd ConjugatedGradient(Eigen::SparseMatrix<double> A, Eigen::VectorXd b, double epsilon = 1e-10, bool check = false);
