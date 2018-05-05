#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

Eigen::VectorXd ConjugatedGradient(Eigen::SparseMatrix<double> A, Eigen::VectorXd b, bool check = false);
