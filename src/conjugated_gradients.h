#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

Eigen::VectorXd ConjugatedGradient(Eigen::SparseMatrix<double> A, Eigen::VectorXd b, bool check = false);
