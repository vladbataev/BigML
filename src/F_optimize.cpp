#include <iostream>
#include <Eigen/Dense>

Eigen::MatrixXd OptimizeByF(Eigen::MatrixXd Y, Eigen::MatrixXd X, double lambda) {
    Eigen::BDCSVD<Eigen::MatrixXd> svd(Eigen::Transpose<Eigen::MatrixXd>(X),
                                       Eigen::ComputeThinU | Eigen::ComputeFullV);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::VectorXd singularValues = svd.singularValues();
    Eigen::VectorXd properValues(singularValues);

    for (int i = 0; i < properValues.rows(); ++i) {
        double now = singularValues(i);
        properValues(i) = now / (now * now + lambda);
    }


    Eigen::DiagonalMatrix<double, Eigen::Dynamic> middle_d(properValues);
    Eigen::MatrixXd middle(middle_d);


    Eigen::MatrixXd proceededMatrix = V * middle * Eigen::Transpose<Eigen::MatrixXd>(U);

    Eigen::MatrixXd result = proceededMatrix * Eigen::Transpose<Eigen::MatrixXd>(Y);
    return  Eigen::Transpose<Eigen::MatrixXd>(result);

}

