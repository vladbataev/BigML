#include "W.h"
#include "X.h"
#include "F_optimize.h"

using namespace std;
using namespace Eigen;

struct Regularizer {
    vector<int> lags;
    double lambdaW;
};

struct Factorization {
};

Factorization Factorize(MatrixXd X, Regularizer opts, size_t steps) {
    CachedWTransform Wt(opts.lags);
    MatrixXd W(X.rows(), opts.lags.size());
    MatrixXd F;

    for (size_t i = 0; i < steps; i++) {
    }
}


int main() {
    return 0;
}
