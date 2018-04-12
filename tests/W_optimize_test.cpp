#include <iostream>
#include "../src/W_optimize.h"

int main()
{
    long T = 512;
    long k = 4;
    Matrix X = Matrix::Random(k, T);
    std::vector<long> lags;
    long L = 30;
    for (long i = 0; i < L; ++i) {
        lags.push_back(i);
    }
    auto result = OptimizeByW(X, lags, 0.5, 2);
}
