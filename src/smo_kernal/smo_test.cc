#include "smo_solver.h"
#include <Eigen/Core>
#include <iostream>


int main() {
    Matrix X(6,2);
    Array y(6,1);
    std::string kernal = "linear";
    double C = 1.0;
    double tol = 1e-6;
    int max_passes = 100;
    double sigma = 1.0;
    double epsilon = 0.01;
    bool verbose = false;

    X << 0, 1, 1, 2, 0, 0, 1, 1, -1,1,1,-1;
    y << 1, 1, -1, -1, 1, -1;

    auto ret = solve(X, y, kernal, C, tol, max_passes);
    std::cout << "Alphas: " << std::get<0>(ret) << std::endl;
    std::cout << "Support Vectors: " << std::get<1>(ret) << std::endl;
    std::cout << "Bias: " << std::get<2>(ret) << std::endl;
}