#pragma once
#include <Eigen/Core>
#include <tuple>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Array<double, Eigen::Dynamic, 1> Array;

std::tuple<Array, Matrix, double> solve(const Matrix&, const Array& , std::string, double, double, int );
std::vector<int> get_support_idx(const Array&) ;
double calculate_E(const Array&, const Array&, const Matrix&, double, int);
int get_j(int, int);
Matrix calculate_K(const Matrix&, std::string);