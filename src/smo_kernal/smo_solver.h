#pragma once
#include <Eigen/Core>
#include <tuple>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Array<double, Eigen::Dynamic, 1> Array;

struct solverInfo {
    const Matrix& X;
    const Array& y;
    double C;
    double tol;
    int num_samples;
    int max_passes;
    Array alphas;
    std::vector<std::tuple<double, bool>> E_cache;
    Matrix& K;
    double b;

    solverInfo(
        const Matrix& X_, 
        const Array& y_,
        Matrix& K_,
        double C_,
        double tol_,
        int num_samples_,
        int max_passes_
    ): X(X_), y(y_), C(C_), tol(tol_), num_samples(num_samples_),
       alphas(Array::Zero(num_samples, 1)),
       E_cache(std::vector<std::tuple<double, bool>>(num_samples, std::make_tuple(0, false))),
       K(K_),
       b(0), max_passes(max_passes_) {}
};

std::tuple<Array, Matrix, double> solve(const Matrix&, const Array& , Matrix& , double, double, int, bool);
std::tuple<Array, Matrix, double> solve_simple(solverInfo&);
std::tuple<Array, Matrix, double> solve_heuristic(solverInfo &);

std::tuple<Array, Matrix, double> get_ret(solverInfo &);

int j_loop(solverInfo&, int, bool);
std::vector<int> get_support_idx(solverInfo&);
double calculate_E(solverInfo&, int);
int get_j(solverInfo&, int, bool, double);
Matrix calculate_K(const Matrix&, std::string);
void update_Ecache(solverInfo&, int);