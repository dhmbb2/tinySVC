#include <Eigen/Core>
#include <tuple>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include "smo_solver.h"

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Array<double, Eigen::Dynamic, 1> Array;

std::tuple<Array, Matrix, double>
solve(
    const Matrix &X, 
    const Array &y, 
    std::string kernal, 
    double C, 
    double tol, 
    int max_passes
) {
    srand(static_cast<unsigned int>(time(0)));
    int passes = 0;
    int num_changed_alphas = 0;
    int num_samples = X.rows();
    Array alphas = Array::Zero(num_samples, 1);
    double b = 0;
    int j = 0;
    Matrix K = calculate_K(X, kernal);

    double Ei = 0;
    double Ej = 0;
    while(passes < max_passes) {
        num_changed_alphas = 0;
        for (int i = 0; i < num_samples; i++) {
            Ei = calculate_E(alphas, y, K, b, i);
            if (!((y(i) * Ei < -tol && alphas(i) < C) || (y(i) * Ei > tol && alphas(i) > 0)))
                continue;
            j = get_j(i, num_samples);
            Ej = calculate_E(alphas, y, K, b, j);

            double L = 0;
            double H = 0;
            if (y(i) != y(j)) {
                L = std::max(0.0, alphas(j) - alphas(i));
                H = std::min(C, C + alphas(j) - alphas(i));
            } else {
                L = std::max(0.0, alphas(j) + alphas(i) - C);
                H = std::min(C, alphas(j) + alphas(i));
            }
            if (L == H) {
                continue;
            }

            double eta = 2 * K(i, j) - K(i, i) - K(j, j);
            if (eta >= 0) {
                continue;
            }
            double alpha_j_new = alphas(j) - y(j) * (Ei - Ej) / eta;
            if (alpha_j_new > H) {
                alpha_j_new = H;
            } else if (alpha_j_new < L) {
                alpha_j_new = L;
            }

            if (std::abs(alpha_j_new - alphas(j)) < 1e-5) {
                continue;
            }
            double alpha_i_new = alphas(i) + y(i) * y(j) * (alphas(j) - alpha_j_new);
            double b1 = b - Ei - y(i) * (alpha_i_new - alphas(i)) * K(i, i) - y(j) * (alpha_j_new - alphas(j)) * K(i, j);
            double b2 = b - Ej - y(i) * (alpha_i_new - alphas(i)) * K(i, j) - y(j) * (alpha_j_new - alphas(j)) * K(j, j);
            
            if (0 < alpha_i_new && alpha_i_new < C) {
                b = b1;
            } else if (0 < alpha_j_new && alpha_j_new < C) {
                b = b2;
            } else {
                b = (b1 + b2) / 2;
            }
            alphas(i) = alpha_i_new;
            alphas(j) = alpha_j_new;
            num_changed_alphas++;
        }
        if (num_changed_alphas == 0) {
            passes++;
        } else {
            passes = 0;
        }
    }
    std::vector<int> support_idx = get_support_idx(alphas);
    Array supports = Array::Zero(support_idx.size(), 1);
    Matrix support_vector = Matrix::Zero(support_idx.size(), X.cols());
    for (int i = 0; i < support_idx.size(); i++) {
        supports(i) = alphas(support_idx[i]) * y[support_idx[i]];
        support_vector.row(i) = X.row(support_idx[i]);
    }

    return std::make_tuple(supports, support_vector, b);
}


// Helper functions
std::vector<int>
get_support_idx(const Array& alphas) {
    std::vector<int> support_idx;
    for (int i = 0; i < alphas.size(); i++) {
        if (alphas(i) > 1e-6) {
            support_idx.push_back(i);
        }
    }
    return std::move(support_idx);
}

double calculate_E(const Array& alphas, const Array& y, const Matrix& K, double b, int i) {
    return ((alphas * y) * K.col(i).array()).sum() + b - y(i);
}

int get_j(int i, int num_samples) {
    int j = i;
    while(j == i) {
        j = rand() % num_samples;
    }
    return j;
}

Matrix calculate_K(const Matrix &X, std::string kernal) {
    return std::move(X * X.transpose());
}
