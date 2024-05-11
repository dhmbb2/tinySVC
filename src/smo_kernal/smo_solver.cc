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
    int max_passes,
    bool heu
) {
    srand(static_cast<unsigned int>(time(0)));
    Matrix K = calculate_K(X, kernal);
    solverInfo info(X, y, K, C, tol, X.rows(), max_passes);
    if (heu)
        return solve_heuristic(info);
    else 
        return solve_simple(info);
}

std::tuple<Array, Matrix, double>
solve_simple(solverInfo& info) {
    int passes = 0;
    int num_changed_alphas = 0;
    while(passes < info.max_passes) {
        num_changed_alphas = 0;
        for (int i = 0; i < info.num_samples; ++i) {
            num_changed_alphas += j_loop(info, i, false);
        }

        if (num_changed_alphas == 0) {
            passes++;
        } else {
            passes = 0;
        }
    }

    return get_ret(info);
}

std::tuple<Array, Matrix, double>
solve_heuristic(solverInfo& info) {
    int passes = 0;
    int num_changed_alphas = 0;
    bool iterate_whole_set = 1;

    while((passes < info.max_passes) && (num_changed_alphas > 0 || iterate_whole_set)) {
        num_changed_alphas = 0;
        if (iterate_whole_set) {
            for (int i = 0; i < info.num_samples; i++)
                num_changed_alphas += j_loop(info, i, true);
            std::cout << "num_changed_alphas: " << num_changed_alphas << '\n';
            passes++;
        } else {
            auto supporting_vecs_idx = get_support_idx(info);
            for (int idx : supporting_vecs_idx)
                num_changed_alphas += j_loop(info, idx, true);
            std::cout << "num_changed_alphas: " << num_changed_alphas << '\n';
            passes++;
        }

        if (iterate_whole_set) 
            iterate_whole_set = false;
        else if (num_changed_alphas == 0)
            iterate_whole_set = true;
    }
    std::cout << passes << '\n';

    return get_ret(info);
}

std::tuple<Array, Matrix, double>
get_ret(solverInfo& info) {
    std::vector<int> support_idx = get_support_idx(info);
    Array supports = Array::Zero(support_idx.size(), 1);
    Matrix support_vector = Matrix::Zero(support_idx.size(), info.X.cols());
    for (int i = 0; i < support_idx.size(); i++) {
        supports(i) = info.alphas(support_idx[i]) * info.y[support_idx[i]];
        support_vector.row(i) = info.X.row(support_idx[i]);
    }
    return std::make_tuple(supports, support_vector, info.b);
}

int j_loop(solverInfo& info, int i, bool heu) {
    double Ei = calculate_E(info, i);
    if (!((info.y(i) * Ei < -info.tol && info.alphas(i) < info.C) || (info.y(i) * Ei > info.tol && info.alphas(i) > 0)))
        return 0;
    int j = get_j(info, i, heu, Ei);
    double Ej = calculate_E(info, j);

    double L = 0;
    double H = 0;
    if (info.y(i) != info.y(j)) {
        L = std::max(0.0, info.alphas(j) - info.alphas(i));
        H = std::min(info.C, info.C + info.alphas(j) - info.alphas(i));
    } else {
        L = std::max(0.0, info.alphas(j) + info.alphas(i) - info.C);
        H = std::min(info.C, info.alphas(j) + info.alphas(i));
    }
    if (L == H) {
        return 0;
    }

    double eta = 2 * info.K(i, j) - info.K(i, i) - info.K(j, j);
    if (eta >= 0) {
        return 0;
    }
    double alpha_j_new = info.alphas(j) - info.y(j) * (Ei - Ej) / eta;
    if (alpha_j_new > H) {
        alpha_j_new = H;
    } else if (alpha_j_new < L) {
        alpha_j_new = L;
    }

    if (std::abs(alpha_j_new - info.alphas(j)) < 1e-5) {
        return 0;
    }
    double alpha_i_new = info.alphas(i) + info.y(i) * info.y(j) * (info.alphas(j) - alpha_j_new);
    double b1 = info.b - Ei - info.y(i) * (alpha_i_new - info.alphas(i)) * info.K(i, i) - info.y(j) * (alpha_j_new - info.alphas(j)) * info.K(i, j);
    double b2 = info.b - Ej - info.y(i) * (alpha_i_new - info.alphas(i)) * info.K(i, j) - info.y(j) * (alpha_j_new - info.alphas(j)) * info.K(j, j);

    if (0 < alpha_i_new && alpha_i_new < info.C) {
        info.b = b1;
    } else if (0 < alpha_j_new && alpha_j_new < info.C) {
        info.b = b2;
    } else {
        info.b = (b1 + b2) / 2;
    }

    info.alphas(i) = alpha_i_new;
    info.alphas(j) = alpha_j_new;
    update_Ecache(info, i);
    update_Ecache(info, j);
    return 1;
}

// Helper functions
std::vector<int>
get_support_idx(solverInfo& info) {
    std::vector<int> support_idx;
    for (int i = 0; i < info.alphas.size(); i++) {
        if (info.alphas(i) > 0 && info.alphas(i) < info.C) {
            support_idx.push_back(i);
        }
    }
    return std::move(support_idx);
}

double calculate_E(solverInfo& info, int i) {
    return ((info.alphas * info.y) * info.K.col(i).array()).sum() + info.b - info.y(i);
}

void update_Ecache(solverInfo& info, int i) {
    double Ei = calculate_E(info, i);
    info.E_cache[i] = std::make_tuple(Ei, i);
}

int get_j(solverInfo& info, int i, bool heu, double Ei) {
    int j = i;
    if (!heu) {
        while (j == i)
            j = rand() % info.num_samples;
        return j;
    }

    // iterate over E_cache to find validate E 
    // to serve as the pool for choosing j
    info.E_cache[i] = std::make_tuple(Ei, true);
    auto pool = std::vector<int>();
    for (int k = 0; k < info.num_samples; k++) {
        if (!std::get<1>(info.E_cache[k]) || k == i) {
            continue;
        }
        pool.push_back(k);
    }
    // if there is none other validate E,
    //  just randomly pick a j
    if (pool.size() == 1) {
        while (j == i) 
            j = rand() % info.num_samples;
    } else {
        // find the j that maximize |Ei - Ej|
        double max_diff = 0;
        for (auto idx : pool) {
            double Ej = std::get<0>(info.E_cache[idx]);
            // double Ej = calculate_E(info, idx);
            double diff = std::abs(Ei - Ej);
            if (diff > max_diff) {
                max_diff = diff;
                j = idx;
            }
        }
    }
    return j;
}

Matrix calculate_K(const Matrix &X, std::string kernal) {
    return std::move(X * X.transpose());
}
