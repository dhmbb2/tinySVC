#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "smo_solver.h"

namespace py = pybind11;

PYBIND11_MODULE( ccsmo, solver ){
    solver.doc() = "cpp inplemented smo solver supported by eigen and pybind11";
    solver.def("solve", &solve, py::return_value_policy::move, "solve the svm problem using smo algorithm");
}