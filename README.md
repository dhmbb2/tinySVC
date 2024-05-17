# TinySVC ![Static Badge](https://img.shields.io/badge/toy-Machine_Learning)

Course project for SJTU Machine Learning. Implement a tiny support vector machine with heutistic parameters picking, OVO and OVR multiclass strategies. All options can be switched on and off for testing. Also Implement a MKL class with MKLGL algorithm for optimal multi-kernel coefficient training.

## Project Structure
![Code Structure]("assets/structure.jpg")

## Run
Just download the code and start running.

A cpp smo solver is also implemented(though relativly inferior in performance due to the deficiency of Eigen to run floating point arithmetics). A dynamic lib is provided in `lib\` which is compiled on Ubuntu 18.04.4 LTS with gcc version 9.4.0.

If you wish to compile yourself, just download Eigen, python3-dev and pybind11.
```bash
sudo apt-get install eigen, python3-dev
pip install pybind11
```
Change your python interpreter directory in `src/smo_kernel/CMakeLists.txt`, enter `build` and run `cmake..; make`.

## Experiment
All experimenting code can be found in `experiment.py`. You may need to download `CIFAR10` adn `MNIST` to see the results.
