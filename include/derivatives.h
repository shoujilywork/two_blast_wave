#ifndef DERIVATIVES_H
#define DERIVATIVES_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
using namespace Eigen;

Eigen::MatrixXd dx_p(const Eigen::MatrixXd& u, double h);
Eigen::MatrixXd dx_n(const Eigen::MatrixXd& u, double h);
Eigen::MatrixXd dy_p(const Eigen::MatrixXd& u, double h);
Eigen::MatrixXd dy_n(const Eigen::MatrixXd& u, double h);

#endif