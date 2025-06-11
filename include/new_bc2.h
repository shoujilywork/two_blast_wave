// new_bc2.h
#ifndef NEW_BC2_H
#define NEW_BC2_H

#include <Eigen/Dense>

void new_bc2(Eigen::MatrixXd& u, Eigen::MatrixXd& v, 
             Eigen::MatrixXd& rho, Eigen::MatrixXd& p,
             const Eigen::VectorXd& x, const Eigen::VectorXd& y);

#endif // NEW_BC2_H