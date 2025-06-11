// new_ic.h
#ifndef NEW_IC_H
#define NEW_IC_H

#include <Eigen/Dense>

void new_ic(Eigen::MatrixXd& u, Eigen::MatrixXd& v, 
            Eigen::MatrixXd& rho, Eigen::MatrixXd& p,
            const Eigen::VectorXd& x, const Eigen::VectorXd& y);

#endif // NEW_IC_H