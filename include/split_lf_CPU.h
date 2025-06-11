#ifndef SPLIT_LF_H
#define SPLIT_LF_H

#include <Eigen/Dense>

void split_lf(Eigen::MatrixXd& r, Eigen::MatrixXd& p, 
              Eigen::MatrixXd& u, Eigen::MatrixXd& v, 
              Eigen::MatrixXd& E,
              Eigen::MatrixXd& fp1, Eigen::MatrixXd& fp2, 
              Eigen::MatrixXd& fp3, Eigen::MatrixXd& fp4,
              Eigen::MatrixXd& fn1, Eigen::MatrixXd& fn2, 
              Eigen::MatrixXd& fn3, Eigen::MatrixXd& fn4,
              Eigen::MatrixXd& gp1, Eigen::MatrixXd& gp2, 
              Eigen::MatrixXd& gp3, Eigen::MatrixXd& gp4,
              Eigen::MatrixXd& gn1, Eigen::MatrixXd& gn2, 
              Eigen::MatrixXd& gn3, Eigen::MatrixXd& gn4);

#endif // SPLIT_LF_H