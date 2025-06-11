#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include "../include/split_lf.h"
using namespace Eigen;

void split_lf(MatrixXd& r, MatrixXd& p, MatrixXd& u, MatrixXd& v, MatrixXd& E,
              MatrixXd& fp1, MatrixXd& fp2, MatrixXd& fp3, MatrixXd& fp4,
              MatrixXd& fn1, MatrixXd& fn2, MatrixXd& fn3, MatrixXd& fn4,
              MatrixXd& gp1, MatrixXd& gp2, MatrixXd& gp3, MatrixXd& gp4,
              MatrixXd& gn1, MatrixXd& gn2, MatrixXd& gn3, MatrixXd& gn4)
{
    // 计算声速c
    MatrixXd c = (1.4 * p.array() / r.array()).cwiseMax(0).sqrt();
    
    // 计算特征速度a和b
    double a = (u.array() + c.array()).abs().maxCoeff();
    double b = (v.array() + c.array()).abs().maxCoeff();
    
    // 计算通量f和g
    MatrixXd f1 = r.cwiseProduct(u);
    MatrixXd f2 = r.cwiseProduct(u.array().square().matrix()) + p;
    MatrixXd f3 = r.cwiseProduct(u).cwiseProduct(v);
    MatrixXd f4 = (E + p).cwiseProduct(u);
    
    MatrixXd g1 = r.cwiseProduct(v);
    MatrixXd g2 = f3;
    MatrixXd g3 = r.cwiseProduct(v.array().square().matrix()) + p;
    MatrixXd g4 = (E + p).cwiseProduct(v);
    
    // 定义q变量
    MatrixXd q1 = r;
    MatrixXd q2 = f1;
    MatrixXd q3 = g1;
    MatrixXd q4 = E;
    
    // 计算分裂通量
    fp1 = (f1 + a * q1) / 2;
    fn1 = (f1 - a * q1) / 2;
    gp1 = (g1 + b * q1) / 2;
    gn1 = (g1 - b * q1) / 2;
    
    fp2 = (f2 + a * q2) / 2;
    fn2 = (f2 - a * q2) / 2;
    gp2 = (g2 + b * q2) / 2;
    gn2 = (g2 - b * q2) / 2;
    
    fp3 = (f3 + a * q3) / 2;
    fn3 = (f3 - a * q3) / 2;
    gp3 = (g3 + b * q3) / 2;
    gn3 = (g3 - b * q3) / 2;
    
    fp4 = (f4 + a * q4) / 2;
    fn4 = (f4 - a * q4) / 2;
    gp4 = (g4 + b * q4) / 2;
    gn4 = (g4 - b * q4) / 2;
}