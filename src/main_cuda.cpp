#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <thread>
#include <fstream>
#include <string>
#include <iomanip>  // For formatting filenames
#include <omp.h>
#include "../include/split_lf.h"
#include "../include/derivatives.h"
#include "../include/new_ic.h"
#include "../include/new_bc2.h"


using namespace Eigen;
using namespace std;

const int n = 65;
const int m = 65;
const double dt = 0.0005;
const double gamma_ = 1.4;

int main() {
    // 初始化网格
    VectorXd x = VectorXd::LinSpaced(n, 0, 2);
    VectorXd y = VectorXd::LinSpaced(m, 0, 1.1);
    double dx = 2.0 / (n-1);
    double dy = 1.1 / (m-1);

    // 初始化场变量
    MatrixXd u = MatrixXd::Constant(m, n, 1.861755);
    MatrixXd v = MatrixXd::Constant(m, n, -0.1956783);
    MatrixXd rho = MatrixXd::Constant(m, n, 1.262002);
    MatrixXd p = MatrixXd::Constant(m, n, 0.9907762);

    // 设置初始条件
    new_ic(u, v, rho, p, x, y);

    // 计算能量
    MatrixXd E = p/(gamma_-1) + 0.5*rho.cwiseProduct(u.cwiseProduct(u) + v.cwiseProduct(v));
    MatrixXd ru = rho.cwiseProduct(u);
    MatrixXd rv = rho.cwiseProduct(v);

    MatrixXd fp1= MatrixXd::Zero(m, n);
    MatrixXd fp2= MatrixXd::Zero(m, n);
    MatrixXd fp3= MatrixXd::Zero(m, n);
    MatrixXd fp4= MatrixXd::Zero(m, n);
    MatrixXd fn1= MatrixXd::Zero(m, n);
    MatrixXd fn2= MatrixXd::Zero(m, n);
    MatrixXd fn3= MatrixXd::Zero(m, n);
    MatrixXd fn4= MatrixXd::Zero(m, n);
    MatrixXd gp1= MatrixXd::Zero(m, n);
    MatrixXd gp2= MatrixXd::Zero(m, n);
    MatrixXd gp3= MatrixXd::Zero(m, n);
    MatrixXd gp4= MatrixXd::Zero(m, n);
    MatrixXd gn1= MatrixXd::Zero(m, n);
    MatrixXd gn2= MatrixXd::Zero(m, n);
    MatrixXd gn3= MatrixXd::Zero(m, n);
    MatrixXd gn4= MatrixXd::Zero(m, n);

    // 迭代求解
    double tolerance = 999;
    MatrixXd old = MatrixXd::Constant(m, n, 111);
    int j = 1;

    while (tolerance > 0.0001) {
            for (int i = 0; i < 50; i++) {
                // 第一步

                split_lf(rho.data(), p.data(), u.data(), v.data(), E.data(),
                         fp1.data(), fp2.data(), fp3.data(), fp4.data(),
                         fn1.data(), fn2.data(), fn3.data(), fn4.data(),
                         gp1.data(), gp2.data(), gp3.data(), gp4.data(),
                         gn1.data(), gn2.data(), gn3.data(), gn4.data(),m,n);

                MatrixXd rk1 = -(dx_p(fp1, dx) + dx_n(fn1, dx) + dy_p(gp1, dy) + dy_n(gn1, dy)) * dt;
                MatrixXd ruk1 = -(dx_p(fp2, dx) + dx_n(fn2, dx) + dy_p(gp2, dy) + dy_n(gn2, dy)) * dt;
                MatrixXd rvk1 = -(dx_p(fp3, dx) + dx_n(fn3, dx) + dy_p(gp3, dy) + dy_n(gn3, dy)) * dt;
                MatrixXd Ek1 = -(dx_p(fp4, dx) + dx_n(fn4, dx) + dy_p(gp4, dy) + dy_n(gn4, dy)) * dt;

                MatrixXd ru1 = ru + ruk1/2;
                MatrixXd rv1 = rv + rvk1/2;
                MatrixXd r1 = rho + rk1/2;
                MatrixXd u1 = ru1.cwiseQuotient(r1);
                MatrixXd v1 = rv1.cwiseQuotient(r1);
                MatrixXd E1 = E + Ek1/2;
                MatrixXd p1 = (gamma_-1)*(E1 - 0.5*r1.cwiseProduct(u1.cwiseProduct(u1) + v1.cwiseProduct(v1)));

                new_bc2(u1, v1, r1, p1, x, y);

                // 第二步
                split_lf(r1.data(), p1.data(), u1.data(), v1.data(), E1.data(),
                         fp1.data(), fp2.data(), fp3.data(), fp4.data(),
                         fn1.data(), fn2.data(), fn3.data(), fn4.data(),
                         gp1.data(), gp2.data(), gp3.data(), gp4.data(),
                         gn1.data(), gn2.data(), gn3.data(), gn4.data(),m,n);

                MatrixXd rk2 = -(dx_p(fp1, dx) + dx_n(fn1, dx) + dy_p(gp1, dy) + dy_n(gn1, dy)) * dt;
                MatrixXd ruk2 = -(dx_p(fp2, dx) + dx_n(fn2, dx) + dy_p(gp2, dy) + dy_n(gn2, dy)) * dt;
                MatrixXd rvk2 = -(dx_p(fp3, dx) + dx_n(fn3, dx) + dy_p(gp3, dy) + dy_n(gn3, dy)) * dt;
                MatrixXd Ek2 = -(dx_p(fp4, dx) + dx_n(fn4, dx) + dy_p(gp4, dy) + dy_n(gn4, dy)) * dt;

                MatrixXd ru2 = ru + ruk2/2;
                MatrixXd rv2 = rv + rvk2/2;
                MatrixXd r2 = rho + rk2/2;
                MatrixXd u2 = ru2.cwiseQuotient(r2);
                MatrixXd v2 = rv2.cwiseQuotient(r2);
                MatrixXd E2 = E + Ek2/2;
                MatrixXd p2 = (gamma_-1)*(E2 - 0.5*r2.cwiseProduct(u2.cwiseProduct(u2) + v2.cwiseProduct(v2)));
                
                new_bc2(u2, v2, r2, p2, x, y);

                // 第三步
                split_lf(r2.data(), p2.data(), u2.data(), v2.data(), E2.data(),
                         fp1.data(), fp2.data(), fp3.data(), fp4.data(),
                         fn1.data(), fn2.data(), fn3.data(), fn4.data(),
                         gp1.data(), gp2.data(), gp3.data(), gp4.data(),
                         gn1.data(), gn2.data(), gn3.data(), gn4.data(),m,n);

                MatrixXd rk3 = -(dx_p(fp1, dx) + dx_n(fn1, dx) + dy_p(gp1, dy) + dy_n(gn1, dy)) * dt;
                MatrixXd ruk3 = -(dx_p(fp2, dx) + dx_n(fn2, dx) + dy_p(gp2, dy) + dy_n(gn2, dy)) * dt;
                MatrixXd rvk3 = -(dx_p(fp3, dx) + dx_n(fn3, dx) + dy_p(gp3, dy) + dy_n(gn3, dy)) * dt;
                MatrixXd Ek3 = -(dx_p(fp4, dx) + dx_n(fn4, dx) + dy_p(gp4, dy) + dy_n(gn4, dy)) * dt;

                MatrixXd ru3 = ru + ruk3;
                MatrixXd rv3 = rv + rvk3;
                MatrixXd r3 = rho + rk3;
                MatrixXd u3 = ru3.cwiseQuotient(r3);
                MatrixXd v3 = rv3.cwiseQuotient(r3);
                MatrixXd E3 = E + Ek3;
                MatrixXd p3 = (gamma_-1)*(E3 - 0.5*r3.cwiseProduct(u3.cwiseProduct(u3) + v3.cwiseProduct(v3)));
                
                new_bc2(u3, v3, r3, p3, x, y);

                // 第四步
                split_lf(r3.data(), p3.data(), u3.data(), v3.data(), E3.data(),
                         fp1.data(), fp2.data(), fp3.data(), fp4.data(),
                         fn1.data(), fn2.data(), fn3.data(), fn4.data(),
                         gp1.data(), gp2.data(), gp3.data(), gp4.data(),
                         gn1.data(), gn2.data(), gn3.data(), gn4.data(),m,n);
                MatrixXd rk4 = -(dx_p(fp1, dx) + dx_n(fn1, dx) + dy_p(gp1, dy) + dy_n(gn1, dy)) * dt;
                MatrixXd ruk4 = -(dx_p(fp2, dx) + dx_n(fn2, dx) + dy_p(gp2, dy) + dy_n(gn2, dy)) * dt;
                MatrixXd rvk4 = -(dx_p(fp3, dx) + dx_n(fn3, dx) + dy_p(gp3, dy) + dy_n(gn3, dy)) * dt;
                MatrixXd Ek4 = -(dx_p(fp4, dx) + dx_n(fn4, dx) + dy_p(gp4, dy) + dy_n(gn4, dy)) * dt;


                rho = rho + (rk1 + 2*rk2 + 2*rk3 + rk4)/6;
                ru = ru + (ruk1 + 2*ruk2 + 2*ruk3 + ruk4)/6;
                rv = rv + (rvk1 + 2*rvk2 + 2*rvk3 + rvk4)/6;
                u = ru.cwiseQuotient(rho);
                v = rv.cwiseQuotient(rho);
                E = E + (Ek1 + 2*Ek2 + 2*Ek3 + Ek4)/6;
                p = (gamma_-1)*(E - 0.5*rho.cwiseProduct(u.cwiseProduct(u) + v.cwiseProduct(v)));
                new_bc2(u, v, rho, p, x, y);
                E = p/(gamma_-1) + 0.5*rho.cwiseProduct(u.cwiseProduct(u) + v.cwiseProduct(v));
                tolerance = (E - old).cwiseAbs().maxCoeff();
                old = E;
                
            }
        j++;
        cout << "Iteration " << j << ", Tolerance: " << tolerance << endl;
    }

    return 0;
}