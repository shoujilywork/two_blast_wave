#include <omp.h>
#include <cmath>
#include <Eigen/Dense>

void new_ic(Eigen::MatrixXd& u, Eigen::MatrixXd& v, 
            Eigen::MatrixXd& rho, Eigen::MatrixXd& p,
            const Eigen::VectorXd& x, const Eigen::VectorXd& y) 
{
    const int m = y.size();
    const int n = x.size();
    const double tan35 = -tan(35.24091734 * M_PI/180.0);

    #pragma omp parallel for collapse(2)
    for(int j=0; j<m; ++j) {
        for(int i=0; i<n; ++i) {
            if(x(i) - 1 < y(j)/tan35) {
                u(j,i) = 2.0;
                v(j,i) = 0.0;
                rho(j,i) = 1.0;
                p(j,i) = 5.0/7.0;
            } else {
                u(j,i) = 1.861754751;
                v(j,i) = -0.195678309;
                rho(j,i) = 1.262002269;
                p(j,i) = 0.990776211;
            }
        }
    }
}