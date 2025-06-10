#include <Eigen/Dense>
#include <omp.h>
#include "../include/nov_5.h"
#include "../include/derivatives.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;

// y方向导数计算（OpenMP并行版）
MatrixXd dy_n(const MatrixXd& u, double h) {
    const int m = u.rows();
    const int n = u.cols();
    MatrixXd dudh = MatrixXd::Zero(m, n);
    
    #pragma omp parallel for
    for(int i = 0; i < n; ++i) {
        // 获取列向量并转置为行向量输入nov_5
        VectorXd col_vec = u.col(i);
        VectorXd derivative = nov_5_n(col_vec, h);
        
        // 存储结果到对应列
        dudh.col(i) = derivative;
    }
    
    return dudh;
}