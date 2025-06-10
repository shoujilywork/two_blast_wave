#include <Eigen/Dense>
#include <omp.h>
#include "../include/nov_5.h"
#include "../include/derivatives.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;

// 并行化的dx_p函数
MatrixXd dx_p(const MatrixXd& u, double h) {
    const int m = u.rows();
    const int n = u.cols();
    MatrixXd dudh = MatrixXd::Zero(m, n);
    
    #pragma omp parallel for
    for(int i = 0; i < m; ++i) {
        // 将行向量转换为列向量输入nov_5
        VectorXd row_vec = u.row(i).transpose();
        VectorXd derivative = nov_5(row_vec, h);
        
        // 将结果转置回行向量存储
        dudh.row(i) = derivative.transpose();
    }
    
    return dudh;
}