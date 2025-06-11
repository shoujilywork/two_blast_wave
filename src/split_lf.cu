#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include "../include/split_lf.h"

#define BLOCK_SIZE 16

// CUDA核函数：计算声速c
__global__ void computeSoundSpeed(double* c, const double* r, const double* p, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < rows && j < cols) {
        c[i*cols + j] = sqrt(fmax(0.0, 1.4 * p[i*cols + j] / r[i*cols + j]));
    }
}

// CUDA核函数：计算特征速度（并行归约优化版）
__global__ void computeCharacteristicSpeed(double* a, double* b, 
                                         const double* u, const double* v, 
                                         const double* c, int size) {
    extern __shared__ double sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    double my_a = 0.0, my_b = 0.0;
    if (i < size) {
        my_a = fabs(u[i] + c[i]);
        my_b = fabs(v[i] + c[i]);
    }
    
    // 并行归约求最大值
    sdata[tid] = my_a;
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        a[blockIdx.x] = sdata[0];
    }
    
    // 同样的方法计算b
    sdata[tid] = my_b;
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        b[blockIdx.x] = sdata[0];
    }
}

// CUDA核函数：计算通量和分裂通量（使用共享内存优化）
__global__ void computeFluxes(double* fp1, double* fp2, double* fp3, double* fp4,
                             double* fn1, double* fn2, double* fn3, double* fn4,
                             double* gp1, double* gp2, double* gp3, double* gp4,
                             double* gn1, double* gn2, double* gn3, double* gn4,
                             const double* r, const double* p,
                             const double* u, const double* v,
                             const double* E, double a, double b,
                             int rows, int cols) {
    __shared__ double s_r[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double s_p[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double s_u[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double s_v[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double s_E[BLOCK_SIZE][BLOCK_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = blockIdx.y * BLOCK_SIZE + ty;
    int j = blockIdx.x * BLOCK_SIZE + tx;
    
    // 加载数据到共享内存
    if (i < rows && j < cols) {
        s_r[ty][tx] = r[i*cols + j];
        s_p[ty][tx] = p[i*cols + j];
        s_u[ty][tx] = u[i*cols + j];
        s_v[ty][tx] = v[i*cols + j];
        s_E[ty][tx] = E[i*cols + j];
    }
    __syncthreads();
    
    if (i < rows && j < cols) {
        // 计算通量f和g
        double f1 = s_r[ty][tx] * s_u[ty][tx];
        double f2 = s_r[ty][tx] * s_u[ty][tx] * s_u[ty][tx] + s_p[ty][tx];
        double f3 = s_r[ty][tx] * s_u[ty][tx] * s_v[ty][tx];
        double f4 = (s_E[ty][tx] + s_p[ty][tx]) * s_u[ty][tx];
        
        double g1 = s_r[ty][tx] * s_v[ty][tx];
        double g2 = f3;
        double g3 = s_r[ty][tx] * s_v[ty][tx] * s_v[ty][tx] + s_p[ty][tx];
        double g4 = (s_E[ty][tx] + s_p[ty][tx]) * s_v[ty][tx];
        
        // 定义q变量
        double q1 = s_r[ty][tx];
        double q2 = f1;
        double q3 = g1;
        double q4 = s_E[ty][tx];
        
        // 计算分裂通量
        fp1[i*cols + j] = (f1 + a * q1) / 2;
        fn1[i*cols + j] = (f1 - a * q1) / 2;
        gp1[i*cols + j] = (g1 + b * q1) / 2;
        gn1[i*cols + j] = (g1 - b * q1) / 2;
        
        fp2[i*cols + j] = (f2 + a * q2) / 2;
        fn2[i*cols + j] = (f2 - a * q2) / 2;
        gp2[i*cols + j] = (g2 + b * q2) / 2;
        gn2[i*cols + j] = (g2 - b * q2) / 2;
        
        fp3[i*cols + j] = (f3 + a * q3) / 2;
        fn3[i*cols + j] = (f3 - a * q3) / 2;
        gp3[i*cols + j] = (g3 + b * q3) / 2;
        gn3[i*cols + j] = (g3 - b * q3) / 2;
        
        fp4[i*cols + j] = (f4 + a * q4) / 2;
        fn4[i*cols + j] = (f4 - a * q4) / 2;
        gp4[i*cols + j] = (g4 + b * q4) / 2;
        gn4[i*cols + j] = (g4 - b * q4) / 2;
    }
}

// 主函数：封装CUDA调用
void split_lf(const double* r, const double* p, const double* u, const double* v, const double* E,
                  double* fp1, double* fp2, double* fp3, double* fp4,
                  double* fn1, double* fn2, double* fn3, double* fn4,
                  double* gp1, double* gp2, double* gp3, double* gp4,
                  double* gn1, double* gn2, double* gn3, double* gn4,
                  int rows, int cols) {
    // 设备内存指针
    double *d_r, *d_p, *d_u, *d_v, *d_E;
    double *d_fp1, *d_fp2, *d_fp3, *d_fp4;
    double *d_fn1, *d_fn2, *d_fn3, *d_fn4;
    double *d_gp1, *d_gp2, *d_gp3, *d_gp4;
    double *d_gn1, *d_gn2, *d_gn3, *d_gn4;
    double *d_c, *d_a, *d_b;
    
    // 分配设备内存
    size_t size = rows * cols * sizeof(double);
    cudaMalloc(&d_r, size);
    cudaMalloc(&d_p, size);
    cudaMalloc(&d_u, size);
    cudaMalloc(&d_v, size);
    cudaMalloc(&d_E, size);
    cudaMalloc(&d_c, size);
    
    // 分配输出设备内存
    cudaMalloc(&d_fp1, size);
    cudaMalloc(&d_fp2, size);
    cudaMalloc(&d_fp3, size);
    cudaMalloc(&d_fp4, size);
    cudaMalloc(&d_fn1, size);
    cudaMalloc(&d_fn2, size);
    cudaMalloc(&d_fn3, size);
    cudaMalloc(&d_fn4, size);
    cudaMalloc(&d_gp1, size);
    cudaMalloc(&d_gp2, size);
    cudaMalloc(&d_gp3, size);
    cudaMalloc(&d_gp4, size);
    cudaMalloc(&d_gn1, size);
    cudaMalloc(&d_gn2, size);
    cudaMalloc(&d_gn3, size);
    cudaMalloc(&d_gn4, size);
    
    // 分配特征速度存储
    cudaMalloc(&d_a, sizeof(double));
    cudaMalloc(&d_b, sizeof(double));
    
    // 拷贝数据到设备
    cudaMemcpy(d_r, r, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, p, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_E, E, size, cudaMemcpyHostToDevice);
    
    // 计算声速c
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, 
                 (rows + blockDim.y - 1) / blockDim.y);
    computeSoundSpeed<<<gridDim, blockDim>>>(d_c, d_r, d_p, rows, cols);
    
    // 计算特征速度a和b
    int blockSize = 256;
    int gridSize = (rows * cols + blockSize - 1) / blockSize;
    computeCharacteristicSpeed<<<gridSize, blockSize, 2*blockSize*sizeof(double)>>>(
        d_a, d_b, d_u, d_v, d_c, rows * cols);
    
    // 主机上的a和b
    double h_a, h_b;
    cudaMemcpy(&h_a, d_a, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_b, d_b, sizeof(double), cudaMemcpyDeviceToHost);
    
    // 计算通量和分裂通量
    computeFluxes<<<gridDim, blockDim>>>(d_fp1, d_fp2, d_fp3, d_fp4,
                                       d_fn1, d_fn2, d_fn3, d_fn4,
                                       d_gp1, d_gp2, d_gp3, d_gp4,
                                       d_gn1, d_gn2, d_gn3, d_gn4,
                                       d_r, d_p, d_u, d_v, d_E, 
                                       h_a, h_b, rows, cols);
    
    // 拷贝结果回主机
    cudaMemcpy(fp1, d_fp1, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(fp2, d_fp2, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(fp3, d_fp3, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(fp4, d_fp4, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(fn1, d_fn1, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(fn2, d_fn2, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(fn3, d_fn3, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(fn4, d_fn4, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(gp1, d_gp1, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(gp2, d_gp2, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(gp3, d_gp3, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(gp4, d_gp4, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(gn1, d_gn1, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(gn2, d_gn2, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(gn3, d_gn3, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(gn4, d_gn4, size, cudaMemcpyDeviceToHost);
    
    // 释放设备内存
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_E);
    cudaFree(d_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_fp1);
    cudaFree(d_fp2);
    cudaFree(d_fp3);
    cudaFree(d_fp4);
    cudaFree(d_fn1);
    cudaFree(d_fn2);
    cudaFree(d_fn3);
    cudaFree(d_fn4);
    cudaFree(d_gp1);
    cudaFree(d_gp2);
    cudaFree(d_gp3);
    cudaFree(d_gp4);
    cudaFree(d_gn1);
    cudaFree(d_gn2);
    cudaFree(d_gn3);
    cudaFree(d_gn4);
}