#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

//
__global__ void max_reduce_kernel(const double* input, double* output, int size) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < size) ? input[i] : -INFINITY;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

//
__global__ void split_lf_kernel(
    const double* r, const double* p, 
    const double* u, const double* v, const double* E,
    double* fp1, double* fp2, double* fp3, double* fp4,
    double* fn1, double* fn2, double* fn3, double* fn4,
    double* gp1, double* gp2, double* gp3, double* gp4,
    double* gn1, double* gn2, double* gn3, double* gn4,
    double a, double b, int rows, int cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < cols && j < rows) {
        int idx = j * cols + i;
        
        //
        double f1 = r[idx] * u[idx];
        double f2 = r[idx] * u[idx] * u[idx] + p[idx];
        double f3 = r[idx] * u[idx] * v[idx];
        double f4 = (E[idx] + p[idx]) * u[idx];
        
        double g1 = r[idx] * v[idx];
        double g2 = f3;
        double g3 = r[idx] * v[idx] * v[idx] + p[idx];
        double g4 = (E[idx] + p[idx]) * v[idx];
        
        // 
        double q1 = r[idx];
        double q2 = f1;
        double q3 = g1;
        double q4 = E[idx];
        
        // 
        fp1[idx] = (f1 + a * q1) / 2.0;
        fn1[idx] = (f1 - a * q1) / 2.0;
        gp1[idx] = (g1 + b * q1) / 2.0;
        gn1[idx] = (g1 - b * q1) / 2.0;
        
        fp2[idx] = (f2 + a * q2) / 2.0;
        fn2[idx] = (f2 - a * q2) / 2.0;
        gp2[idx] = (g2 + b * q2) / 2.0;
        gn2[idx] = (g2 - b * q2) / 2.0;
        
        fp3[idx] = (f3 + a * q3) / 2.0;
        fn3[idx] = (f3 - a * q3) / 2.0;
        gp3[idx] = (g3 + b * q3) / 2.0;
        gn3[idx] = (g3 - b * q3) / 2.0;
        
        fp4[idx] = (f4 + a * q4) / 2.0;
        fn4[idx] = (f4 - a * q4) / 2.0;
        gp4[idx] = (g4 + b * q4) / 2.0;
        gn4[idx] = (g4 - b * q4) / 2.0;
    }
}

void split_lf(
    const double* r, const double* p, 
    const double* u, const double* v, const double* E,
    double* fp1, double* fp2, double* fp3, double* fp4,
    double* fn1, double* fn2, double* fn3, double* fn4,
    double* gp1, double* gp2, double* gp3, double* gp4,
    double* gn1, double* gn2, double* gn3, double* gn4,
    int rows, int cols)
{
    // 1
    double *d_r, *d_p, *d_u, *d_v, *d_E;
    double *d_fp1, *d_fp2, *d_fp3, *d_fp4;
    double *d_fn1, *d_fn2, *d_fn3, *d_fn4;
    double *d_gp1, *d_gp2, *d_gp3, *d_gp4;
    double *d_gn1, *d_gn2, *d_gn3, *d_gn4;
    double *d_temp, *d_max;
    
    size_t size = rows * cols * sizeof(double);
    size_t temp_size = rows * cols;
    
    cudaMalloc(&d_r, size);
    cudaMalloc(&d_p, size);
    cudaMalloc(&d_u, size);
    cudaMalloc(&d_v, size);
    cudaMalloc(&d_E, size);
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

    cudaMalloc(&d_temp, temp_size * sizeof(double));
    cudaMalloc(&d_max, sizeof(double));
    
    // 2. 
    cudaMemcpy(d_r, r, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, p, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_E, E, size, cudaMemcpyHostToDevice);
    
    // 3.
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, 
              (rows + block.y - 1) / block.y);

    double a = 0.0, b = 0.0; // 
    split_lf_kernel<<<grid, block>>>(
        d_r, d_p, d_u, d_v, d_E,
        d_fp1, d_fp2, d_fp3, d_fp4,
        d_fn1, d_fn2, d_fn3, d_fn4,
        d_gp1, d_gp2, d_gp3, d_gp4,
        d_gn1, d_gn2, d_gn3, d_gn4,
        a, b, rows, cols);
    
    // 5. 
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
    
    // 6
    cudaFree(d_r); cudaFree(d_p); cudaFree(d_u); cudaFree(d_v); cudaFree(d_E);
    cudaFree(d_fp1); cudaFree(d_fp2); cudaFree(d_fp3); cudaFree(d_fp4);
    cudaFree(d_fn1); cudaFree(d_fn2); cudaFree(d_fn3); cudaFree(d_fn4);
    cudaFree(d_gp1); cudaFree(d_gp2); cudaFree(d_gp3); cudaFree(d_gp4);
    cudaFree(d_gn1); cudaFree(d_gn2); cudaFree(d_gn3); cudaFree(d_gn4);
    cudaFree(d_temp); cudaFree(d_max);
}
