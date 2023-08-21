#include <stdio.h>
#include <cuda_runtime.h>


__device__ int CalcLcs(const int* generated, const int m, const int* original, const int n, int* L, const int tid, const int start_idx, const int lengthL) {
    int offset = start_idx+tid;
    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++) {
            if (i == 0 || j == 0)
                L[i*lengthL + j+offset] = 0;
            else if (generated[i - 1] == original[j - 1])
                L[i*lengthL + j+offset] = L[(i-1)*lengthL + j-1+offset] + 1;
            else
                L[i*lengthL + j+offset] = max(L[(i-1)*lengthL + j+offset], L[i*lengthL + j-1+offset]);
        }
    }
    return L[m*lengthL + n+offset];
}

// CUDA kernel
__global__ void lcsKernel(const int* generated, int* originals, int* lcs, const int* divide_points, int* L, const int size_gen, const int size_div, const int lengthL) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size_div-1) {
        int start_idx = divide_points[tid];
        int end_idx = divide_points[tid + 1];

        int subarray_size = end_idx - start_idx;
        int* original = &originals[start_idx];
        lcs[tid] = CalcLcs(generated, size_gen, original, subarray_size, L, tid, start_idx, lengthL);
    }
}

extern "C" {
// Entry point function to be called from Python
void cudaLcs(int* generated, int* originals, int* lcs, int* divide_points, int size_gen, int size_org, int size_div) {
    int* d_generated;
    int* d_originals;
    int* d_lcs;
    int* d_divide_points;
    int* d_L;
    printf("here\n");

    // Allocate device memory
    cudaMalloc((void**)&d_generated, sizeof(int) * size_gen);
    cudaMalloc((void**)&d_originals, sizeof(int) * size_org);
    cudaMalloc((void**)&d_lcs, sizeof(int) * (size_div-1));
    cudaMalloc((void**)&d_divide_points, sizeof(int) * size_div); // We need one extra element for the last index
    // TODO, calcualte the maximum allowed size and if needed divide computation.
    // LCS needs (m+1)*(n+1) sized matrix. We have size_div-1 originals, so we add size_div-1
    cudaError_t cudaStatus = cudaMalloc((void**)&d_L, sizeof(int) * (size_gen + 1)*(size_org+size_div-1));
    //cudaStatus = cudaMemset(d_L, 0, sizeof(int) * (size_gen + 1)*(size_org+size_div-1));
    if (cudaStatus != cudaSuccess) {
        // Handle error
        printf("CUDA error: %s\n", cudaGetErrorString(cudaStatus));
    }
    // Copy input data from host to device
    cudaMemcpy(d_generated, generated, sizeof(int) * size_gen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_originals, originals, sizeof(int) * size_org, cudaMemcpyHostToDevice);
    cudaMemcpy(d_divide_points, divide_points, sizeof(int) * size_div, cudaMemcpyHostToDevice); // Copy divide_points with an extra element
    printf("Memory copy to device successful\n");

    // Launch kernel
    int block_size = 256;
    int grid_size = (size_div + block_size - 1) / block_size;
    lcsKernel<<<grid_size, block_size>>>(d_generated, d_originals, d_lcs, d_divide_points, d_L, size_gen, size_div, size_org+size_div-1);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    } else {
        printf("CUDA success\n");
    }

    // Copy result data from device to host
    cudaMemcpy(lcs, d_lcs, sizeof(int) * (size_div-1), cudaMemcpyDeviceToHost);
    printf("Memory copy to host successful\n");
    
    // Free device memory
    cudaFree(d_generated);
    cudaFree(d_originals);
    cudaFree(d_lcs);
    cudaFree(d_divide_points);
    cudaFree(d_L);
}
}