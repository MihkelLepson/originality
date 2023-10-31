#include <stdio.h>
#include <cuda_runtime.h>

__device__ float CalcLcs(const int* target,
                       const int m,
                       const int* reference,
                       const int n,
                       int* L,
                       const int tid,
                       const int start_idx,
                       const int lengthL,
                       const int max_distance) {
    int offset = start_idx+tid;
    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++) {
            if (i == 0 || j == 0)
                L[i*lengthL + j+offset] = 0;
            else if (target[i - 1] == reference[j - 1])
                L[i*lengthL + j+offset] = L[(i-1)*lengthL + j-1+offset] + 1;
            else
                L[i*lengthL + j+offset] = max(L[(i-1)*lengthL + j+offset], L[i*lengthL + j-1+offset]);
        }
    }
    int len = L[m*lengthL + n+offset];
    if (len <= 1) {
        return (float)len;
    }
    // Find the distances between members of LCS.
    // In locations1 we store the locations in the reference text and in locations2 we store the locations in the target text
    int locations1[512];
    int locations2[512];

    int row = m;
    int col = n;
    int val = len;

    while (val > 0) {
        if ((col > 1) && (L[row*lengthL + (col-1)+offset] == val)) {
            col -= 1;
        } else if ((row > 1) && (L[(row-1)*lengthL + col + offset] == val)) {
            row -= 1;
        } else {
            locations1[val-1] = col;
            locations2[val-1] = row;
            val -= 1;
        }
    }
    // The lcs without taking into account distances between members is atleast 2. (We returned if was <= 1).
    // We add the rest of the members but we modify their value based on the distance.
    float lcs = 1.0;
    for (int k = 0; k < len-1; k++) {
        // We take the maximum distance of the two.
        if (locations1[k+1] - locations1[k] >= locations2[k+1] - locations2[k]) {
            // If the two members in the lcs are consecutive then there is no penalty.
            lcs =  lcs + 1.0 - (float)(min(locations1[k+1] - locations1[k] - 1, max_distance)) / max_distance;
        } else {
            lcs = lcs + 1.0 - (float)(min(locations2[k+1] - locations2[k] - 1, max_distance)) / max_distance;
        }
    }

    return lcs;
}

// CUDA kernel
__global__ void lcsKernel(int* targets,
                          int* referneces,
                          float* lcs,
                          const int* divide_points,
                          int* L,
                          const int size_tar,
                          const int size_div,
                          const int lengthL) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size_div-1) {
        int start_idx = divide_points[tid];
        int end_idx = divide_points[tid + 1];

        int subarray_size = end_idx - start_idx;
        int* reference = &referneces[start_idx];
        lcs[tid] = CalcLcs(targets, size_tar, reference, subarray_size, L, tid, start_idx, lengthL, 20);
    }
}

extern "C" {
// Entry point function to be called from Python
void cudaLcs(int* targets,
             int* referneces,
             int* lcs,
             int* divide_points_tar,
             int* divide_points_ref,
             int size_ref,
             int size_div_tar,
             int size_div_ref) {
    
    int* d_targets;
    int* d_referneces;
    float* d_lcs;
    int* d_divide_points;
    int* d_L;

    // Allocate device memory
    cudaMalloc((void**)&d_referneces, sizeof(int) * size_ref);
    cudaMalloc((void**)&d_lcs, sizeof(float) * (size_div_ref-1));
    cudaMalloc((void**)&d_divide_points, sizeof(int) * size_div_ref); // We need one extra element for the last index

    // Copy input data from host to device
    cudaMemcpy(d_referneces, referneces, sizeof(int) * size_ref, cudaMemcpyHostToDevice);
    cudaMemcpy(d_divide_points, divide_points_ref, sizeof(int) * size_div_ref, cudaMemcpyHostToDevice); // Copy divide_points with an extra element

    // Define the kernel parameters
    int block_size = 256;
    int grid_size = (size_div_ref + block_size - 1) / block_size;

    int size_tar;
    // We process single target text at a time.
    for(int i = 0; i < size_div_tar-1; i++) {
        size_tar = divide_points_tar[i+1]-divide_points_tar[i];
        // Allocate the memory for target text.
        cudaMalloc((void**)&d_targets, sizeof(int) * size_tar);
        cudaMemcpy(d_targets, &targets[divide_points_tar[i]], sizeof(int) * size_tar, cudaMemcpyHostToDevice);
        // Allocate the memory for dynamic programming matrix. The matrix can have more then 2**15 cells which
        // is more than the local's memory size (depends on the GPU). Hence we use the slower global memory.
        cudaMalloc((void**)&d_L, sizeof(int) * (size_tar + 1)*(size_ref+size_div_ref-1));
        cudaMemset(d_L, 0, sizeof(int) * (size_tar + 1)*(size_ref+size_div_ref-1));

        // Calculate the LCS with each reference text
        lcsKernel<<<grid_size, block_size>>>(d_targets, d_referneces, d_lcs, d_divide_points, d_L, size_tar, size_div_ref, size_ref+size_div_ref-1);
        cudaDeviceSynchronize();
        // Check if errors occured in kernel function.
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
        // Get results
        cudaMemcpy(&lcs[i*(size_div_ref-1)], d_lcs, sizeof(float) * (size_div_ref-1), cudaMemcpyDeviceToHost);

        cudaFree(d_targets);
        cudaFree(d_L);
    }
    
    // Free device memory
    cudaFree(d_referneces);
    cudaFree(d_lcs);
    cudaFree(d_divide_points);
}
}
