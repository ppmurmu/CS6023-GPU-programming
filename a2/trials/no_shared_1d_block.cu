#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;

// CUDA Kernel for 2D Convolution
__global__ void dkernel(long int *matrix, long int *filter, long int *result, int h, int w, int c, int r, int s, int k)
{
    // Calculate global thread position
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = idx % w;
    int y = idx / w;

    // Check if thread is within bounds
    if (idx < h * w)
    {
        // Each thread computes one output pixel for each filter
        for (int f = 0; f < k; ++f)
        {
            long int sum = 0;

            // Apply filter - iterate over all filter elements and all channels
            for (int ch = 0; ch < c; ++ch)
            {
                for (int i = 0; i < r; ++i)
                {
                    for (int j = 0; j < s; ++j)
                    {
                        // Calculate row and column indices for input matrix
                        int row = y + i - r / 2;
                        int col = x + j - s / 2;

                        // Apply zero padding - only access the matrix if indices are valid
                        if (row >= 0 && row < h && col >= 0 && col < w)
                        {
                            // For channel ch, the position in the stacked image is (ch*h + row, col)
                            sum += matrix[(ch * h + row) * w + col] *
                                   filter[f * (r * s * c) + ch * (r * s) + i * s + j];
                        }
                    }
                }
            }

            // Store the result
            result[f * h * w + idx] = sum;
        }
    }
}

int main(int argc, char **argv)
{
    int h, w, c;
    cin >> h >> w >> c;
    long int *h_mat = new long int[h * w * c];
    for (long int i = 0; i < h * w * c; i++)
    {
        cin >> h_mat[i];
    }

    int cf, r, s, k;
    cin >> cf >> r >> s >> k;

    long int *h_filter = new long int[r * s * c * k];
    for (long int i = 0; i < r * s * c * k; i++)
    {
        cin >> h_filter[i];
    }
    long int *h_ans = new long int[h * w * k];

    /**
     *
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     *
     **/

    auto start = std::chrono::high_resolution_clock::now(); // keep it just before the kernel launch

    /****************************************************Start Here***********************************************************/

    /**
        Do device allocations, kernel launches and copying everything here
        and the final answer should be stored back in h_ans, use cudaFree to free up the allocated memory on GPU
    */

    long int *d_mat, *d_filter, *d_result;
    cudaMalloc((void **)&d_mat, h * w * c * sizeof(long int));
    cudaMalloc((void **)&d_filter, r * s * c * k * sizeof(long int));
    cudaMalloc((void **)&d_result, h * w * k * sizeof(long int));

    // Copy data from host to device
    cudaMemcpy(d_mat, h_mat, h * w * c * sizeof(long int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, r * s * c * k * sizeof(long int), cudaMemcpyHostToDevice);

    //---
    // After input processing in main():

    //--
    // Define block and grid dimensions
    int blockSize = 256;
    int gridSize = (h * w + blockSize - 1) / blockSize;
    // Calculate shared memory size needed for optimized kernel
    // int sharedMemSize = c * blockSize.y * blockSize.x * sizeof(long int);

    // Launch the kernel
    dkernel<<<gridSize, blockSize>>>(d_mat, d_filter, d_result, h, w, c, r, s, k);

    // Copy results back to host
    cudaMemcpy(h_ans, d_result, h * w * k * sizeof(long int), cudaMemcpyDeviceToHost);
    // Copy results back to host
    cudaMemcpy(h_ans, d_result, sizeof(long int) * h * w * k, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_mat);
    cudaFree(d_filter);
    cudaFree(d_result);

    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    auto end = std::chrono::high_resolution_clock::now(); // keep it just after the kernel launch
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     *
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     *
     */

    cudaDeviceSynchronize();
    std::ofstream file("cuda.out");
    if (file.is_open())
    {
        for (long int i = 0; i < h * k; i++)
        {
            for (long int j = 0; j < w; j++)
            {
                file << h_ans[i * w + j] << " ";
            }
            file << "\n";
        }
        file.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if (file2.is_open())
    {
        file2 << elapsed1.count() << "\n";
        file2.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    return 0;
}
