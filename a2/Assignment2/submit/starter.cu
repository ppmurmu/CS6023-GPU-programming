#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda/cuda_runtime.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;

__global__ void dkernel(long int *matrix, long int *filter, long int *result, int h, int w, int c, int r, int s, int k)
{
    // Shared memory for input matrix and filter
    extern __shared__ long int shared_mem[];
    long int *shared_matrix = shared_mem;
    long int *shared_filter = shared_mem + (blockDim.y + r - 1) * (blockDim.x + s - 1);

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global indices
    int x = bx * blockDim.x + tx;
    int y = by * blockDim.y + ty;

    // Load input matrix into shared memory with padding
    for (int ch = 0; ch < c; ++ch)
    {
        if (x < w && y < h)
        {
            shared_matrix[ty * (blockDim.x + s - 1) + tx] = matrix[ch * h * w + y * w + x];
        }
        else
        {
            shared_matrix[ty * (blockDim.x + s - 1) + tx] = 0; // Zero-padding
        }
    }

    __syncthreads();

    // Perform convolution for each filter
    for (int f = 0; f < k; ++f)
    {
        if (x < w && y < h)
        {
            long int sum = 0;

            // Apply filter
            for (int i = 0; i < r; ++i)
            {
                for (int j = 0; j < s; ++j)
                {
                    for (int ch = 0; ch < c; ++ch)
                    {
                        sum += shared_matrix[(ty + i) * (blockDim.x + s - 1) + (tx + j)] *
                               filter[f * c * r * s + ch * r * s + i * s + j];
                    }
                }
            }

            // Store result in global memory
            result[f * h * w + y * w + x] = sum;
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
