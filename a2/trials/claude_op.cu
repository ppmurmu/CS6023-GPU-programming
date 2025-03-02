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
    // Calculate global thread position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if thread is within bounds
    if (x < w && y < h)
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
            result[f * h * w + y * w + x] = sum;
        }
    }
}

//------CORRRECT CODE--------
__global__ void dkernel_shared(long int *matrix, long int *filter, long int *result, int h, int w, int c, int r, int s, int k)
{
    // Block dimensions
    int BLOCK_SIZE_X = blockDim.x;
    int BLOCK_SIZE_Y = blockDim.y;

    // Define the output region size for each block
    int OUTPUT_TILE_WIDTH = BLOCK_SIZE_X - r + 1;
    int OUTPUT_TILE_HEIGHT = BLOCK_SIZE_Y - s + 1;

    // Calculate global thread position for output
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int out_x = blockIdx.x * OUTPUT_TILE_WIDTH + tx;
    int out_y = blockIdx.y * OUTPUT_TILE_HEIGHT + ty;

    // Calculate the top-left position of the input tile (including halo)
    int tile_start_x = blockIdx.x * OUTPUT_TILE_WIDTH - r / 2;
    int tile_start_y = blockIdx.y * OUTPUT_TILE_HEIGHT - s / 2;

    // Shared memory for input tile (including halo regions)
    extern __shared__ long int shared_mem[];

    // Load input matrix tile into shared memory (with padding)
    for (int ch = 0; ch < c; ++ch)
    {
        // Each thread may need to load multiple elements to cover the entire tile
        for (int dy = ty; dy < BLOCK_SIZE_Y; dy += blockDim.y)
        {
            for (int dx = tx; dx < BLOCK_SIZE_X; dx += blockDim.x)
            {
                // Calculate global position for loading
                int load_x = tile_start_x + dx;
                int load_y = tile_start_y + dy;

                // Calculate shared memory index
                int sharedIdx = ch * BLOCK_SIZE_Y * BLOCK_SIZE_X + dy * BLOCK_SIZE_X + dx;

                // Check bounds and load data or use zero padding
                if (load_x >= 0 && load_x < w && load_y >= 0 && load_y < h)
                {
                    shared_mem[sharedIdx] = matrix[(ch * h + load_y) * w + load_x];
                }
                else
                {
                    shared_mem[sharedIdx] = 0; // Zero padding
                }
            }
        }
    }

    __syncthreads();

    // Only threads that compute output values proceed
    if (tx < OUTPUT_TILE_WIDTH && ty < OUTPUT_TILE_HEIGHT && out_x < w && out_y < h)
    {
        // Compute convolution for each filter
        for (int f = 0; f < k; ++f)
        {
            long int sum = 0;

            // Iterate over filter dimensions and channels
            for (int ch = 0; ch < c; ++ch)
            {
                for (int i = 0; i < r; ++i)
                {
                    for (int j = 0; j < s; ++j)
                    {
                        // Calculate position in shared memory
                        // tx + r/2 and ty + s/2 adjust for the halo offset
                        int sharedIdx = ch * BLOCK_SIZE_Y * BLOCK_SIZE_X +
                                        (ty + i) * BLOCK_SIZE_X + (tx + j);

                        // Calculate filter index
                        int filterIdx = f * (r * s * c) + ch * (r * s) + i * s + j;

                        sum += shared_mem[sharedIdx] * filter[filterIdx];
                    }
                }
            }

            // Write result to global memory
            result[f * h * w + out_y * w + out_x] = sum;
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

    // Device memory allocation
    long int *d_mat, *d_filter, *d_result;
    cudaMalloc((void **)&d_mat, h * w * c * sizeof(long int));
    cudaMalloc((void **)&d_filter, r * s * c * k * sizeof(long int));
    cudaMalloc((void **)&d_result, h * w * k * sizeof(long int));

    // Copy data from host to device
    cudaMemcpy(d_mat, h_mat, h * w * c * sizeof(long int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, r * s * c * k * sizeof(long int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);

    // Calculate shared memory size needed for optimized kernel
    int sharedMemSize = c * blockSize.y * blockSize.x * sizeof(long int);

    // Launch the kernel
    dkernel_shared<<<gridSize, blockSize, sharedMemSize>>>(d_mat, d_filter, d_result, h, w, c, r, s, k);

    // Copy results back to host
    cudaMemcpy(h_ans, d_result, h * w * k * sizeof(long int), cudaMemcpyDeviceToHost);

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