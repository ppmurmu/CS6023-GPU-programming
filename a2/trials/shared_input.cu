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

    // We'll use a fixed-size shared memory tile for each channel's data
    // This ensures we don't exceed shared memory limits
    extern __shared__ long int shared_input[];

    // Check if thread is within bounds
    if (idx < h * w)
    {
        // Each thread computes one output pixel for each filter
        for (int f = 0; f < k; ++f)
        {
            long int sum = 0;

            // Process each channel sequentially to conserve shared memory
            for (int ch = 0; ch < c; ++ch)
            {
                // Determine tile dimensions for this block
                // Calculate the region of the matrix this block processes
                int block_x_start = (blockIdx.x * blockDim.x) % w;
                int block_y_start = (blockIdx.x * blockDim.x) / w;

                // Determine the extended region needed including filter radius
                int radius_h = r / 2;
                int radius_w = s / 2;

                // Calculate tile dimensions (just enough for this block plus filter radius)
                int tile_width = min(blockDim.x, w - block_x_start) + 2 * radius_w;
                tile_width = min(tile_width, w); // Cap at matrix width

                // Calculate how many rows this block needs
                int rows_in_block = min((blockDim.x + w - 1) / w, h - block_y_start);
                int tile_height = rows_in_block + 2 * radius_h;
                tile_height = min(tile_height, h); // Cap at matrix height

                // Calculate memory offsets for loading
                int load_start_x = max(0, block_x_start - radius_w);
                int load_start_y = max(0, block_y_start - radius_h);
                int load_end_x = min(w, block_x_start + blockDim.x + radius_w);
                int load_end_y = min(h, block_y_start + rows_in_block + radius_h);

                // Adjusted tile width/height based on matrix bounds
                int actual_tile_width = load_end_x - load_start_x;
                int actual_tile_height = load_end_y - load_start_y;

                // Collaborative loading of the input tile
                for (int i = threadIdx.x; i < actual_tile_width * actual_tile_height; i += blockDim.x)
                {
                    int local_y = i / actual_tile_width;
                    int local_x = i % actual_tile_width;

                    int global_y = load_start_y + local_y;
                    int global_x = load_start_x + local_x;

                    // Load from global memory into shared memory
                    shared_input[local_y * actual_tile_width + local_x] =
                        matrix[(ch * h + global_y) * w + global_x];
                }

                // Ensure all threads finish loading
                __syncthreads();

                // Only threads within bounds compute the convolution
                if (idx < h * w)
                {
                    // Compute convolution using shared memory
                    for (int i = 0; i < r; ++i)
                    {
                        for (int j = 0; j < s; ++j)
                        {
                            // Calculate global position for this filter tap
                            int row = y + i - radius_h;
                            int col = x + j - radius_w;

                            // Check if within matrix bounds
                            if (row >= 0 && row < h && col >= 0 && col < w)
                            {
                                // Calculate corresponding position in shared memory tile
                                int local_row = row - load_start_y;
                                int local_col = col - load_start_x;

                                // Check if within the tile we loaded
                                if (local_row >= 0 && local_row < actual_tile_height &&
                                    local_col >= 0 && local_col < actual_tile_width)
                                {
                                    // Use shared memory
                                    sum += shared_input[local_row * actual_tile_width + local_col] *
                                           filter[f * (r * s * c) + ch * (r * s) + i * s + j];
                                }
                                else
                                {
                                    // Fall back to global memory for elements outside our tile
                                    sum += matrix[(ch * h + row) * w + col] *
                                           filter[f * (r * s * c) + ch * (r * s) + i * s + j];
                                }
                            }
                        }
                    }
                }

                // Ensure all threads are done with current tile before loading next channel
                __syncthreads();
            }

            // Store the final result for this filter
            if (idx < h * w)
            {
                result[f * h * w + idx] = sum;
            }
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

    // Calculate reasonable tile dimensions for shared memory
    int radius_h = r / 2;
    int radius_w = s / 2;

    // Calculate maximum width for a block
    int block_width = min(blockSize, w);
    // Calculate number of rows
    int rows_per_block = min((blockSize + w - 1) / w, h);

    // Calculate extended tile dimensions
    int tile_width = min(block_width + 2 * radius_w, w);
    int tile_height = min(rows_per_block + 2 * radius_h, h);

    // Calculate shared memory size for one channel at a time
    int sharedMemSize = tile_width * tile_height * sizeof(long int);

    // Make sure we don't exceed shared memory limits (typically 48KB)
    int max_shared_mem = 48 * 1024; // 48KB typical limit
    if (sharedMemSize > max_shared_mem)
    {
        // Adjust tile size to fit in shared memory if needed
        float scale_factor = (float)max_shared_mem / sharedMemSize;
        tile_width = max(16, (int)(tile_width * sqrt(scale_factor)));
        tile_height = max(16, (int)(tile_height * sqrt(scale_factor)));
        sharedMemSize = tile_width * tile_height * sizeof(long int);
    }

    // Launch the kernel
    dkernel<<<gridSize, blockSize, sharedMemSize>>>(d_mat, d_filter, d_result, h, w, c, r, s, k);
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
