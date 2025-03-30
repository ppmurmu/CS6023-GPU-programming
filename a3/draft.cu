#include <iostream>
#include <vector>
#include <string>
#include <cuda.h>
#include <chrono>

using namespace std;

#define MOD 1000000007

struct Edge
{
    int src, dest, weight;
    string type; // Terrain type
};

// CUDA Kernel for MST Calculation (Placeholder)
__global__ void computeMST(Edge *edges, int *mstWeight, int V, int E)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= E)
        return;

    // TODO: Implement Kruskal’s or Prim’s algorithm in CUDA
}

int main()
{
    int V, E;
    cin >> V >> E;
    vector<Edge> edges(E);

    // Read input edges
    for (int i = 0; i < E; i++)
    {
        cin >> edges[i].src >> edges[i].dest >> edges[i].weight >> edges[i].type;
    }

    // Allocate memory on GPU
    Edge *d_edges;
    int *d_mstWeight;
    int h_mstWeight = 0;

    cudaMalloc(&d_edges, E * sizeof(Edge));
    cudaMalloc(&d_mstWeight, sizeof(int));
    cudaMemcpy(d_edges, edges.data(), E * sizeof(Edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mstWeight, &h_mstWeight, sizeof(int), cudaMemcpyHostToDevice);

    auto start = chrono::high_resolution_clock::now();

    // Launch CUDA Kernel
    computeMST<<<(E + 255) / 256, 256>>>(d_edges, d_mstWeight, V, E);
    cudaDeviceSynchronize();

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    // Copy result back
    cudaMemcpy(&h_mstWeight, d_mstWeight, sizeof(int), cudaMemcpyDeviceToHost);

    // Print only the total MST weight
    cout << h_mstWeight << endl;
    // cout << elapsed.count() << " s\n";

    // Free GPU memory
    cudaFree(d_edges);
    cudaFree(d_mstWeight);

    auto start = std::chrono::high_resolution_clock::now(); // keep it just before the kernel launch

    /****************************************************Start Here***********************************************************/

    /**
        Do device allocations, kernel launches and copying everything here
        and the final answer should be stored back in h_ans, use cudaFree to free up the allocated memory on GPU
    */

    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    auto end = std::chrono::high_resolution_clock::now(); // keep it just after the kernel launch
    std::chrono::duration<double> elapsed1 = end - start;
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
