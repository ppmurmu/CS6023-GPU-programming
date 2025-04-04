#include <iostream>
#include <vector>
#include <string>
#include <cuda.h>
#include <chrono>
#include <fstream>
using namespace std;
using std::cin;
using std::cout;

const int MOD = 1000000007;

struct Edge
{
    int src, dest, weight;
    int factor; // Terrain factor multiplier
};

// adjust weights kernel
__global__ void adjustWeights(Edge *edges, int E, int MOD)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < E)
    {
        edges[idx].weight = (edges[idx].weight * edges[idx].factor) % MOD;
    }
}

int main(int argc, char **argv)
{
    int V, E;
    cin >> V >> E;
    vector<Edge> edges(E);

    // Read input edges
    for (int i = 0; i < E; i++)
    {
        string type;
        cin >> edges[i].src >> edges[i].dest >> edges[i].weight >> type;

        // Assign factor based on terrain type
        if (type == "green")
            edges[i].factor = 2;
        else if (type == "traffic")
            edges[i].factor = 5;
        else if (type == "dept")
            edges[i].factor = 3;
        else
            edges[i].factor = 1;
    }
    // //Adjust weights based on terrain factors
    // for (auto &edge : edges)
    // {
    //     edge.weight = (edge.weight * edge.factor) % MOD;
    // }

    //--------------TIMER-----
    auto start = chrono::high_resolution_clock::now();

    // Allocate memory on GPU
    Edge *d_edges;
    int *d_mstWeight;
    int *d_component;
    int *d_minEdgeIdx;
    int *d_minEdgeWeight;
    int h_mstWeight = 0;

    cudaMalloc(&d_edges, E * sizeof(Edge));
    cudaMalloc(&d_mstWeight, sizeof(int));
    cudaMalloc(&d_component, V * sizeof(int));
    cudaMalloc(&d_minEdgeIdx, V * sizeof(int));
    cudaMalloc(&d_minEdgeWeight, V * sizeof(int));

    cudaMemcpy(d_edges, edges.data(), E * sizeof(Edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mstWeight, &h_mstWeight, sizeof(int), cudaMemcpyHostToDevice);

    // Set initial values
    vector<int> h_minEdgeIdx(V, -1);
    vector<int> h_minEdgeWeight(V, INT_MAX);
    cudaMemcpy(d_minEdgeIdx, h_minEdgeIdx.data(), V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_minEdgeWeight, h_minEdgeWeight.data(), V * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (E + threadsPerBlock - 1) / threadsPerBlock;
    adjustWeights<<<blocksPerGrid, threadsPerBlock>>>(d_edges, E, MOD);
    cudaDeviceSynchronize();

    //--- launch kernel to compute MST-----

    //--------------TIMER ends
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed1 = end - start;

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cerr << "CUDA error: " << cudaGetErrorString(err) << endl;
    }

    // Copy result back
    cudaMemcpy(&h_mstWeight, d_mstWeight, sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_edges);
    cudaFree(d_mstWeight);
    cudaFree(d_component);
    cudaFree(d_minEdgeIdx);
    cudaFree(d_minEdgeWeight);

    std::ofstream file("cuda.out");
    if (file.is_open())
    {
        file << h_mstWeight;
        file << "\n";
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