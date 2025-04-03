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

// Serialized CUDA Kernel for Boruvka's MST Calculation
__global__ void computeMST(Edge *edges, int *mstWeight, int *component, int *minEdgeIdx, int *minEdgeWeight, int V, int E)
{
    if (threadIdx.x + blockIdx.x * blockDim.x > 0)
        return; // Only single thread executes

    // Initialize components
    for (int i = 0; i < V; i++)
    {
        component[i] = i;
    }

    bool changed;
    do
    {
        changed = false;

        // Reset min edges for each component
        for (int i = 0; i < V; i++)
        {
            minEdgeIdx[i] = -1;
            minEdgeWeight[i] = INT_MAX;
        }

        // Find the minimum outgoing edge for each component
        for (int i = 0; i < E; i++)
        {
            int u = edges[i].src;
            int v = edges[i].dest;
            int w = edges[i].weight;

            int compU = component[u];
            int compV = component[v];

            if (compU != compV)
            {
                if (w < minEdgeWeight[compU])
                {
                    minEdgeWeight[compU] = w;
                    minEdgeIdx[compU] = i;
                }
                if (w < minEdgeWeight[compV])
                {
                    minEdgeWeight[compV] = w;
                    minEdgeIdx[compV] = i;
                }
            }
        }

        // Merge components
        for (int i = 0; i < V; i++)
        {
            if (minEdgeIdx[i] != -1)
            {
                int edgeIdx = minEdgeIdx[i];
                int u = edges[edgeIdx].src;
                int v = edges[edgeIdx].dest;
                int compU = component[u];
                int compV = component[v];

                if (compU != compV)
                {
                    for (int j = 0; j < V; j++)
                    {
                        if (component[j] == compV)
                        {
                            component[j] = compU;
                        }
                    }
                    *mstWeight = (*mstWeight + edges[edgeIdx].weight) % MOD;

                    changed = true;
                }
            }
        }
    } while (changed);
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

    // Launch CUDA Kernel (single-threaded execution)
    computeMST<<<1, 1>>>(d_edges, d_mstWeight, d_component, d_minEdgeIdx, d_minEdgeWeight, V, E);

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
