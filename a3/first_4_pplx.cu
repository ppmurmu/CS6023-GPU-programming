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

// Initialize components in parallel
__global__ void initComponents(int *component, int V)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < V)
    {
        component[idx] = idx;
    }
}

// Reset min-edge data in parallel
__global__ void resetMinEdges(int *minEdgeIdx, int *minEdgeWeight, int V)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < V)
    {
        minEdgeIdx[idx] = -1;
        minEdgeWeight[idx] = INT_MAX;
    }
}

// Find minimum edge weights for each component
__global__ void findMinEdges(Edge *edges, int E, int *component, int *minEdgeIdx, int *minEdgeWeight)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < E)
    {
        int u = edges[idx].src;
        int v = edges[idx].dest;
        int w = edges[idx].weight;

        int compU = component[u];
        int compV = component[v];

        if (compU != compV)
        {
            // Use atomic operations to update minimum weight
            atomicMin(&minEdgeWeight[compU], w);
            atomicMin(&minEdgeWeight[compV], w);
        }
    }
}

// Match minimum edges with actual edges
__global__ void matchMinEdges(Edge *edges, int E, int *component, int *minEdgeIdx, int *minEdgeWeight)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < E)
    {
        int u = edges[idx].src;
        int v = edges[idx].dest;
        int w = edges[idx].weight;

        int compU = component[u];
        int compV = component[v];

        if (compU != compV)
        {
            if (w == minEdgeWeight[compU])
            {
                // Use CAS to ensure only one thread succeeds in setting the edge
                atomicCAS(&minEdgeIdx[compU], -1, idx);
            }
            if (w == minEdgeWeight[compV])
            {
                atomicCAS(&minEdgeIdx[compV], -1, idx);
            }
        }
    }
}

// Merge components and update MST weight - this is sequential to match original algorithm
__global__ void mergeComponentsAndUpdateMST(Edge *edges, int *mstWeight, int *component, int *minEdgeIdx, int V, bool *changed, int MOD)
{
    if (threadIdx.x + blockIdx.x * blockDim.x > 0)
        return; // Only one thread executes

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
                // Update all vertices in compV to be in compU - exactly like serial version
                for (int j = 0; j < V; j++)
                {
                    if (component[j] == compV)
                    {
                        component[j] = compU;
                    }
                }

                // Update MST weight
                *mstWeight = (*mstWeight + edges[edgeIdx].weight) % MOD;

                *changed = true;
            }
        }
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

    //--------------TIMER-----
    auto start = chrono::high_resolution_clock::now();

    // Allocate memory on GPU
    Edge *d_edges;
    int *d_mstWeight;
    int *d_component;
    int *d_minEdgeIdx;
    int *d_minEdgeWeight;
    bool *d_changed;

    cudaMalloc(&d_edges, E * sizeof(Edge));
    cudaMalloc(&d_mstWeight, sizeof(int));
    cudaMalloc(&d_component, V * sizeof(int));
    cudaMalloc(&d_minEdgeIdx, V * sizeof(int));
    cudaMalloc(&d_minEdgeWeight, V * sizeof(int));
    cudaMalloc(&d_changed, sizeof(bool));

    int h_mstWeight = 0;
    cudaMemcpy(d_edges, edges.data(), E * sizeof(Edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mstWeight, &h_mstWeight, sizeof(int), cudaMemcpyHostToDevice);

    // Calculate grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGridV = (V + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridE = (E + threadsPerBlock - 1) / threadsPerBlock;

    // Adjust edge weights
    adjustWeights<<<blocksPerGridE, threadsPerBlock>>>(d_edges, E, MOD);
    cudaDeviceSynchronize();

    // Initialize components
    initComponents<<<blocksPerGridV, threadsPerBlock>>>(d_component, V);
    cudaDeviceSynchronize();

    // Main Boruvka loop
    bool h_changed = true;
    while (h_changed)
    {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        // Reset minimum edge data
        resetMinEdges<<<blocksPerGridV, threadsPerBlock>>>(d_minEdgeIdx, d_minEdgeWeight, V);
        cudaDeviceSynchronize();

        // Find minimum edge weights (parallel)
        findMinEdges<<<blocksPerGridE, threadsPerBlock>>>(d_edges, E, d_component, d_minEdgeIdx, d_minEdgeWeight);
        cudaDeviceSynchronize();

        // Match minimum edges with actual edges (parallel)
        matchMinEdges<<<blocksPerGridE, threadsPerBlock>>>(d_edges, E, d_component, d_minEdgeIdx, d_minEdgeWeight);
        cudaDeviceSynchronize();

        // Merge components and update MST weight (sequential - for correctness)
        mergeComponentsAndUpdateMST<<<1, 1>>>(d_edges, d_mstWeight, d_component, d_minEdgeIdx, V, d_changed, MOD);
        cudaDeviceSynchronize();

        // Check if any changes were made
        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    //--------------TIMER ends
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed1 = end - start;

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
    cudaFree(d_changed);

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
