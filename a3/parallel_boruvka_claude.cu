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

// Adjust weights kernel
__global__ void adjustWeights(Edge *edges, int E, int MOD)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < E)
    {
        edges[idx].weight = (edges[idx].weight * edges[idx].factor) % MOD;
    }
}

// Find minimum outgoing edge for each component
__global__ void findMinEdges(Edge *edges, int *component, int *minEdgeIdx, int *minEdgeWeight, int V, int E)
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
            // Use atomic operations to update minimum edge for each component
            atomicMin(&minEdgeWeight[compU], w);
            atomicMin(&minEdgeWeight[compV], w);

            // Note: This is a race condition if multiple edges have the same weight
            // We address this by using a separate kernel to update indices
        }
    }
}

// Update edge indices based on minimum weights
__global__ void updateEdgeIndices(Edge *edges, int *component, int *minEdgeIdx, int *minEdgeWeight, int V, int E)
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
                // Potential race condition here too, but we only need one valid edge with minimum weight
                minEdgeIdx[compU] = idx;
            }
            if (w == minEdgeWeight[compV])
            {
                minEdgeIdx[compV] = idx;
            }
        }
    }
}

// Update components in parallel
__global__ void mergeComponents(Edge *edges, int *component, int *minEdgeIdx, int *changed, int *mstWeight, int V, int MOD)
{
    int compId = threadIdx.x + blockIdx.x * blockDim.x;
    if (compId < V)
    {
        if (minEdgeIdx[compId] != -1)
        {
            int edgeIdx = minEdgeIdx[compId];
            int u = edges[edgeIdx].src;
            int v = edges[edgeIdx].dest;
            int compU = component[u];
            int compV = component[v];

            if (compU != compV)
            {
                // Use the smallest component ID as the new component ID
                int newComp = min(compU, compV);
                int oldComp = max(compU, compV);

                // Update component for this vertex
                if (component[compId] == oldComp)
                {
                    component[compId] = newComp;
                    *changed = 1;

                    // Add to MST weight - potential race condition with atomicAdd
                    atomicAdd(mstWeight, edges[edgeIdx].weight);
                    // Handle modulo for mstWeight
                    if (*mstWeight >= MOD)
                    {
                        atomicAdd(mstWeight, -MOD);
                    }
                }
            }
        }
    }
}

// Initialize arrays kernel
__global__ void initializeArrays(int *component, int *minEdgeIdx, int *minEdgeWeight, int V)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < V)
    {
        component[idx] = idx;         // Each vertex starts in its own component
        minEdgeIdx[idx] = -1;         // No minimum edge yet
        minEdgeWeight[idx] = INT_MAX; // Initialize to maximum value
    }
}

// Reset min edges kernel
__global__ void resetMinEdges(int *minEdgeIdx, int *minEdgeWeight, int V)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < V)
    {
        minEdgeIdx[idx] = -1;
        minEdgeWeight[idx] = INT_MAX;
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
    int *d_changed;
    int h_mstWeight = 0;
    int h_changed;

    cudaMalloc(&d_edges, E * sizeof(Edge));
    cudaMalloc(&d_mstWeight, sizeof(int));
    cudaMalloc(&d_component, V * sizeof(int));
    cudaMalloc(&d_minEdgeIdx, V * sizeof(int));
    cudaMalloc(&d_minEdgeWeight, V * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));

    cudaMemcpy(d_edges, edges.data(), E * sizeof(Edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mstWeight, &h_mstWeight, sizeof(int), cudaMemcpyHostToDevice);

    // Define thread configuration
    int threadsPerBlock = 256;
    int blocksPerGridE = (E + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridV = (V + threadsPerBlock - 1) / threadsPerBlock;

    // Adjust weights
    adjustWeights<<<blocksPerGridE, threadsPerBlock>>>(d_edges, E, MOD);
    cudaDeviceSynchronize();

    // Initialize arrays
    initializeArrays<<<blocksPerGridV, threadsPerBlock>>>(d_component, d_minEdgeIdx, d_minEdgeWeight, V);
    cudaDeviceSynchronize();

    // Main Boruvka loop - repeat until no more components can be merged
    do
    {
        h_changed = 0;
        cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);

        // Reset min edges for each component
        resetMinEdges<<<blocksPerGridV, threadsPerBlock>>>(d_minEdgeIdx, d_minEdgeWeight, V);
        cudaDeviceSynchronize();

        // Find minimum outgoing edge for each component
        findMinEdges<<<blocksPerGridE, threadsPerBlock>>>(d_edges, d_component, d_minEdgeIdx, d_minEdgeWeight, V, E);
        cudaDeviceSynchronize();

        // Update edge indices
        updateEdgeIndices<<<blocksPerGridE, threadsPerBlock>>>(d_edges, d_component, d_minEdgeIdx, d_minEdgeWeight, V, E);
        cudaDeviceSynchronize();

        // Merge components
        mergeComponents<<<blocksPerGridV, threadsPerBlock>>>(d_edges, d_component, d_minEdgeIdx, d_changed, d_mstWeight, V, MOD);
        cudaDeviceSynchronize();

        // Copy changed flag back to host
        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);

    } while (h_changed);

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
    cout << h_mstWeight << endl;
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