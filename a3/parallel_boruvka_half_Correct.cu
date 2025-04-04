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

// Find minimum outgoing edge for each component - using edge-parallel approach
__global__ void findMinEdges(Edge *edges, int *component, int *minEdgeIdx, int *minEdgeWeight, int V, int E)
{
    __shared__ int localMinIdx[256];    // Shared memory for thread block
    __shared__ int localMinWeight[256]; // Shared memory for thread block

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize shared memory
    if (tid < V)
    {
        localMinIdx[tid] = -1;
        localMinWeight[tid] = INT_MAX;
    }
    __syncthreads();

    // Process edges
    if (idx < E)
    {
        int u = edges[idx].src;
        int v = edges[idx].dest;
        int w = edges[idx].weight;

        int compU = component[u];
        int compV = component[v];

        if (compU != compV)
        {
            // Update minimum edge for compU using atomic operations on shared memory
            if (compU < 256)
            {
                atomicMin(&localMinWeight[compU], w);
            }

            // Update minimum edge for compV using atomic operations on shared memory
            if (compV < 256)
            {
                atomicMin(&localMinWeight[compV], w);
            }
        }
    }
    __syncthreads();

    // Update the global minima using atomic operations
    if (tid < V && localMinWeight[tid] != INT_MAX)
    {
        atomicMin(&minEdgeWeight[tid], localMinWeight[tid]);
    }
    __syncthreads();

    // Update edge indices
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
                atomicExch(&minEdgeIdx[compU], idx);
            }
            if (w == minEdgeWeight[compV])
            {
                atomicExch(&minEdgeIdx[compV], idx);
            }
        }
    }
}

// Perform component merging with correct synchronization
__global__ void mergeComponents(Edge *edges, int *component, int *minEdgeIdx, int *changed, int *mstWeight, int V, int MOD)
{
    __shared__ int selectedEdges[256]; // Store edge indices to add to MST
    __shared__ int numSelected;

    // Initialize
    if (threadIdx.x == 0)
    {
        numSelected = 0;
    }
    __syncthreads();

    // Each thread processes one component
    int compId = threadIdx.x + blockIdx.x * blockDim.x;
    if (compId < V)
    {
        int edgeIdx = minEdgeIdx[compId];
        if (edgeIdx != -1)
        {
            int u = edges[edgeIdx].src;
            int v = edges[edgeIdx].dest;
            int compU = component[u];
            int compV = component[v];

            if (compU != compV && compId == compU)
            { // Only process if this is the source component
                int pos = atomicAdd(&numSelected, 1);
                if (pos < 256)
                {
                    selectedEdges[pos] = edgeIdx;
                }
            }
        }
    }
    __syncthreads();

    // Process selected edges - each thread processes one edge
    if (threadIdx.x < numSelected)
    {
        int edgeIdx = selectedEdges[threadIdx.x];
        int u = edges[edgeIdx].src;
        int v = edges[edgeIdx].dest;
        int compU = component[u];
        int compV = component[v];

        // Add to MST weight
        atomicAdd(mstWeight, edges[edgeIdx].weight);
        if (*mstWeight >= MOD)
        {
            atomicAdd(mstWeight, -MOD);
        }

        // Mark as changed
        *changed = 1;
    }
    __syncthreads();

    // Update all vertices in the graph with new component IDs
    for (int i = compId; i < V; i += blockDim.x * gridDim.x)
    {
        for (int j = 0; j < numSelected; j++)
        {
            int edgeIdx = selectedEdges[j];
            int u = edges[edgeIdx].src;
            int v = edges[edgeIdx].dest;
            int compU = component[u];
            int compV = component[v];

            if (component[i] == compV)
            {
                component[i] = compU;
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

// Kernel to check for distinct components
__global__ void checkComponents(int *component, int *numComponents, int V)
{
    __shared__ int localComponents[256];
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize
    if (tid < 256)
    {
        localComponents[tid] = -1;
    }
    __syncthreads();

    // Mark components
    if (idx < V)
    {
        int comp = component[idx];
        if (comp < 256)
        {
            localComponents[comp] = 1;
        }
    }
    __syncthreads();

    // Count components
    if (tid == 0)
    {
        int count = 0;
        for (int i = 0; i < V && i < 256; i++)
        {
            if (localComponents[i] == 1)
            {
                count++;
            }
        }
        atomicAdd(numComponents, count);
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
    int *d_numComponents;
    int h_mstWeight = 0;
    int h_changed;
    int h_numComponents;
    int prev_components = 0;

    cudaMalloc(&d_edges, E * sizeof(Edge));
    cudaMalloc(&d_mstWeight, sizeof(int));
    cudaMalloc(&d_component, V * sizeof(int));
    cudaMalloc(&d_minEdgeIdx, V * sizeof(int));
    cudaMalloc(&d_minEdgeWeight, V * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));
    cudaMalloc(&d_numComponents, sizeof(int));

    cudaMemcpy(d_edges, edges.data(), E * sizeof(Edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mstWeight, &h_mstWeight, sizeof(int), cudaMemcpyHostToDevice);

    // Define thread configuration
    int threadsPerBlock = min(256, V); // Limit to V or hardware max
    int blocksPerGridE = (E + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridV = (V + threadsPerBlock - 1) / threadsPerBlock;

    // Adjust weights
    adjustWeights<<<blocksPerGridE, threadsPerBlock>>>(d_edges, E, MOD);
    cudaDeviceSynchronize();

    // Initialize arrays
    initializeArrays<<<blocksPerGridV, threadsPerBlock>>>(d_component, d_minEdgeIdx, d_minEdgeWeight, V);
    cudaDeviceSynchronize();

    // Main Boruvka loop - we need at most log(V) iterations for convergence
    int max_iterations = 0;
    while (max_iterations < V - 1)
    { // At most V-1 edges in MST
        h_changed = 0;
        cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);

        // Reset min edges for each component
        resetMinEdges<<<blocksPerGridV, threadsPerBlock>>>(d_minEdgeIdx, d_minEdgeWeight, V);
        cudaDeviceSynchronize();

        // Find minimum outgoing edge for each component
        findMinEdges<<<blocksPerGridE, threadsPerBlock>>>(d_edges, d_component, d_minEdgeIdx, d_minEdgeWeight, V, E);
        cudaDeviceSynchronize();

        // Merge components
        mergeComponents<<<blocksPerGridV, threadsPerBlock>>>(d_edges, d_component, d_minEdgeIdx, d_changed, d_mstWeight, V, MOD);
        cudaDeviceSynchronize();

        // Copy changed flag back to host
        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);

        // Check for termination - if no changes were made or max iterations reached
        if (!h_changed)
        {
            break;
        }

        max_iterations++;
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
    h_mstWeight %= MOD; // Ensure final result is within modulo bounds
    cout << h_mstWeight << endl;
    // Free GPU memory
    cudaFree(d_edges);
    cudaFree(d_mstWeight);
    cudaFree(d_component);
    cudaFree(d_minEdgeIdx);
    cudaFree(d_minEdgeWeight);
    cudaFree(d_changed);
    cudaFree(d_numComponents);

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