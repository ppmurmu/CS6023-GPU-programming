#include <iostream>
#include <vector>
#include <string>
#include <cuda.h>
#include <chrono>
#include <fstream>
#include <climits>
using namespace std;

const int MOD = 1000000007;

struct Edge
{
    int src, dest, weight;
    int factor; // Terrain factor
};

// Adjust weights based on terrain
__global__ void adjustWeights(Edge *edges, int E, int MOD)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < E)
    {
        edges[idx].weight = (1LL * edges[idx].weight * edges[idx].factor) % MOD;
    }
}

// Init components: component[i] = i
__global__ void initComponents(int *component, int V)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < V)
    {
        component[idx] = idx;
    }
}

// Reset minEdgeIdx and minWeight arrays
__global__ void resetMinArrays(int *minEdgeIdx, int *minEdgeWeight, int V)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < V)
    {
        minEdgeIdx[idx] = -1;
        minEdgeWeight[idx] = INT_MAX;
    }
}

// Parallel find min edge for each component
__global__ void findMinEdges(Edge *edges, int E, int *component, int *minEdgeIdx, int *minEdgeWeight)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < E)
    {
        Edge e = edges[idx];
        int u = e.src;
        int v = e.dest;
        int w = e.weight;

        int cu = component[u];
        int cv = component[v];

        if (cu != cv)
        {
            atomicMin(&minEdgeWeight[cu], w);
            atomicMin(&minEdgeWeight[cv], w);
        }
    }
}

// Match min weight edges and set their indices
__global__ void selectMinEdges(Edge *edges, int E, int *component, int *minEdgeIdx, int *minEdgeWeight)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < E)
    {
        Edge e = edges[idx];
        int u = e.src;
        int v = e.dest;
        int w = e.weight;

        int cu = component[u];
        int cv = component[v];

        if (cu != cv)
        {
            if (w == minEdgeWeight[cu])
                atomicCAS(&minEdgeIdx[cu], -1, idx);
            if (w == minEdgeWeight[cv])
                atomicCAS(&minEdgeIdx[cv], -1, idx);
        }
    }
}

// Sequential component merge
__global__ void mergeComponents(Edge *edges, int *component, int *minEdgeIdx, int *mstWeight, bool *changed, int V, int MOD)
{
    if (threadIdx.x + blockIdx.x * blockDim.x > 0)
        return; // Only one thread runs

    for (int i = 0; i < V; ++i)
    {
        int edgeIdx = minEdgeIdx[i];
        if (edgeIdx != -1)
        {
            Edge e = edges[edgeIdx];
            int u = e.src, v = e.dest;
            int cu = component[u];
            int cv = component[v];

            if (cu != cv)
            {
                int newComp = min(cu, cv);
                int oldComp = max(cu, cv);
                for (int j = 0; j < V; ++j)
                {
                    if (component[j] == oldComp)
                        component[j] = newComp;
                }
                *mstWeight = (*mstWeight + e.weight) % MOD;
                *changed = true;
            }
        }
    }
}

int main()
{
    int V, E;
    cin >> V >> E;
    vector<Edge> h_edges(E);

    for (int i = 0; i < E; i++)
    {
        string type;
        cin >> h_edges[i].src >> h_edges[i].dest >> h_edges[i].weight >> type;

        if (type == "green")
            h_edges[i].factor = 2;
        else if (type == "traffic")
            h_edges[i].factor = 5;
        else if (type == "dept")
            h_edges[i].factor = 3;
        else
            h_edges[i].factor = 1;
    }

    auto start = chrono::high_resolution_clock::now();

    // Allocate memory
    Edge *d_edges;
    int *d_component;
    int *d_minEdgeIdx;
    int *d_minEdgeWeight;
    int *d_mstWeight;
    bool *d_changed;

    cudaMalloc(&d_edges, E * sizeof(Edge));
    cudaMalloc(&d_component, V * sizeof(int));
    cudaMalloc(&d_minEdgeIdx, V * sizeof(int));
    cudaMalloc(&d_minEdgeWeight, V * sizeof(int));
    cudaMalloc(&d_mstWeight, sizeof(int));
    cudaMalloc(&d_changed, sizeof(bool));

    int h_mstWeight = 0;
    cudaMemcpy(d_edges, h_edges.data(), E * sizeof(Edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mstWeight, &h_mstWeight, sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksV = (V + threadsPerBlock - 1) / threadsPerBlock;
    int blocksE = (E + threadsPerBlock - 1) / threadsPerBlock;

    adjustWeights<<<blocksE, threadsPerBlock>>>(d_edges, E, MOD);
    cudaDeviceSynchronize();

    initComponents<<<blocksV, threadsPerBlock>>>(d_component, V);
    cudaDeviceSynchronize();

    bool h_changed = true;

    while (h_changed)
    {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        resetMinArrays<<<blocksV, threadsPerBlock>>>(d_minEdgeIdx, d_minEdgeWeight, V);
        cudaDeviceSynchronize();

        findMinEdges<<<blocksE, threadsPerBlock>>>(d_edges, E, d_component, d_minEdgeIdx, d_minEdgeWeight);
        cudaDeviceSynchronize();

        selectMinEdges<<<blocksE, threadsPerBlock>>>(d_edges, E, d_component, d_minEdgeIdx, d_minEdgeWeight);
        cudaDeviceSynchronize();

        mergeComponents<<<1, 1>>>(d_edges, d_component, d_minEdgeIdx, d_mstWeight, d_changed, V, MOD);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cudaMemcpy(&h_mstWeight, d_mstWeight, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_edges);
    cudaFree(d_component);
    cudaFree(d_minEdgeIdx);
    cudaFree(d_minEdgeWeight);
    cudaFree(d_mstWeight);
    cudaFree(d_changed);

    ofstream file("cuda.out");
    file << h_mstWeight << "\n";
    file.close();

    ofstream file2("cuda_timing.out");
    file2 << elapsed.count() << "\n";
    file2.close();

    return 0;
}
