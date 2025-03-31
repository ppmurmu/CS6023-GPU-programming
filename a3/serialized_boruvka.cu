#include <iostream>
#include <vector>
#include <string>
#include <cuda.h>
#include <chrono>
#include <fstream>
using namespace std;
using std::cin;
using std::cout;
#define MOD 1000000007

struct Edge
{
    int src, dest, weight;
    string type; // Terrain type
};

void adjustEdgeWeights(vector<Edge> &edges)
{
    cout << "adjust weights---";
    for (auto &edge : edges)
    {
        if (edge.type == "green")
        {
            edge.weight *= 2;
        }
        else if (edge.type == "traffic")
        {
            edge.weight *= 5;
        }
        else if (edge.type == "dept")
        {
            edge.weight *= 3;
        }
    }
}

// CUDA kernel-compatible Edge structure (without std::string)
struct CudaEdge
{
    int src, dest, weight;
};

// Serialized CUDA Kernel for Boruvka's MST Calculation
__global__ void computeMST(CudaEdge *edges, int *mstWeight, int *component, int *minEdgeIdx, int *minEdgeWeight, int V, int E)
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
                    *mstWeight += edges[edgeIdx].weight;
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
        cin >> edges[i].src >> edges[i].dest >> edges[i].weight >> edges[i].type;
    }

    // Adjust weights based on terrain factors
    adjustEdgeWeights(edges);

    // Convert to CUDA-compatible edges
    vector<CudaEdge> cudaEdges(E);
    for (int i = 0; i < E; i++)
    {
        cudaEdges[i].src = edges[i].src;
        cudaEdges[i].dest = edges[i].dest;
        cudaEdges[i].weight = edges[i].weight;
    }

    // Allocate memory on GPU
    CudaEdge *d_edges;
    int *d_mstWeight;
    int *d_component;
    int *d_minEdgeIdx;
    int *d_minEdgeWeight;
    int h_mstWeight = 0;

    cudaMalloc(&d_edges, E * sizeof(CudaEdge));
    cudaMalloc(&d_mstWeight, sizeof(int));
    cudaMalloc(&d_component, V * sizeof(int));
    cudaMalloc(&d_minEdgeIdx, V * sizeof(int));
    cudaMalloc(&d_minEdgeWeight, V * sizeof(int));

    cudaMemcpy(d_edges, cudaEdges.data(), E * sizeof(CudaEdge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mstWeight, &h_mstWeight, sizeof(int), cudaMemcpyHostToDevice);

    // Set initial values
    vector<int> h_minEdgeIdx(V, -1);
    vector<int> h_minEdgeWeight(V, INT_MAX);
    cudaMemcpy(d_minEdgeIdx, h_minEdgeIdx.data(), V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_minEdgeWeight, h_minEdgeWeight.data(), V * sizeof(int), cudaMemcpyHostToDevice);

    //--------------TIMER-----
    auto start = chrono::high_resolution_clock::now();

    // Launch CUDA Kernel (single-threaded execution)
    computeMST<<<1, 1>>>(d_edges, d_mstWeight, d_component, d_minEdgeIdx, d_minEdgeWeight, V, E);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

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

    // Print the total MST weight
    cout << h_mstWeight << endl;

    // Free GPU memory
    cudaFree(d_edges);
    cudaFree(d_mstWeight);
    cudaFree(d_component);
    cudaFree(d_minEdgeIdx);
    cudaFree(d_minEdgeWeight);

    return 0;
}
