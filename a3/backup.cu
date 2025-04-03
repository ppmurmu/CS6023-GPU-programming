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
const int INF = -1;

struct Edge
{
    int src, dest, weight;
    int factor; // Terrain factor multiplier
};

// adjust weights kernel
__global__ void adjustWeights(Edge *edges, int *adjMatrix, int E, int V, int MOD)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= E)
        return;

    int u = edges[i].src;
    int v = edges[i].dest;
    int w = edges[i].weight;

    // Adjust weight based on terrain factor
    int factor = 1;
    if (edges[i].factor == 2)
        factor = 2;
    else if (edges[i].factor == 5)
        factor = 5;
    else if (edges[i].factor == 3)
        factor = 3;

    int adjustedWeight = (w * factor) % MOD;

    // Store adjusted weight in adjacency matrix
    adjMatrix[u * V + v] = adjustedWeight;
    adjMatrix[v * V + u] = adjustedWeight; // Undirected graph
}

__global__ void computeMST(int *adjMatrix, int *mstWeight, int *component, int *minEdgeIdx, int *minEdgeWeight, int V, int MOD)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= V)
        return; // Each thread handles a single vertex

    // Initialize components (each vertex starts in its own component)
    component[tid] = tid;

    __syncthreads();

    bool changed;
    do
    {
        changed = false;

        // Reset min edges for each component (parallelized)
        minEdgeIdx[tid] = -1;
        minEdgeWeight[tid] = INT_MAX;

        __syncthreads();

        // Find the minimum outgoing edge for each component (parallelized)
        for (int v = 0; v < V; v++)
        {
            int w = adjMatrix[tid * V + v];
            if (w != INF)
            {
                int compU = component[tid];
                int compV = component[v];

                if (compU != compV)
                {
                    atomicMin(&minEdgeWeight[compU], w);
                    if (w == minEdgeWeight[compU])
                        minEdgeIdx[compU] = tid * V + v; // Store edge index
                }
            }
        }

        __syncthreads();

        // Merge components in parallel
        if (minEdgeIdx[tid] != -1)
        {
            int edgeIdx = minEdgeIdx[tid];
            int u = edgeIdx / V;
            int v = edgeIdx % V;
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

                atomicAdd(mstWeight, minEdgeWeight[tid]);
                atomicExch(&changed, true);
            }
        }

        __syncthreads();
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

    int *d_adjMatrix;
    cudaMalloc(&d_adjMatrix, V * V * sizeof(int));
    cudaMemset(d_adjMatrix, INF, V * V * sizeof(int)); // Initialize to INF (no edges)

    // adjust weight kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (E + threadsPerBlock - 1) / threadsPerBlock;

    adjustWeights<<<blocksPerGrid, threadsPerBlock>>>(d_edges, d_adjMatrix, E, V, MOD);
    cudaDeviceSynchronize(); // Ensure all updates are completed before MST

    // Launch CUDA Kernel (single-threaded execution)
    int threadsPerBlock = 256;
    int blocksPerGrid = (V + threadsPerBlock - 1) / threadsPerBlock;

    computeMST<<<blocksPerGrid, threadsPerBlock>>>(d_adjMatrix, d_mstWeight, d_component, d_minEdgeIdx, d_minEdgeWeight, V, MOD);
    cudaDeviceSynchronize();

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
