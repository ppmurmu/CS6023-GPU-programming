#include <iostream>
#include <vector>
#include <string>
#include <cuda.h>
#include <chrono>
#include <fstream>
#include <climits>
using namespace std;
using std::cin;
using std::cout;

const int MOD = 1000000007;

struct Edge
{
    int src, dest, weight;
    int factor; // constraint factor multiplier
};

// Kernel to adjust edge weights based on constraint factors.
__global__ void adjustWeights(Edge *edges, int E)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < E)
    {
        edges[idx].weight = (edges[idx].weight * edges[idx].factor);
    }
}

// Device function to perform find
__device__ int find(int *parent, int i)
{
    while (i != parent[i])
    {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    return i;
}

// Kernel to initialize each vertex as its own component
__global__ void initComponents(int *d_component, int V)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < V)
    {
        d_component[i] = i;
    }
}

// in the 64 bit candidate edge, i'm using upper  32 bits are storing the weight and the lower 32 bits store the edge index.
// Also each component’s candidate is initialized to INF.
__global__ void initCandidates(unsigned long long *d_minEdgeCandidate, int V, unsigned long long INF)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < V)
    {
        d_minEdgeCandidate[i] = INF;
    }
}

// For each edge, find the current roots of its endpoints. If they lie in different components,
// pack the candidate value and use atomicMin to update the candidate for each component.
__global__ void findMinEdgeCandidates(Edge *d_edges, int E, int *d_component, unsigned long long *d_minEdgeCandidate)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < E)
    {
        int u = d_edges[idx].src;
        int v = d_edges[idx].dest;
        int comp_u = find(d_component, u);
        int comp_v = find(d_component, v);

        if (comp_u == comp_v)
            return;
        unsigned long long candidate = ((unsigned long long)d_edges[idx].weight << 32) | ((unsigned int)idx);
        atomicMin(&d_minEdgeCandidate[comp_u], candidate);
        atomicMin(&d_minEdgeCandidate[comp_v], candidate);
    }
}

// performing union operation and adding edge weight to MST. also update change counter
__global__ void unionComponents(Edge *d_edges, int *d_component, unsigned long long *d_minEdgeCandidate,
                                int V, int *d_mstWeight, int *d_change, unsigned long long INF)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < V)
    {
        // Process only if i is a root
        if (d_component[i] == i)
        {
            unsigned long long candidate = d_minEdgeCandidate[i];
            if (candidate == INF)
                return; // No candidate edge found

            int edgeIdx = candidate & 0xFFFFFFFF;
            int u = d_edges[edgeIdx].src;
            int v = d_edges[edgeIdx].dest;
            int root_u = find(d_component, u);
            int root_v = find(d_component, v);
            if (root_u == root_v)
                return;
            int new_root = min(root_u, root_v);
            int old_root = max(root_u, root_v);

            int old = atomicCAS(&d_component[old_root], old_root, new_root);

            if (old == old_root)
            { // Only one thread will succeed here.
                atomicAdd(d_change, 1);
                atomicAdd(d_mstWeight, d_edges[edgeIdx].weight);
            }
        }
    }
}

// Kernel to compress component trees (path compression) for all vertices.
__global__ void compressComponents(int *d_component, int V)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < V)
    {
        while (d_component[i] != d_component[d_component[i]])
        {
            d_component[i] = d_component[d_component[i]];
        }
    }
}

int main(int argc, char **argv)
{
    int V, E;
    cin >> V >> E;
    vector<Edge> edges(E);

    // Read input edges.
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

    Edge *d_edges;
    int *d_mstWeight;
    int *d_component;

    // variable to store 64-bit candidate array for storing the best (min) edge per component.
    unsigned long long *d_minEdgeCandidate;

    // flag to check if any union occurred in an iteration.
    int *d_change;

    int h_mstWeight = 0;

    // MALLOC here
    cudaMalloc(&d_edges, E * sizeof(Edge));
    cudaMalloc(&d_mstWeight, sizeof(int));
    cudaMalloc(&d_component, V * sizeof(int));
    cudaMalloc(&d_minEdgeCandidate, V * sizeof(unsigned long long));
    cudaMalloc(&d_change, sizeof(int));

    // MEMCPY here
    cudaMemcpy(d_edges, edges.data(), E * sizeof(Edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mstWeight, &h_mstWeight, sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 512;
    int blocksEdges = (E + threadsPerBlock - 1) / threadsPerBlock;
    int blocksVertices = (V + threadsPerBlock - 1) / threadsPerBlock;

    //-------TIMER starts here---------------
    auto start = chrono::high_resolution_clock::now();

    //--adjust weights kernel----
    adjustWeights<<<blocksEdges, threadsPerBlock>>>(d_edges, E);

    // Initializing components
    initComponents<<<blocksVertices, threadsPerBlock>>>(d_component, V);

    //  INF candidate value: the upper 32 bits as INT_MAX and lower 32 bits as all ones.
    unsigned long long INF = (((unsigned long long)INT_MAX) << 32) | 0xFFFFFFFFULL;

    // Main loop of Boruvka's algorithm
    bool finished = false;
    while (!finished)
    {
        // Reset change counter to 0.
        cudaMemset(d_change, 0, sizeof(int));

        // Reinitialize candidate array
        initCandidates<<<blocksVertices, threadsPerBlock>>>(d_minEdgeCandidate, V, INF);

        // Each edge update the candidate for its two endpoints components
        findMinEdgeCandidates<<<blocksEdges, threadsPerBlock>>>(d_edges, E, d_component, d_minEdgeCandidate);

        // union component kernel
        unionComponents<<<blocksVertices, threadsPerBlock>>>(d_edges, d_component, d_minEdgeCandidate, V, d_mstWeight, d_change, INF);

        // compress paths so that each vertex directly points to the representative
        compressComponents<<<blocksVertices, threadsPerBlock>>>(d_component, V);

        // Check if any union happened in this iteration
        int h_change = 0;
        cudaMemcpy(&h_change, d_change, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_change == 0)
        {
            finished = true;
        }
    }

    //--------------TIMER ends----------
    auto end = chrono::high_resolution_clock::now();

    // Copy MST result back to host.
    cudaMemcpy(&h_mstWeight, d_mstWeight, sizeof(int), cudaMemcpyDeviceToHost);

    chrono::duration<double> elapsed1 = end - start;

    cudaDeviceSynchronize();

    h_mstWeight = h_mstWeight % MOD;

    //---print the output------
    cout << h_mstWeight << endl;

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cerr << "CUDA error: " << cudaGetErrorString(err) << endl;
    }

    cout << h_mstWeight << endl;

    // Free GPU memory.
    cudaFree(d_edges);
    cudaFree(d_mstWeight);
    cudaFree(d_component);
    cudaFree(d_minEdgeCandidate);
    cudaFree(d_change);

    // Write MST weight to file.
    // std::ofstream file("cuda.out");
    // if (file.is_open())
    // {
    //     file << h_mstWeight;
    //     file << "\n";
    //     file.close();
    // }
    // else
    // {
    //     std::cout << "Unable to open file";
    // }

    // // Write timing information to file.
    // std::ofstream file2("cuda_timing.out");
    // if (file2.is_open())
    // {
    //     file2 << elapsed1.count() << "\n";
    //     file2.close();
    // }
    // else
    // {
    //     std::cout << "Unable to open file";
    // }

    return 0;
}
