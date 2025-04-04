#include <iostream>
#include <vector>
#include <string>
#include <cuda.h>
#include <chrono>
#include <fstream>
using namespace std;

const int MOD = 1000000007;
const int INF = 1e9;

struct Edge
{
    int src, dest, weight;
    int factor;
};

// Kernel: adjust weights of edges using their factors
__global__ void adjustWeights(Edge *edges, int E, int MOD)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= E)
        return;

    int w = edges[i].weight;
    int factor = edges[i].factor;
    edges[i].weight = (1LL * w * factor) % MOD;
}

// Placeholder kernel for MST – currently serial
__global__ void computeMST(Edge *edges, int *mstWeight, int V, int E)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    // Very basic serial Prim-like logic for testing
    bool inMST[1000]; // adjust or move to dynamic memory if V > 1000
    int minEdge[1000];
    for (int i = 0; i < V; i++)
    {
        inMST[i] = false;
        minEdge[i] = INF;
    }

    minEdge[0] = 0;
    *mstWeight = 0;

    for (int count = 0; count < V; count++)
    {
        int u = -1;
        for (int i = 0; i < V; i++)
            if (!inMST[i] && (u == -1 || minEdge[i] < minEdge[u]))
                u = i;

        if (minEdge[u] == INF)
            return;
        inMST[u] = true;
        *mstWeight += minEdge[u];

        for (int i = 0; i < E; i++)
        {
            int src = edges[i].src;
            int dest = edges[i].dest;
            int w = edges[i].weight;
            if (src == u && !inMST[dest] && w < minEdge[dest])
            {
                minEdge[dest] = w;
            }
            else if (dest == u && !inMST[src] && w < minEdge[src])
            {
                minEdge[src] = w;
            }
        }
    }
}

int main(int argc, char **argv)
{
    int V, E;
    cin >> V >> E;
    vector<Edge> edges(E);

    for (int i = 0; i < E; i++)
    {
        string type;
        cin >> edges[i].src >> edges[i].dest >> edges[i].weight >> type;
        if (type == "green")
            edges[i].factor = 2;
        else if (type == "traffic")
            edges[i].factor = 5;
        else if (type == "dept")
            edges[i].factor = 3;
        else
            edges[i].factor = 1;
    }

    auto start = chrono::high_resolution_clock::now();

    Edge *d_edges;
    int *d_mstWeight;
    int h_mstWeight = 0;

    cudaMalloc(&d_edges, E * sizeof(Edge));
    cudaMalloc(&d_mstWeight, sizeof(int));

    cudaMemcpy(d_edges, edges.data(), E * sizeof(Edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mstWeight, &h_mstWeight, sizeof(int), cudaMemcpyHostToDevice);

    // Parallel adjustment of weights
    adjustWeights<<<(E + 255) / 256, 256>>>(d_edges, E, MOD);
    cudaDeviceSynchronize();

    // Placeholder serial MST kernel
    computeMST<<<1, 1>>>(d_edges, d_mstWeight, V, E);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_mstWeight, d_mstWeight, sizeof(int), cudaMemcpyDeviceToHost);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cudaFree(d_edges);
    cudaFree(d_mstWeight);

    ofstream file("cuda.out");
    file << h_mstWeight << "\n";
    file.close();
    ofstream file2("cuda_timing.out");
    file2 << elapsed.count() << "\n";
    file2.close();

    return 0;
}
