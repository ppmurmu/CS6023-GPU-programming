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
const int INF = 1e9;

struct Edge
{
    int src, dest, weight;
    int factor;
};

__global__ void adjustWeights(Edge *edges, int *adjMatrix, int E, int V, int MOD)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= E)
        return;

    int u = edges[i].src;
    int v = edges[i].dest;
    int w = edges[i].weight;

    int factor = edges[i].factor;
    int adjustedWeight = (w * factor) % MOD;

    adjMatrix[u * V + v] = adjustedWeight;
    adjMatrix[v * V + u] = adjustedWeight;
}

__global__ void computeMST(int *adjMatrix, int *mstWeight, int V)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return; // Ensure only one thread executes

    bool inMST[1000]; // Assuming V <= 1000, adjust size if needed
    int minWeight[1000];
    for (int i = 0; i < V; i++)
    {
        inMST[i] = false;
        minWeight[i] = INF;
    }
    minWeight[0] = 0;
    *mstWeight = 0;

    for (int i = 0; i < V; i++)
    {
        int u = -1;
        for (int j = 0; j < V; j++)
        {
            if (!inMST[j] && (u == -1 || minWeight[j] < minWeight[u]))
            {
                u = j;
            }
        }

        if (minWeight[u] == INF)
            return;
        inMST[u] = true;
        *mstWeight += minWeight[u];

        for (int v = 0; v < V; v++)
        {
            if (!inMST[v] && adjMatrix[u * V + v] != INF && adjMatrix[u * V + v] < minWeight[v])
            {
                minWeight[v] = adjMatrix[u * V + v];
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
    int *d_adjMatrix, *d_mstWeight;
    int h_mstWeight = 0;

    cudaMalloc(&d_edges, E * sizeof(Edge));
    cudaMalloc(&d_adjMatrix, V * V * sizeof(int));
    cudaMalloc(&d_mstWeight, sizeof(int));

    cudaMemcpy(d_edges, edges.data(), E * sizeof(Edge), cudaMemcpyHostToDevice);
    vector<int> h_adjMatrix(V * V, INF);
    cudaMemcpy(d_adjMatrix, h_adjMatrix.data(), V * V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mstWeight, &h_mstWeight, sizeof(int), cudaMemcpyHostToDevice);

    adjustWeights<<<(E + 255) / 256, 256>>>(d_edges, d_adjMatrix, E, V, MOD);
    cudaDeviceSynchronize();

    computeMST<<<1, 1>>>(d_adjMatrix, d_mstWeight, V);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_mstWeight, d_mstWeight, sizeof(int), cudaMemcpyDeviceToHost);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cudaFree(d_edges);
    cudaFree(d_adjMatrix);
    cudaFree(d_mstWeight);

    ofstream file("cuda.out");
    file << h_mstWeight << "\n";
    file.close();
    ofstream file2("cuda_timing.out");
    file2 << elapsed.count() << "\n";
    file2.close();

    return 0;
}
