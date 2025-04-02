// CUDA Kernel for adjusting edge weights
__global__ void adjustEdgeWeightsKernel(CudaEdge *edges, int *terrainTypes, int E)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < E)
    {
        if (terrainTypes[idx] == 0)
        { // green
            edges[idx].weight = (edges[idx].weight * 2) % MOD;
        }
        else if (terrainTypes[idx] == 1)
        { // traffic
            edges[idx].weight = (edges[idx].weight * 5) % MOD;
        }
        else if (terrainTypes[idx] == 2)
        { // dept
            edges[idx].weight = (edges[idx].weight * 3) % MOD;
        }
        else
        { // default case
            edges[idx].weight = (edges[idx].weight * 1) % MOD;
        }
    }
}

void parallelAdjustEdgeWeights(vector<Edge> &edges, vector<CudaEdge> &cudaEdges)
{
    int E = edges.size();

    // Map terrain types to integers
    vector<int> terrainTypes(E);
    for (int i = 0; i < E; i++)
    {
        if (edges[i].type == "green")
        {
            terrainTypes[i] = 0;
        }
        else if (edges[i].type == "traffic")
        {
            terrainTypes[i] = 1;
        }
        else if (edges[i].type == "dept")
        {
            terrainTypes[i] = 2;
        }
        else
        {
            terrainTypes[i] = -1; // default case
        }
    }

    // Allocate memory on GPU
    CudaEdge *d_edges;
    int *d_terrainTypes;

    cudaMalloc(&d_edges, E * sizeof(CudaEdge));
    cudaMalloc(&d_terrainTypes, E * sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_edges, cudaEdges.data(), E * sizeof(CudaEdge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_terrainTypes, terrainTypes.data(), E * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with enough threads and blocks
    int blockSize = 256;                             // Number of threads per block
    int numBlocks = (E + blockSize - 1) / blockSize; // Number of blocks
    adjustEdgeWeightsKernel<<<numBlocks, blockSize>>>(d_edges, d_terrainTypes, E);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy adjusted weights back to host
    cudaMemcpy(cudaEdges.data(), d_edges, E * sizeof(CudaEdge), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_edges);
    cudaFree(d_terrainTypes);
}
