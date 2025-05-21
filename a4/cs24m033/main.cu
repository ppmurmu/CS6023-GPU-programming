// CS24M033 GPU Assignment 4
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <algorithm>
#include <climits>
#include <cmath>
#include <unordered_map>
#include <utility>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

// Structure to represent a road between cities
struct Road
{
    int to;
    int length;
    int capacity;
};

// CUDA error checking
#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
inline void check_cuda(cudaError_t result, const char *func, const char *file, int line)
{
    if (result != cudaSuccess)
    {
        cerr << "CUDA error: " << cudaGetErrorString(result) << " at " << file << ":" << line << " '" << func << "'" << endl;
        exit(1);
    }
}

// CUDA kernel for initializing distance matrix
__global__ void initializeDistancesKernel(int *d_distances, int num_cities)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_cities * num_cities)
    {
        int source = tid / num_cities;
        int dest = tid % num_cities;

        if (source == dest)
        {
            d_distances[tid] = 0; // Distance to self is 0
        }
        else
        {
            d_distances[tid] = INT_MAX; // Initially, all other distances are infinity
        }
    }
}

// Helper function for finding min distance vertex not in visited
__device__ int minDistance(int *dist, bool *visited, int num_cities)
{
    int min = INT_MAX, min_index = -1;

    for (int v = 0; v < num_cities; v++)
    {
        if (!visited[v] && dist[v] <= min)
        {
            min = dist[v];
            min_index = v;
        }
    }
    return min_index;
}

// CUDA kernel for Dijkstra's algorithm (multiple sources in parallel)
__global__ void parallelDijkstraKernel(int *d_graph, int *d_distances, int *d_paths, int num_cities)
{
    int source = blockIdx.x; // Each block handles one source vertex

    if (source >= num_cities)
        return;

    // Create a local distance array for this source
    int *dist = new int[num_cities];
    bool *visited = new bool[num_cities];

    // Initialize
    for (int i = 0; i < num_cities; i++)
    {
        dist[i] = INT_MAX;
        visited[i] = false;
    }
    dist[source] = 0;

    // Find shortest path for all vertices
    for (int count = 0; count < num_cities - 1; count++)
    {
        int u = minDistance(dist, visited, num_cities);

        if (u == -1)
            break; // No more reachable vertices

        visited[u] = true;

        // Update dist value of adjacent vertices
        for (int v = 0; v < num_cities; v++)
        {
            // Get edge weight from the flattened graph matrix
            int edge = d_graph[u * num_cities + v];

            // Update dist[v] if not visited, there is an edge, and path through u is shorter
            if (!visited[v] && edge != 0 && dist[u] != INT_MAX && dist[u] + edge < dist[v])
            {
                dist[v] = dist[u] + edge;
                d_paths[source * num_cities + v] = u; // predecessor for path reconstruction
            }
        }
    }

    // Copy results back to global memory
    for (int i = 0; i < num_cities; i++)
    {
        d_distances[source * num_cities + i] = dist[i];
    }

    delete[] dist;
    delete[] visited;
}

// Kernel for path reconstruction
__global__ void pathReconstructionKernel(int *d_paths, int *d_reconstructed_paths, int *d_path_lengths, int num_cities, int max_path_length)
{
    int source = blockIdx.x;
    int dest = threadIdx.x;

    if (source >= num_cities || dest >= num_cities)
        return;

    // Skip self-paths (handled separately)
    if (source == dest)
    {
        d_path_lengths[source * num_cities + dest] = 1;
        d_reconstructed_paths[(source * num_cities + dest) * max_path_length] = source;
        return;
    }

    // Check if path exists
    if (d_paths[source * num_cities + dest] == -1)
    {
        d_path_lengths[source * num_cities + dest] = 0;
        return;
    }

    // Count the path length first
    int length = 0;
    int current = dest;
    int temp_path[1000]; // Temporary buffer for path

    // Follow predecessors to get path length
    while (current != -1 && current != source)
    {
        length++;
        current = d_paths[source * num_cities + current];

        // Safety check for cycles
        if (length >= max_path_length - 1)
        {
            d_path_lengths[source * num_cities + dest] = 0; // Mark as invalid path
            return;
        }
    }

    if (current != source)
    {
        // No valid path
        d_path_lengths[source * num_cities + dest] = 0;
        return;
    }

    // Valid path found, record length (including source)
    length++;
    d_path_lengths[source * num_cities + dest] = length;

    // Now reconstruct the path
    current = dest;
    int idx = length - 1;

    temp_path[idx--] = current;
    while (current != source)
    {
        current = d_paths[source * num_cities + current];
        temp_path[idx--] = current;
    }

    // Copy to output
    for (int i = 0; i < length; i++)
    {
        d_reconstructed_paths[(source * num_cities + dest) * max_path_length + i] = temp_path[i];
    }
}

// CUDA kernel for parallel shelter evaluation
__global__ void evaluateSheltersKernel(
    int *d_shelterCities,
    int *d_shelterCapacities,
    int *d_distances,
    float *d_scores,
    int sourceCity,
    int peopleToEvacuate,
    int num_shelters,
    int max_distance_elderly,
    bool forElderly,
    int num_cities)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_shelters)
        return;

    int shelterCity = d_shelterCities[tid];
    int capacity = d_shelterCapacities[tid];

    // Skip full shelters
    if (capacity <= 0)
    {
        d_scores[tid] = -1.0f;
        return;
    }

    int dist = d_distances[shelterCity];

    // Skip if distance exceeds elderly limit and we're checking for elderly
    if (forElderly && dist > max_distance_elderly)
    {
        d_scores[tid] = -1.0f;
        return;
    }

    // Skip if no path exists
    if (dist == INT_MAX)
    {
        d_scores[tid] = -1.0f;
        return;
    }

    // Calculate people we can save
    int peopleSaved = min(peopleToEvacuate, capacity);

    // Score formula prioritizes people saved but also considers time
    float score = peopleSaved / (1.0f + 0.1f * dist);

    d_scores[tid] = score;
}

// Convert adjacency list to adjacency matrix for CUDA
int *createGraphMatrix(const vector<vector<Road>> &adjacencyList, int num_cities)
{
    int *matrix = new int[num_cities * num_cities];

    // Initialize all to 0 (no connection)
    memset(matrix, 0, num_cities * num_cities * sizeof(int));

    // Fill in the matrix with road lengths
    for (int i = 0; i < num_cities; i++)
    {
        for (const Road &road : adjacencyList[i])
        {
            matrix[i * num_cities + road.to] = road.length;
        }
    }

    return matrix;
}

// Compute shortest paths using CUDA
void computeShortestPathsCuda(
    const vector<vector<Road>> &adjacencyList,
    vector<vector<int>> &distances,
    vector<vector<vector<int>>> &shortestPaths,
    int num_cities)
{

    // Choose approach based on number of cities
    bool smallDataset = (num_cities <= 1000);

    // Convert adjacency list to matrix
    int *graphMatrix = createGraphMatrix(adjacencyList, num_cities);

    // Flattened arrays for distances and paths
    int *flatDistances = new int[num_cities * num_cities];
    int *flatPaths = new int[num_cities * num_cities];

    // Initialize on host first
    for (int i = 0; i < num_cities; i++)
    {
        for (int j = 0; j < num_cities; j++)
        {
            if (i == j)
            {
                flatDistances[i * num_cities + j] = 0;
            }
            else
            {
                flatDistances[i * num_cities + j] = INT_MAX;
            }
            flatPaths[i * num_cities + j] = -1;
        }
    }

    // Device memory
    int *d_graph, *d_distances, *d_paths;

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_graph, num_cities * num_cities * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_distances, num_cities * num_cities * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_paths, num_cities * num_cities * sizeof(int)));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_graph, graphMatrix, num_cities * num_cities * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_distances, flatDistances, num_cities * num_cities * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_paths, flatPaths, num_cities * num_cities * sizeof(int), cudaMemcpyHostToDevice));

    if (smallDataset)
    {
        // For small datasets, run all sources in parallel
        parallelDijkstraKernel<<<num_cities, 1>>>(d_graph, d_distances, d_paths, num_cities);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    else
    {
        // For large datasets, process in batches
        const int BATCH_SIZE = 256;
        for (int start = 0; start < num_cities; start += BATCH_SIZE)
        {
            int batchSize = min(BATCH_SIZE, num_cities - start);
            parallelDijkstraKernel<<<batchSize, 1>>>(d_graph, d_distances, d_paths, num_cities);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }
    }

    // Copy results back
    CHECK_CUDA_ERROR(cudaMemcpy(flatDistances, d_distances, num_cities * num_cities * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(flatPaths, d_paths, num_cities * num_cities * sizeof(int), cudaMemcpyDeviceToHost));

    // Reconstruct paths
    if (smallDataset)
    {
        // We'll reconstruct all paths for small datasets
        int max_path_length = num_cities;
        int *flatReconstructedPaths = new int[num_cities * num_cities * max_path_length];
        int *pathLengths = new int[num_cities * num_cities];

        // Initialize path lengths
        memset(pathLengths, 0, num_cities * num_cities * sizeof(int));

        int *d_reconstructed_paths, *d_path_lengths;
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_reconstructed_paths, num_cities * num_cities * max_path_length * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_path_lengths, num_cities * num_cities * sizeof(int)));

        CHECK_CUDA_ERROR(cudaMemcpy(d_path_lengths, pathLengths, num_cities * num_cities * sizeof(int), cudaMemcpyHostToDevice));

        // Launch kernel for path reconstruction
        pathReconstructionKernel<<<num_cities, num_cities>>>(d_paths, d_reconstructed_paths, d_path_lengths, num_cities, max_path_length);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Copy results back
        CHECK_CUDA_ERROR(cudaMemcpy(pathLengths, d_path_lengths, num_cities * num_cities * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(flatReconstructedPaths, d_reconstructed_paths, num_cities * num_cities * max_path_length * sizeof(int), cudaMemcpyDeviceToHost));

        // Convert to output format
        for (int i = 0; i < num_cities; i++)
        {
            for (int j = 0; j < num_cities; j++)
            {
                int pathLength = pathLengths[i * num_cities + j];

                if (pathLength > 0)
                {
                    shortestPaths[i][j].resize(pathLength);
                    for (int k = 0; k < pathLength; k++)
                    {
                        shortestPaths[i][j][k] = flatReconstructedPaths[(i * num_cities + j) * max_path_length + k];
                    }
                }
                else
                {
                    shortestPaths[i][j].clear();
                }
            }
        }

        // Free additional memory
        CHECK_CUDA_ERROR(cudaFree(d_reconstructed_paths));
        CHECK_CUDA_ERROR(cudaFree(d_path_lengths));
        delete[] flatReconstructedPaths;
        delete[] pathLengths;
    }
    else
    {
        for (int i = 0; i < num_cities; i++)
        {
            for (int j = 0; j < num_cities; j++)
            {
                // Skip self-paths
                if (i == j)
                {
                    shortestPaths[i][j] = {i};
                    continue;
                }

                // Check if a path exists
                if (flatDistances[i * num_cities + j] == INT_MAX || flatPaths[i * num_cities + j] == -1)
                {
                    shortestPaths[i][j].clear();
                    continue;
                }

                // Reconstruct the path
                vector<int> path;
                int current = j;

                // Follow predecessors to reconstruct the path
                while (current != i)
                {
                    path.push_back(current);
                    current = flatPaths[i * num_cities + current];

                    // Safety check for cycles
                    if (path.size() >= num_cities)
                    {
                        path.clear();
                        break;
                    }
                }

                if (!path.empty())
                {
                    path.push_back(i);
                    reverse(path.begin(), path.end());
                    shortestPaths[i][j] = path;
                }
                else
                {
                    shortestPaths[i][j].clear();
                }
            }
        }
    }

    // Convert to output distances
    for (int i = 0; i < num_cities; i++)
    {
        for (int j = 0; j < num_cities; j++)
        {
            distances[i][j] = flatDistances[i * num_cities + j];
        }
    }

    // Free memory
    CHECK_CUDA_ERROR(cudaFree(d_graph));
    CHECK_CUDA_ERROR(cudaFree(d_distances));
    CHECK_CUDA_ERROR(cudaFree(d_paths));
    delete[] graphMatrix;
    delete[] flatDistances;
    delete[] flatPaths;
}

// Parallel shelter evaluation wrapper
vector<pair<int, double>> evaluateShelters(
    int sourceCity,
    int peopleToEvacuate,
    const vector<int> &shelterCities,
    const unordered_map<int, int> &shelterCapacity,
    const vector<vector<int>> &distances,
    int max_distance_elderly,
    bool forElderly = false)
{
    int num_shelters = shelterCities.size();
    if (num_shelters == 0 || peopleToEvacuate <= 0)
    {
        return {};
    }

    // Prepare data for GPU
    int *h_shelterCities = new int[num_shelters];
    int *h_shelterCapacities = new int[num_shelters];
    float *h_scores = new float[num_shelters];

    // Copy shelter cities and capacities
    for (int i = 0; i < num_shelters; i++)
    {
        h_shelterCities[i] = shelterCities[i];
        h_shelterCapacities[i] = shelterCapacity.at(shelterCities[i]);
    }

    // Get distances from source city to all other cities
    int num_cities = distances.size();
    int *h_distancesRow = new int[num_cities];
    for (int j = 0; j < num_cities; j++)
    {
        h_distancesRow[j] = distances[sourceCity][j];
    }

    // Allocate GPU memory
    int *d_shelterCities, *d_shelterCapacities, *d_distancesRow;
    float *d_scores;

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_shelterCities, num_shelters * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_shelterCapacities, num_shelters * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_distancesRow, num_cities * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_scores, num_shelters * sizeof(float)));

    // Copy data to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_shelterCities, h_shelterCities, num_shelters * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_shelterCapacities, h_shelterCapacities, num_shelters * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_distancesRow, h_distancesRow, num_cities * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (num_shelters + blockSize - 1) / blockSize;

    evaluateSheltersKernel<<<numBlocks, blockSize>>>(
        d_shelterCities, d_shelterCapacities, d_distancesRow, d_scores,
        sourceCity, peopleToEvacuate, num_shelters, max_distance_elderly, forElderly, num_cities);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Get results
    CHECK_CUDA_ERROR(cudaMemcpy(h_scores, d_scores, num_shelters * sizeof(float), cudaMemcpyDeviceToHost));

    // Create result vector
    vector<pair<int, double>> shelterScores;
    for (int i = 0; i < num_shelters; i++)
    {
        if (h_scores[i] > 0)
        {
            shelterScores.push_back({shelterCities[i], h_scores[i]});
        }
    }

    // Sort by score (highest first)
    sort(shelterScores.begin(), shelterScores.end(),
         [](const pair<int, double> &a, const pair<int, double> &b)
         {
             return a.second > b.second;
         });

    // Free memory
    delete[] h_shelterCities;
    delete[] h_shelterCapacities;
    delete[] h_scores;
    delete[] h_distancesRow;

    CHECK_CUDA_ERROR(cudaFree(d_shelterCities));
    CHECK_CUDA_ERROR(cudaFree(d_shelterCapacities));
    CHECK_CUDA_ERROR(cudaFree(d_distancesRow));
    CHECK_CUDA_ERROR(cudaFree(d_scores));

    return shelterScores;
}

// Calculate traversal time
double calculateTraversalTime(int numPeople, int roadCapacity, int roadLength)
{
    // Speed is 5 km/h
    double timePerBatch = static_cast<double>(roadLength) / 5.0; // Time in hours
    int numBatches = ceil(static_cast<double>(numPeople) / roadCapacity);
    return timePerBatch * numBatches;
}

// Generate evacuation paths
void generateEvacuationPaths(
    int num_cities,
    const vector<vector<Road>> &adjacencyList,
    const vector<int> &populatedCities,
    const unordered_map<int, pair<int, int>> &populatedCityInfo,
    const vector<int> &shelterCities,
    unordered_map<int, int> &shelterCapacity,
    const vector<vector<int>> &distances,
    const vector<vector<vector<int>>> &shortestPaths,
    int max_distance_elderly,
    long long *path_size,
    long long **paths,
    long long *num_drops,
    long long ***drops,
    int num_populated_cities)
{

    // Set timeout for large datasets
    auto startTime = chrono::high_resolution_clock::now();
    const int TIMEOUT_SECONDS = 590; // 9 minutes 50 seconds timeout

    // For each populated city
    for (int i = 0; i < num_populated_cities; i++)
    {
        // Check for timeout
        auto currentTime = chrono::high_resolution_clock::now();
        auto elapsedSeconds = chrono::duration_cast<chrono::seconds>(currentTime - startTime).count();

        if (elapsedSeconds > TIMEOUT_SECONDS)
        {
            // For remaining cities, use a simplified evacuation strategy
            for (int j = i; j < num_populated_cities; j++)
            {
                int sourceCity = populatedCities[j];
                int prime_age = populatedCityInfo.at(sourceCity).first;
                int elderly = populatedCityInfo.at(sourceCity).second;

                // Simplified evacuation: find closest shelter with capacity
                vector<pair<int, double>> shelters = evaluateShelters(
                    sourceCity, prime_age + elderly, shelterCities, shelterCapacity,
                    distances, max_distance_elderly, false);

                vector<int> evacPath;
                vector<vector<long long>> evacDrops;

                // Start with source city
                evacPath.push_back(sourceCity);

                if (shelters.empty())
                {
                    // No available shelter, drop everyone at source
                    evacDrops.push_back({(long long)sourceCity, (long long)prime_age, (long long)elderly});
                }
                else
                {
                    int targetShelter = shelters[0].first;

                    // Check if elderly can reach the shelter
                    bool elderlyCanReach = (distances[sourceCity][targetShelter] <= max_distance_elderly);

                    // If elderly can't reach, drop them at source
                    if (!elderlyCanReach && elderly > 0)
                    {
                        evacDrops.push_back({(long long)sourceCity, 0, (long long)elderly});
                        elderly = 0;
                    }

                    // Add path to shelter
                    const vector<int> &pathToShelter = shortestPaths[sourceCity][targetShelter];
                    if (!pathToShelter.empty())
                    {
                        for (int k = 1; k < pathToShelter.size(); k++)
                        {
                            evacPath.push_back(pathToShelter[k]);
                        }

                        // Drop remaining people at shelter
                        int capacity = shelterCapacity[targetShelter];
                        int elderlyToDrop = min(elderly, capacity);
                        capacity -= elderlyToDrop;
                        elderly -= elderlyToDrop;

                        int primeToDrop = min(prime_age, capacity);
                        capacity -= primeToDrop;
                        prime_age -= primeToDrop;

                        shelterCapacity[targetShelter] -= (elderlyToDrop + primeToDrop);

                        if (elderlyToDrop > 0 || primeToDrop > 0)
                        {
                            evacDrops.push_back({(long long)targetShelter, (long long)primeToDrop, (long long)elderlyToDrop});
                        }

                        // If anyone remains, drop at the shelter city (they won't be saved)
                        if (prime_age > 0 || elderly > 0)
                        {
                            evacDrops.push_back({(long long)targetShelter, (long long)prime_age, (long long)elderly});
                        }
                    }
                    else
                    {
                        // No path to shelter, drop everyone at source
                        evacDrops.push_back({(long long)sourceCity, (long long)prime_age, (long long)elderly});
                    }
                }

                // Set path and drops
                path_size[j] = evacPath.size();
                paths[j] = new long long[evacPath.size()];
                for (int k = 0; k < evacPath.size(); k++)
                {
                    paths[j][k] = evacPath[k];
                }

                num_drops[j] = evacDrops.size();
                drops[j] = new long long *[evacDrops.size()];
                for (int k = 0; k < evacDrops.size(); k++)
                {
                    drops[j][k] = new long long[3];
                    drops[j][k][0] = evacDrops[k][0]; // City
                    drops[j][k][1] = evacDrops[k][1]; // Prime age
                    drops[j][k][2] = evacDrops[k][2]; // Elderly
                }
            }

            // Skip the normal processing
            return;
        }

        int sourceCity = populatedCities[i];
        int prime_age = populatedCityInfo.at(sourceCity).first;
        int elderly = populatedCityInfo.at(sourceCity).second;

        vector<int> evacPath;
        vector<vector<long long>> evacDrops;

        // Start with the source city
        evacPath.push_back(sourceCity);
        int currentCity = sourceCity;
        int distanceTraveled = 0;

        // Check if source city is a shelter
        if (shelterCapacity.find(sourceCity) != shelterCapacity.end() && shelterCapacity[sourceCity] > 0)
        {
            int capacity = shelterCapacity[sourceCity];

            // Drop elderly first
            int elderlyToDrop = min(elderly, capacity);
            capacity -= elderlyToDrop;
            elderly -= elderlyToDrop;

            // Then drop prime age
            int primeToDrop = min(prime_age, capacity);
            capacity -= primeToDrop;
            prime_age -= primeToDrop;

            // Update shelter capacity
            shelterCapacity[sourceCity] -= (elderlyToDrop + primeToDrop);

            // Record drop
            if (elderlyToDrop > 0 || primeToDrop > 0)
            {
                evacDrops.push_back({(long long)sourceCity, (long long)primeToDrop, (long long)elderlyToDrop});
            }
        }

        // Add iteration counter to prevent infinite loops
        int maxIterations = min(100, num_cities * 2); // Adjust based on graph size
        int iterationCount = 0;

        // Continue evacuation until all people are dropped
        while ((prime_age > 0 || elderly > 0) && iterationCount++ < maxIterations)
        {
            // First handle elderly (they have distance restrictions)
            int nextShelterForElderly = -1;
            auto elderlyShelters = evaluateShelters(
                currentCity, elderly, shelterCities, shelterCapacity,
                distances, max_distance_elderly, true);

            if (elderly > 0 && !elderlyShelters.empty())
            {
                nextShelterForElderly = elderlyShelters[0].first;
            }

            // If no shelter for elderly within range, drop them at current city
            if (elderly > 0 && nextShelterForElderly == -1)
            {
                evacDrops.push_back({(long long)currentCity, 0, (long long)elderly});
                elderly = 0;
            }

            // Now handle prime-age people
            int nextShelterForPrime = -1;
            auto primeShelters = evaluateShelters(
                currentCity, prime_age, shelterCities, shelterCapacity,
                distances, INT_MAX, false);

            if (prime_age > 0 && !primeShelters.empty())
            {
                nextShelterForPrime = primeShelters[0].first;
            }

            // If no shelter available for anyone, drop remaining at current city
            if (nextShelterForElderly == -1 && nextShelterForPrime == -1)
            {
                if (prime_age > 0)
                {
                    evacDrops.push_back({(long long)currentCity, (long long)prime_age, 0});
                    prime_age = 0;
                }
                break;
            }

            // Decide which shelter to go to (prioritize elderly if they have a shelter)
            int targetShelter = (nextShelterForElderly != -1) ? nextShelterForElderly : nextShelterForPrime;

            // Get path to the shelter
            const vector<int> &pathToShelter = shortestPaths[currentCity][targetShelter];

            // Skip first city if it's the current city
            int startIdx = (pathToShelter.size() > 0 && pathToShelter[0] == currentCity) ? 1 : 0;

            // Check for empty paths
            if (pathToShelter.empty() || startIdx >= pathToShelter.size())
            {
                // No path to shelter found, drop everyone at current city
                if (elderly > 0)
                {
                    evacDrops.push_back({(long long)currentCity, 0, (long long)elderly});
                    elderly = 0;
                }
                if (prime_age > 0)
                {
                    evacDrops.push_back({(long long)currentCity, (long long)prime_age, 0});
                    prime_age = 0;
                }
                break;
            }

            // Add maximum path length safeguard
            const int MAX_PATH_LENGTH = min(1000, num_cities * 2); // Adjust based on graph size
            bool pathTooLong = false;

            // Travel along the path to the shelter
            for (int j = startIdx; j < pathToShelter.size(); j++)
            {
                // Check if the path is getting too long
                if (evacPath.size() >= MAX_PATH_LENGTH)
                {
                    pathTooLong = true;
                    break;
                }

                int nextCity = pathToShelter[j];

                // Calculate distance to next city
                int addedDistance = 0;
                for (const Road &road : adjacencyList[currentCity])
                {
                    if (road.to == nextCity)
                    {
                        addedDistance = road.length;
                        break;
                    }
                }

                // Check if elderly can reach the next city
                bool elderlyCanReach = (distanceTraveled + addedDistance) <= max_distance_elderly;

                // If elderly can't reach next city, drop them at current city
                if (elderly > 0 && !elderlyCanReach)
                {
                    evacDrops.push_back({(long long)currentCity, 0, (long long)elderly});
                    elderly = 0;
                }

                // Add next city to path
                evacPath.push_back(nextCity);

                // Update current city and distance
                currentCity = nextCity;
                distanceTraveled += addedDistance;

                // If current city is a shelter, try to drop people
                if (shelterCapacity.find(currentCity) != shelterCapacity.end() && shelterCapacity[currentCity] > 0)
                {
                    int capacity = shelterCapacity[currentCity];

                    // Drop elderly first (if they can reach)
                    int elderlyToDrop = min(elderly, capacity);
                    capacity -= elderlyToDrop;
                    elderly -= elderlyToDrop;

                    // Then drop prime age
                    int primeToDrop = min(prime_age, capacity);
                    capacity -= primeToDrop;
                    prime_age -= primeToDrop;

                    // Update shelter capacity
                    shelterCapacity[currentCity] -= (elderlyToDrop + primeToDrop);

                    // Record drop
                    if (elderlyToDrop > 0 || primeToDrop > 0)
                    {
                        evacDrops.push_back({(long long)currentCity, (long long)primeToDrop, (long long)elderlyToDrop});
                    }

                    // If all evacuees are dropped, break
                    if (prime_age == 0 && elderly == 0)
                    {
                        break;
                    }
                }
            }

            // If path was too long, drop remaining population at current city
            if (pathTooLong)
            {
                if (elderly > 0)
                {
                    evacDrops.push_back({(long long)currentCity, 0, (long long)elderly});
                    elderly = 0;
                }
                if (prime_age > 0)
                {
                    evacDrops.push_back({(long long)currentCity, (long long)prime_age, 0});
                    prime_age = 0;
                }
                break;
            }
        }

        // Handle case where we hit the maximum iterations
        if (iterationCount >= maxIterations)
        {
            // Drop any remaining population at current city
            if (elderly > 0)
            {
                evacDrops.push_back({(long long)currentCity, 0, (long long)elderly});
            }
            if (prime_age > 0)
            {
                evacDrops.push_back({(long long)currentCity, (long long)prime_age, 0});
            }
        }

        // Set the path size
        path_size[i] = evacPath.size();

        // Allocate and copy the path
        paths[i] = new long long[evacPath.size()];
        for (int j = 0; j < evacPath.size(); j++)
        {
            paths[i][j] = evacPath[j];
        }

        // Set the number of drops
        num_drops[i] = evacDrops.size();

        // Allocate and copy the drops
        drops[i] = new long long *[evacDrops.size()];
        for (int j = 0; j < evacDrops.size(); j++)
        {
            drops[i][j] = new long long[3];
            drops[i][j][0] = evacDrops[j][0]; // City
            drops[i][j][1] = evacDrops[j][1]; // Prime age
            drops[i][j][2] = evacDrops[j][2]; // Elderly
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return 1;
    }

    // Start timing
    auto startTime = chrono::high_resolution_clock::now();

    // Read input file
    ifstream infile(argv[1]);
    if (!infile)
    {
        cerr << "Error: Cannot open file " << argv[1] << "\n";
        return 1;
    }

    long long num_cities;
    infile >> num_cities;
    cout << "Number of cities: " << num_cities << endl;

    long long num_roads;
    infile >> num_roads;
    cout << "Number of roads: " << num_roads << endl;

    // Store roads as a flat array
    int *roads = new int[num_roads * 4];
    for (int i = 0; i < num_roads; i++)
    {
        infile >> roads[4 * i] >> roads[4 * i + 1] >> roads[4 * i + 2] >> roads[4 * i + 3];
    }

    int num_shelters;
    infile >> num_shelters;
    cout << "Number of shelters: " << num_shelters << endl;

    // Store shelters
    long long *shelter_city = new long long[num_shelters];
    long long *shelter_capacity = new long long[num_shelters];
    for (int i = 0; i < num_shelters; i++)
    {
        infile >> shelter_city[i] >> shelter_capacity[i];
    }

    int num_populated_cities;
    infile >> num_populated_cities;
    cout << "Number of populated cities: " << num_populated_cities << endl;

    // Store populated cities
    long long *city = new long long[num_populated_cities];
    long long *pop = new long long[num_populated_cities * 2]; // [prime-age, elderly] pairs
    for (long long i = 0; i < num_populated_cities; i++)
    {
        infile >> city[i] >> pop[2 * i] >> pop[2 * i + 1];
    }

    int max_distance_elderly;
    infile >> max_distance_elderly;
    cout << "Max distance for elderly: " << max_distance_elderly << endl;
    infile.close();

    // Build adjacency list from roads array
    vector<vector<Road>> adjacencyList(num_cities);
    for (int i = 0; i < num_roads; i++)
    {
        int u = roads[4 * i];
        int v = roads[4 * i + 1];
        int length = roads[4 * i + 2];
        int capacity = roads[4 * i + 3];

        adjacencyList[u].push_back({v, length, capacity});
        adjacencyList[v].push_back({u, length, capacity});
    }

    // Create shelter capacity map and list
    unordered_map<int, int> shelterCapacity;
    vector<int> shelterCities;
    for (int i = 0; i < num_shelters; i++)
    {
        int city_id = shelter_city[i];
        int capacity = shelter_capacity[i];

        shelterCapacity[city_id] = capacity;
        shelterCities.push_back(city_id);
    }

    // Create populated city info map and list
    unordered_map<int, pair<int, int>> populatedCityInfo;
    vector<int> populatedCities;
    for (int i = 0; i < num_populated_cities; i++)
    {
        int city_id = city[i];
        int prime_age = pop[2 * i];
        int elderly = pop[2 * i + 1];

        populatedCityInfo[city_id] = {prime_age, elderly};
        populatedCities.push_back(city_id);
    }

    // Initialize distances and paths
    vector<vector<int>> distances(num_cities, vector<int>(num_cities, INT_MAX));
    vector<vector<vector<int>>> shortestPaths(num_cities, vector<vector<int>>(num_cities));

    // Compute shortest paths using CUDA
    computeShortestPathsCuda(adjacencyList, distances, shortestPaths, num_cities);

    // Set answer variables
    long long *path_size = new long long[num_populated_cities];
    long long **paths = new long long *[num_populated_cities];
    long long *num_drops = new long long[num_populated_cities];
    long long ***drops = new long long **[num_populated_cities];

    // Generate evacuation paths
    generateEvacuationPaths(
        num_cities, adjacencyList, populatedCities, populatedCityInfo,
        shelterCities, shelterCapacity, distances, shortestPaths,
        max_distance_elderly, path_size, paths, num_drops, drops, num_populated_cities);

    // Write output
    ofstream outfile(argv[2]);
    if (!outfile)
    {
        cerr << "Error: Cannot open file " << argv[2] << "\n";
        return 1;
    }

    for (long long i = 0; i < num_populated_cities; i++)
    {
        long long currentPathSize = path_size[i];
        for (long long j = 0; j < currentPathSize; j++)
        {
            outfile << paths[i][j] << " ";
        }
        outfile << "\n";
    }

    for (long long i = 0; i < num_populated_cities; i++)
    {
        long long currentDropSize = num_drops[i];
        for (long long j = 0; j < currentDropSize; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                outfile << drops[i][j][k] << " ";
            }
        }
        outfile << "\n";
    }

    // Free allocated memory
    delete[] roads;
    delete[] shelter_city;
    delete[] shelter_capacity;
    delete[] city;
    delete[] pop;

    for (int i = 0; i < num_populated_cities; i++)
    {
        delete[] paths[i];
        for (int j = 0; j < num_drops[i]; j++)
        {
            delete[] drops[i][j];
        }
        delete[] drops[i];
    }

    delete[] path_size;
    delete[] paths;
    delete[] num_drops;
    delete[] drops;

    // Print execution time
    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    cout << "Total execution time: " << duration / 1000.0 << " seconds" << endl;

    return 0;
}