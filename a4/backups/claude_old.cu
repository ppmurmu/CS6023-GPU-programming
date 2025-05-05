#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <algorithm>
#include <climits>
#include <cmath>
#include <unordered_map>
#include <utility>
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
// Process cities in batches to avoid memory issues
__global__ void batchDijkstraKernel(int *d_graph, int *d_distances, int *d_paths, int num_cities, int start_source, int batch_size)
{
    int local_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_id >= batch_size)
        return;

    int source = start_source + local_id;
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

// First pass to determine path lengths
__global__ void computePathLengthsKernel(int *d_paths, int *d_path_lengths, int num_cities, int start_source, int batch_size)
{
    int local_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_id >= batch_size * num_cities)
        return;

    int batch_offset = local_id / num_cities;
    int source = start_source + batch_offset;
    int dest = local_id % num_cities;

    if (source >= num_cities || dest >= num_cities)
        return;

    // Skip self-paths (handled separately)
    if (source == dest)
    {
        d_path_lengths[source * num_cities + dest] = 1;
        return;
    }

    // Check if path exists
    if (d_paths[source * num_cities + dest] == -1)
    {
        d_path_lengths[source * num_cities + dest] = 0;
        return;
    }

    // Count the path length by following predecessors
    int length = 0;
    int current = dest;

    while (current != -1 && current != source)
    {
        length++;
        current = d_paths[source * num_cities + current];

        // Safety check for cycles (shouldn't happen with Dijkstra, but just in case)
        if (length > num_cities)
        {
            d_path_lengths[source * num_cities + dest] = 0; // Mark as invalid path
            return;
        }
    }

    if (current == source)
    {
        d_path_lengths[source * num_cities + dest] = length + 1; // +1 for source
    }
    else
    {
        d_path_lengths[source * num_cities + dest] = 0; // No valid path
    }
}

// Convert adjacency list to compressed sparse row (CSR) format for CUDA
void convertToCSR(const vector<vector<Road>> &adjacencyList, vector<int> &rowOffsets, vector<int> &colIndices, vector<int> &edgeWeights)
{
    int numVertices = adjacencyList.size();
    rowOffsets.resize(numVertices + 1);

    // Count number of edges and set row offsets
    rowOffsets[0] = 0;
    for (int i = 0; i < numVertices; i++)
    {
        rowOffsets[i + 1] = rowOffsets[i] + adjacencyList[i].size();
    }

    // Reserve space for column indices and weights
    int numEdges = rowOffsets[numVertices];
    colIndices.resize(numEdges);
    edgeWeights.resize(numEdges);

    // Fill column indices and weights
    for (int i = 0; i < numVertices; i++)
    {
        int offset = rowOffsets[i];
        for (size_t j = 0; j < adjacencyList[i].size(); j++)
        {
            colIndices[offset + j] = adjacencyList[i][j].to;
            edgeWeights[offset + j] = adjacencyList[i][j].length;
        }
    }
}

// Convert CSR to adjacency matrix for CUDA processing (but only for populated cities and shelters)
int *createGraphMatrix(const vector<vector<Road>> &adjacencyList, int num_cities)
{
    // For large graphs, we'll create a matrix for all cities
    // but we could optimize this to only include relevant cities
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

// Compute shortest paths using CUDA - with batched processing to save memory
void computeShortestPathsCuda(
    const vector<vector<Road>> &adjacencyList,
    vector<vector<int>> &distances,
    vector<vector<vector<int>>> &shortestPaths,
    int num_cities,
    const vector<int> &populatedCities,
    const vector<int> &shelterCities)
{
    cout << "Converting adjacency list to matrix..." << endl;
    // This goes at the start of the computeShortestPathsCuda function
    // Create graph matrix only for relevant cities to save memory
    int *graphMatrix = createGraphMatrix(adjacencyList, num_cities);

    // Create flattened arrays for distances and paths
    int *flatDistances = new int[num_cities * num_cities];
    int *flatPaths = new int[num_cities * num_cities];
    int *pathLengths = new int[num_cities * num_cities];

    // Initialize distances and paths on host
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
            pathLengths[i * num_cities + j] = 0;
        }
    }

    // Device memory
    int *d_graph, *d_distances, *d_paths, *d_path_lengths;

    // Allocate device memory
    cudaMalloc((void **)&d_graph, num_cities * num_cities * sizeof(int));
    cudaMalloc((void **)&d_distances, num_cities * num_cities * sizeof(int));
    cudaMalloc((void **)&d_paths, num_cities * num_cities * sizeof(int));
    cudaMalloc((void **)&d_path_lengths, num_cities * num_cities * sizeof(int));

    // Copy graph and initial data to device
    cudaMemcpy(d_graph, graphMatrix, num_cities * num_cities * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_distances, flatDistances, num_cities * num_cities * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_paths, flatPaths, num_cities * num_cities * sizeof(int), cudaMemcpyHostToDevice);

    // Process in batches to avoid memory issues
    const int BATCH_SIZE = 64; // Adjust based on available GPU memory

    for (int start_source = 0; start_source < num_cities; start_source += BATCH_SIZE)
    {
        int current_batch_size = min(BATCH_SIZE, num_cities - start_source);

        // Launch Dijkstra kernel for a batch of sources
        int threadsPerBlock = 256;
        int numBlocks = (current_batch_size + threadsPerBlock - 1) / threadsPerBlock;

        batchDijkstraKernel<<<numBlocks, threadsPerBlock>>>(d_graph, d_distances, d_paths, num_cities, start_source, current_batch_size);
        cudaDeviceSynchronize();

        // Compute path lengths for the batch
        int totalThreads = current_batch_size * num_cities;
        numBlocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

        cout << "Processing batch starting from " << start_source << " (batch size: " << current_batch_size << ")..." << endl;
        // This goes inside the for loop before launching batchDijkstraKernel
        computePathLengthsKernel<<<numBlocks, threadsPerBlock>>>(d_paths, d_path_lengths, num_cities, start_source, current_batch_size);
        cudaDeviceSynchronize();
    }

    // Copy results back to host
    cudaMemcpy(flatDistances, d_distances, num_cities * num_cities * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(flatPaths, d_paths, num_cities * num_cities * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(pathLengths, d_path_lengths, num_cities * num_cities * sizeof(int), cudaMemcpyDeviceToHost);

    // Convert flat distances to 2D vector
    for (int i = 0; i < num_cities; i++)
    {
        for (int j = 0; j < num_cities; j++)
        {
            distances[i][j] = flatDistances[i * num_cities + j];
        }
    }
    cout << "Shortest paths computation complete." << endl;
    // This goes right before returning from computeShortestPathsCuda
    // Only reconstruct paths that are needed (between populated cities and shelters)
    // to save memory
    for (int source : populatedCities)
    {
        vector<int> destinations = shelterCities;                                                // Start with shelters
        destinations.insert(destinations.end(), populatedCities.begin(), populatedCities.end()); // Add populated cities

        for (int dest : destinations)
        {
            if (source == dest)
            {
                shortestPaths[source][dest] = {source}; // Path to self
                continue;
            }

            int pathLength = pathLengths[source * num_cities + dest];
            if (pathLength == 0)
            {
                // No path exists
                shortestPaths[source][dest].clear();
                continue;
            }

            // Reconstruct path by following predecessors
            vector<int> path(pathLength);
            int current = dest;
            int idx = pathLength - 1;

            while (current != source)
            {
                path[idx--] = current;
                current = flatPaths[source * num_cities + current];
            }
            path[0] = source;

            shortestPaths[source][dest] = path;
        }
    }

    // Free device memory
    cudaFree(d_graph);
    cudaFree(d_distances);
    cudaFree(d_paths);
    cudaFree(d_path_lengths);

    // Free host memory
    delete[] graphMatrix;
    delete[] flatDistances;
    delete[] flatPaths;
    delete[] pathLengths;
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

    int dist = d_distances[sourceCity * num_cities + shelterCity];

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

    // Flatten distance matrix for just the required row
    int num_cities = distances.size();
    int *h_distancesRow = new int[num_cities];

    for (int j = 0; j < num_cities; j++)
    {
        h_distancesRow[j] = distances[sourceCity][j];
    }

    // Allocate GPU memory
    int *d_shelterCities, *d_shelterCapacities, *d_distancesRow;
    float *d_scores;

    cudaMalloc((void **)&d_shelterCities, num_shelters * sizeof(int));
    cudaMalloc((void **)&d_shelterCapacities, num_shelters * sizeof(int));
    cudaMalloc((void **)&d_distancesRow, num_cities * sizeof(int));
    cudaMalloc((void **)&d_scores, num_shelters * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_shelterCities, h_shelterCities, num_shelters * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shelterCapacities, h_shelterCapacities, num_shelters * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_distancesRow, h_distancesRow, num_cities * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (num_shelters + blockSize - 1) / blockSize;

    // Modified kernel call that only needs the single row of distances
    evaluateSheltersKernel<<<numBlocks, blockSize>>>(
        d_shelterCities, d_shelterCapacities, d_distancesRow, d_scores,
        sourceCity, peopleToEvacuate, num_shelters, max_distance_elderly, forElderly, num_cities);

    // Get results
    cudaMemcpy(h_scores, d_scores, num_shelters * sizeof(float), cudaMemcpyDeviceToHost);

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

    cudaFree(d_shelterCities);
    cudaFree(d_shelterCapacities);
    cudaFree(d_distancesRow);
    cudaFree(d_scores);

    return shelterScores;
}

// Road traversal time calculation
double calculateTraversalTime(int numPeople, int roadCapacity, int roadLength)
{
    // Speed is 5 km/h
    double timePerBatch = static_cast<double>(roadLength) / 5.0; // Time in hours
    int numBatches = ceil(static_cast<double>(numPeople) / roadCapacity);
    return timePerBatch * numBatches;
}

// Generate evacuation paths for all populated cities
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
    // Add timeout tracking

    // For each populated city
    for (int i = 0; i < num_populated_cities; i++)
    {
        // Check for timeout

                int sourceCity = populatedCities[i];
        int prime_age = populatedCityInfo.at(sourceCity).first;
        int elderly = populatedCityInfo.at(sourceCity).second;

        vector<int> evacPath;
        vector<vector<long long>> evacDrops;

        // Start with the source city
        evacPath.push_back(sourceCity);
        int currentCity = sourceCity;
        int distanceTraveled = 0;

        cout << "Processing evacuation for city " << sourceCity << " (" << i + 1 << "/" << num_populated_cities << ")..." << endl;

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
        int maxIterations = 100; // Adjust based on expected complexity
        int iterationCount = 0;

        // Continue evacuation until all people are dropped
        while ((prime_age > 0 || elderly > 0) && iterationCount++ < maxIterations)
        {
            // Debug output for shelter evaluation
            cout << "  Evaluating shelters from city " << currentCity << " (prime: " << prime_age << ", elderly: " << elderly << ")" << endl;

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
                distances, max_distance_elderly, false);

            // Debug output for shelter evaluation results
            cout << "  Found " << elderlyShelters.size() << " shelters for elderly, "
                 << primeShelters.size() << " shelters for prime-age" << endl;

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
                cout << "  WARNING: Empty path to shelter " << targetShelter << ". Dropping population at current city." << endl;
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
            const int MAX_PATH_LENGTH = 1000; // Adjust as needed
            bool pathTooLong = false;

            // Travel along the path to the shelter
            for (int j = startIdx; j < pathToShelter.size(); j++)
            {
                // Check if the path is getting too long
                if (evacPath.size() >= MAX_PATH_LENGTH)
                {
                    cout << "  WARNING: Path length exceeds maximum allowed. Dropping remaining population." << endl;
                    pathTooLong = true;
                    break;
                }

                int nextCity = pathToShelter[j];

                // Debug output for city traversal
                cout << "  Moving to city " << nextCity << " (distance so far: " << distanceTraveled << ")" << endl;

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

            // If path was too long, drop remaining population
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
            cout << "WARNING: Maximum iterations reached for city " << sourceCity << ". Dropping remaining population." << endl;
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

    //--------------input--------------------------------
    ifstream infile(argv[1]); // Read input file from command-line argument
    if (!infile)
    {
        cerr << "Error: Cannot open file " << argv[1] << "\n";
        return 1;
    }

    long long num_cities;
    infile >> num_cities;
    cout << "num cities = " << num_cities << endl;

    long long num_roads;
    infile >> num_roads;
    cout << "num roads = " << num_roads << endl;

    // Store roads as a flat array: [u1, v1, length1, capacity1, u2, v2, length2, capacity2, ...]
    int *roads = new int[num_roads * 4];

    for (int i = 0; i < num_roads; i++)
    {
        infile >> roads[4 * i] >> roads[4 * i + 1] >> roads[4 * i + 2] >> roads[4 * i + 3];
    }

    int num_shelters;
    infile >> num_shelters;
    cout << "num shelters = " << num_shelters << endl;

    // Store shelters separately
    long long *shelter_city = new long long[num_shelters];
    long long *shelter_capacity = new long long[num_shelters];

    for (int i = 0; i < num_shelters; i++)
    {
        infile >> shelter_city[i] >> shelter_capacity[i];
    }

    int num_populated_cities;
    infile >> num_populated_cities;
    cout << "num pop cities = " << num_populated_cities << endl;

    // Store populated cities separately
    long long *city = new long long[num_populated_cities];
    long long *pop = new long long[num_populated_cities * 2]; // Flattened [prime-age, elderly] pairs

    for (long long i = 0; i < num_populated_cities; i++)
    {
        infile >> city[i] >> pop[2 * i] >> pop[2 * i + 1];
    }

    int max_distance_elderly;
    infile >> max_distance_elderly;
    cout << "max dist elderly= " << max_distance_elderly << endl;

    infile.close();

    //-------------------------end of input----------

    cout << "Input loaded. Starting preprocessing..." << endl;
    // This goes right after: infile.close();

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
    // Only allocate for the needed cities to save memory
    vector<vector<int>> distances(num_cities, vector<int>(num_cities, INT_MAX));
    vector<vector<vector<int>>> shortestPaths(num_cities, vector<vector<int>>(num_cities));

    // Compute shortest paths using CUDA with memory optimizations
    cout << "Starting shortest path computation for " << num_cities << " cities and " << num_roads << " roads..." << endl;
    // This goes right before: computeShortestPathsCuda(...) call
    computeShortestPathsCuda(adjacencyList, distances, shortestPaths, num_cities, populatedCities, shelterCities);

    // set your answer to these variables
    long long *path_size = new long long[num_populated_cities];
    long long **paths = new long long *[num_populated_cities];
    long long *num_drops = new long long[num_populated_cities];
    long long ***drops = new long long **[num_populated_cities];

    // Generate evacuation paths
    cout << "Generating evacuation paths for " << num_populated_cities << " populated cities..." << endl;
    // This goes before calling generateEvacuationPaths
    generateEvacuationPaths(
        num_cities, adjacencyList, populatedCities, populatedCityInfo,
        shelterCities, shelterCapacity, distances, shortestPaths,
        max_distance_elderly, path_size, paths, num_drops, drops, num_populated_cities);

    //------------output-----------------

    ofstream outfile(argv[2]); // Output file from command-line argument
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

    return 0;
}