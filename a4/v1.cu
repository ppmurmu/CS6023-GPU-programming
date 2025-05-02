// v3_descriptive.cu
/*
Output Format:
  For each populated city (in input order):
    - Line with the evacuation path: space-separated city IDs.
  For each populated city:
    - Line listing drop events: shelterCityID primeCount elderlyCount
  At the end:
    Saved_primes X
    Saved_elderly Y

Overall Plan:
 1. Read input: numCities, numRoads, roads, shelters, populated cities, maxElderlyDistance.
 2. Build adjacency list and run multi-source Dijkstra from all shelters to compute:
    - shortestDistanceKm to each city,
    - parentEdge for path reconstruction,
    - assignedShelterIndex per city.
 3. For each populated city, reconstruct its cityPath and edgePath.
 4. Flatten edgePath arrays and copy all data to GPU: edgePaths, edgeLengths, edgeCapacities, population counts.
 5. GPU kernel simulates capacity batching + queue contention using atomicMax, computes arrivalTime and preliminary saved counts.
 6. Copy back saved counts and arrivalTimes, enforce shelter capacities in arrival-time order, compute totalSaved.
 7. Write per-city paths, drop events, and saved summary.
*/

#include <bits/stdc++.h>
#include <cuda_runtime.h>
using namespace std;
using ll = long long;

// GPU kernel: simulate evacuation across each city’s edge path
__global__ void evacuationKernel(
    int numPopulatedCities,
    const int *devicePathOffsets,    // size = numPopulatedCities+1
    const int *deviceEdgePaths,      // flattened edge IDs
    const int *deviceEdgeLengths,    // km per edge
    const int *deviceEdgeCapacities, // capacity per edge
    const ll *devicePrimeCount,      // prime-age count per city
    const ll *deviceElderlyCount,    // elderly count per city
    ll maxElderlyDistanceKm,         // km
    ll *deviceEdgeReadyTime,         // minute-ready per edge
    ll *deviceArrivalTime,           // arrival time per city (minutes)
    ll *deviceSavedPrime,            // saved prime-age per city
    ll *deviceSavedElderly           // saved elderly per city
)
{
    int cityIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (cityIndex >= numPopulatedCities)
        return;

    ll totalGroupSize = devicePrimeCount[cityIndex] + deviceElderlyCount[cityIndex];
    ll currentTime = 0;
    int startOffset = devicePathOffsets[cityIndex];
    int endOffset = devicePathOffsets[cityIndex + 1];

    // Traverse each road in the path
    for (int offset = startOffset; offset < endOffset; ++offset)
    {
        int edgeId = deviceEdgePaths[offset];
        int lengthKm = deviceEdgeLengths[edgeId];
        int capacity = deviceEdgeCapacities[edgeId];
        int numBatches = (totalGroupSize + capacity - 1) / capacity;
        // time to traverse this edge in minutes: (lengthKm/5 km/h)*60
        ll travelTime = (ll)lengthKm * 12LL * numBatches;

        // contention: wait for previous groups using atomicMax
        ll prevReadyTime = atomicMax(&deviceEdgeReadyTime[edgeId], currentTime);
        ll departureTime = max(currentTime, prevReadyTime);
        ll finishTime = departureTime + travelTime;
        // update edge ready time for next groups
        atomicMax(&deviceEdgeReadyTime[edgeId], finishTime);
        currentTime = finishTime;
    }

    deviceArrivalTime[cityIndex] = currentTime;
    deviceSavedPrime[cityIndex] = devicePrimeCount[cityIndex];
    // elderly saved only if within max distance limit (converted to minutes)
    ll elderlyMaxTime = maxElderlyDistanceKm * 12LL;
    deviceSavedElderly[cityIndex] = (currentTime <= elderlyMaxTime) ? deviceElderlyCount[cityIndex] : 0LL;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return 1;
    }

    // 1) Read input
    ifstream inputFile(argv[1]);
    int numCities, numRoads;
    inputFile >> numCities >> numRoads;
    struct Road
    {
        int u, v, lengthKm, capacity;
    };
    vector<Road> roads(numRoads);
    vector<vector<pair<int, int>>> adjacency(numCities);
    for (int i = 0; i < numRoads; ++i)
    {
        inputFile >> roads[i].u >> roads[i].v >> roads[i].lengthKm >> roads[i].capacity;
        adjacency[roads[i].u].emplace_back(roads[i].v, i);
        adjacency[roads[i].v].emplace_back(roads[i].u, i);
    }

    int numShelters;
    inputFile >> numShelters;
    vector<int> shelterCities(numShelters);
    vector<ll> shelterCapacities(numShelters);
    for (int i = 0; i < numShelters; ++i)
    {
        inputFile >> shelterCities[i] >> shelterCapacities[i];
    }

    int numPopulatedCities;
    inputFile >> numPopulatedCities;
    vector<int> populatedCityIds(numPopulatedCities);
    vector<ll> primeCount(numPopulatedCities), elderlyCount(numPopulatedCities);
    for (int i = 0; i < numPopulatedCities; ++i)
    {
        inputFile >> populatedCityIds[i] >> primeCount[i] >> elderlyCount[i];
    }

    ll maxElderlyDistanceKm;
    inputFile >> maxElderlyDistanceKm;
    inputFile.close();

    // 2) Multi-source Dijkstra to compute shortestDistanceKm and parentEdge
    const ll INF = LLONG_MAX / 4;
    vector<ll> shortestDistanceKm(numCities, INF);
    vector<int> parentEdge(numCities, -1);
    vector<int> assignedShelterIndex(numCities, -1);
    using DistPair = pair<ll, int>;
    priority_queue<DistPair, vector<DistPair>, greater<DistPair>> pq;
    // initialize with shelters
    for (int si = 0; si < numShelters; ++si)
    {
        int cityId = shelterCities[si];
        shortestDistanceKm[cityId] = 0;
        assignedShelterIndex[cityId] = si;
        pq.emplace(0, cityId);
    }

    while (!pq.empty())
    {
        auto [distKm, city] = pq.top();
        pq.pop();
        if (distKm != shortestDistanceKm[city])
            continue;
        for (auto &[neighbor, edgeId] : adjacency[city])
        {
            ll newDist = distKm + roads[edgeId].lengthKm;
            if (newDist < shortestDistanceKm[neighbor])
            {
                shortestDistanceKm[neighbor] = newDist;
                parentEdge[neighbor] = edgeId;
                assignedShelterIndex[neighbor] = assignedShelterIndex[city];
                pq.emplace(newDist, neighbor);
            }
        }
    }

    // 3) Reconstruct paths for each populated city
    vector<vector<int>> cityPaths(numPopulatedCities);
    vector<vector<int>> edgePaths(numPopulatedCities);
    for (int i = 0; i < numPopulatedCities; ++i)
    {
        int cityId = populatedCityIds[i];
        int shelterCityId = shelterCities[assignedShelterIndex[cityId]];
        // build edge path backwards
        int current = cityId;
        while (current != shelterCityId)
        {
            int eid = parentEdge[current];
            edgePaths[i].push_back(eid);
            current = (roads[eid].u == current ? roads[eid].v : roads[eid].u);
        }
        // build city path forwards
        current = cityId;
        cityPaths[i].push_back(current);
        for (int eid : edgePaths[i])
        {
            current = (roads[eid].u == current ? roads[eid].v : roads[eid].u);
            cityPaths[i].push_back(current);
        }
        reverse(edgePaths[i].begin(), edgePaths[i].end());
    }

    // Flatten edge paths
    vector<int> hostPathOffsets(numPopulatedCities + 1), hostFlattenedEdges;
    hostPathOffsets[0] = 0;
    for (int i = 0; i < numPopulatedCities; ++i)
    {
        hostPathOffsets[i + 1] = hostPathOffsets[i] + edgePaths[i].size();
        hostFlattenedEdges.insert(hostFlattenedEdges.end(),
                                  edgePaths[i].begin(), edgePaths[i].end());
    }

    // 4) Allocate and copy data to GPU
    int *d_pathOffsets, *d_edgePathIndices, *d_edgeLengths, *d_edgeCap;
    ll *d_primeCount, *d_elderlyCount, *d_edgeReadyTime, *d_arrivalTime;
    ll *d_savedPrime, *d_savedElderly;
    cudaMalloc(&d_pathOffsets, sizeof(int) * (numPopulatedCities + 1));
    cudaMalloc(&d_edgePathIndices, sizeof(int) * hostFlattenedEdges.size());
    cudaMalloc(&d_edgeLengths, sizeof(int) * numRoads);
    cudaMalloc(&d_edgeCap, sizeof(int) * numRoads);
    cudaMalloc(&d_primeCount, sizeof(ll) * numPopulatedCities);
    cudaMalloc(&d_elderlyCount, sizeof(ll) * numPopulatedCities);
    cudaMalloc(&d_edgeReadyTime, sizeof(ll) * numRoads);
    cudaMalloc(&d_arrivalTime, sizeof(ll) * numPopulatedCities);
    cudaMalloc(&d_savedPrime, sizeof(ll) * numPopulatedCities);
    cudaMalloc(&d_savedElderly, sizeof(ll) * numPopulatedCities);

    vector<int> edgeLengths(numRoads), edgeCapacities(numRoads);
    for (int i = 0; i < numRoads; ++i)
    {
        edgeLengths[i] = roads[i].lengthKm;
        edgeCapacities[i] = roads[i].capacity;
    }
    cudaMemcpy(d_pathOffsets, hostPathOffsets.data(), sizeof(int) * (numPopulatedCities + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgePathIndices, hostFlattenedEdges.data(), sizeof(int) * hostFlattenedEdges.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeLengths, edgeLengths.data(), sizeof(int) * numRoads, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeCap, edgeCapacities.data(), sizeof(int) * numRoads, cudaMemcpyHostToDevice);
    cudaMemcpy(d_primeCount, primeCount.data(), sizeof(ll) * numPopulatedCities, cudaMemcpyHostToDevice);
    cudaMemcpy(d_elderlyCount, elderlyCount.data(), sizeof(ll) * numPopulatedCities, cudaMemcpyHostToDevice);
    cudaMemset(d_edgeReadyTime, 0, sizeof(ll) * numRoads);

    // 5) Launch CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numPopulatedCities + threadsPerBlock - 1) / threadsPerBlock;
    evacuationKernel<<<blocksPerGrid, threadsPerBlock>>>(
        numPopulatedCities,
        d_pathOffsets,
        d_edgePathIndices,
        d_edgeLengths,
        d_edgeCap,
        d_primeCount,
        d_elderlyCount,
        maxElderlyDistanceKm,
        d_edgeReadyTime,
        d_arrivalTime,
        d_savedPrime,
        d_savedElderly);
    cudaDeviceSynchronize();

    // 6) Copy back saved counts and arrival times
    vector<ll> savedPrimeResult(numPopulatedCities), savedElderlyResult(numPopulatedCities), arrivalTimes(numPopulatedCities);
    cudaMemcpy(savedPrimeResult.data(), d_savedPrime, sizeof(ll) * numPopulatedCities, cudaMemcpyDeviceToHost);
    cudaMemcpy(savedElderlyResult.data(), d_savedElderly, sizeof(ll) * numPopulatedCities, cudaMemcpyDeviceToHost);
    cudaMemcpy(arrivalTimes.data(), d_arrivalTime, sizeof(ll) * numPopulatedCities, cudaMemcpyDeviceToHost);

    // 7) Enforce shelter capacities in arrival-time order
    vector<vector<int>> indicesByShelter(numShelters);
    for (int i = 0; i < numPopulatedCities; ++i)
    {
        int shelterIndex = assignedShelterIndex[populatedCityIds[i]];
        indicesByShelter[shelterIndex].push_back(i);
    }
    ll totalSavedPrime = 0, totalSavedElderly = 0;
    for (int si = 0; si < numShelters; ++si)
    {
        auto &idxList = indicesByShelter[si];
        sort(idxList.begin(), idxList.end(), [&](int a, int b)
             { return arrivalTimes[a] < arrivalTimes[b]; });
        ll usedCapacity = 0;
        for (int ci : idxList)
        {
            ll canTakeP = min(savedPrimeResult[ci], shelterCapacities[si] - usedCapacity);
            usedCapacity += canTakeP;
            totalSavedPrime += canTakeP;
            ll canTakeE = min(savedElderlyResult[ci], shelterCapacities[si] - usedCapacity);
            usedCapacity += canTakeE;
            totalSavedElderly += canTakeE;
        }
    }

    // 8) Write output
    ofstream outputFile(argv[2]);
    for (int i = 0; i < numPopulatedCities; ++i)
    {
        for (int cityId : cityPaths[i])
            outputFile << cityId << " ";
        outputFile << "\n";
    }
    for (int i = 0; i < numPopulatedCities; ++i)
    {
        int shelterId = shelterCities[assignedShelterIndex[populatedCityIds[i]]];
        outputFile << shelterId << " "
                   << savedPrimeResult[i] << " "
                   << savedElderlyResult[i] << "\n";
    }
    outputFile << "Saved_primes " << totalSavedPrime << "\n";
    outputFile << "Saved_elderly " << totalSavedElderly << "\n";

    // Cleanup GPU memory
    cudaFree(d_pathOffsets);
    cudaFree(d_edgePathIndices);
    cudaFree(d_edgeLengths);
    cudaFree(d_edgeCap);
    cudaFree(d_primeCount);
    cudaFree(d_elderlyCount);
    cudaFree(d_edgeReadyTime);
    cudaFree(d_arrivalTime);
    cudaFree(d_savedPrime);
    cudaFree(d_savedElderly);

    return 0;
}
