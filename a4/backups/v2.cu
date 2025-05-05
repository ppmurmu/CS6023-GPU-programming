// evac_cuda_descriptive_complete.cu
// CUDA‐accelerated evacuation simulator with descriptive names and comments
// Compile with: nvcc -O2 evac_cuda_descriptive_complete.cu -o evac_cuda

#include <bits/stdc++.h>
#include <cuda_runtime.h>
using namespace std;
using ll = long long;

// -----------------------------------------------------------------------------
// CUDA kernel: Simulate one populated city's evacuating group walking its path
// -----------------------------------------------------------------------------
__global__ void evacuationKernel(
    int numPopCities,
    const int *pathOffsets,    // start index of each city's edge path
    const int *edgeSequence,   // flattened list of road IDs for all cities
    const int *roadLengthsKm,  // km per road ID
    const int *roadCapacities, // capacity per road ID
    const ll *primeCounts,     // prime-age count per city
    const ll *elderlyCounts,   // elderly count per city
    ll *roadReadyMinute,       // per-road next-available time (minutes)
    ll *arrivalMinute,         // per-city final arrival time (minutes)
    ll *savedPrime,            // per-city preliminarily saved primes (all)
    ll *savedElderly           // per-city preliminarily saved elderly (all)
)
{
    int cityIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cityIdx >= numPopCities)
        return;

    // Compute group size and initialize current simulated time
    ll groupSize = primeCounts[cityIdx] + elderlyCounts[cityIdx];
    ll currentMin = 0;

    // Walk each road in the city's path
    int startOff = pathOffsets[cityIdx];
    int endOff = pathOffsets[cityIdx + 1];
    for (int off = startOff; off < endOff; ++off)
    {
        int roadId = edgeSequence[off];
        int km = roadLengthsKm[roadId];
        int capacity = roadCapacities[roadId];
        int batches = int((groupSize + capacity - 1) / capacity);
        // travel time = (km / 5 km/h)*60 min * batches = 12 * km * batches
        ll travelMin = ll(km) * 12LL * batches;

        // Queue contention: wait for road to be free
        ll prevReady = atomicMax(&roadReadyMinute[roadId], currentMin);
        ll departMin = max(currentMin, prevReady);
        ll finishMin = departMin + travelMin;
        // Update road-ready and current time
        atomicMax(&roadReadyMinute[roadId], finishMin);
        currentMin = finishMin;
    }

    // Record results
    arrivalMinute[cityIdx] = currentMin;
    savedPrime[cityIdx] = primeCounts[cityIdx];     // all primes
    savedElderly[cityIdx] = elderlyCounts[cityIdx]; // filter on host
}

// -----------------------------------------------------------------------------
// Main program
// -----------------------------------------------------------------------------
int main(int argc, char **argv)
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return 1;
    }

    // 1) Read input
    ifstream fin(argv[1]);
    int numCities, numRoads;
    fin >> numCities >> numRoads;
    struct Road
    {
        int u, v, lenKm, cap;
    };
    vector<Road> roads(numRoads);
    vector<vector<pair<int, int>>> adjacency(numCities);
    for (int i = 0; i < numRoads; ++i)
    {
        fin >> roads[i].u >> roads[i].v >> roads[i].lenKm >> roads[i].cap;
        adjacency[roads[i].u].emplace_back(roads[i].v, i);
        adjacency[roads[i].v].emplace_back(roads[i].u, i);
    }

    int numShelters;
    fin >> numShelters;
    vector<int> shelterCity(numShelters);
    vector<ll> shelterCapacity(numShelters);
    for (int i = 0; i < numShelters; ++i)
    {
        fin >> shelterCity[i] >> shelterCapacity[i];
    }

    int numPopCities;
    fin >> numPopCities;
    vector<int> popCityId(numPopCities);
    vector<ll> primeCount(numPopCities), elderlyCount(numPopCities);
    for (int i = 0; i < numPopCities; ++i)
    {
        fin >> popCityId[i] >> primeCount[i] >> elderlyCount[i];
    }

    ll maxElderlyDistanceKm;
    fin >> maxElderlyDistanceKm;
    fin.close();

    // 2) Multi-source Dijkstra to compute:
    //    distKm[city]       = shortest km to nearest shelter
    //    parentEdge[city]   = road ID leading toward that nearest shelter
    //    assignedShelter[city] = index of that shelter
    const ll INF = LLONG_MAX / 4;
    vector<ll> distKm(numCities, INF);
    vector<int> parentEdge(numCities, -1), assignedShelter(numCities, -1);
    using Pair = pair<ll, int>;
    priority_queue<Pair, vector<Pair>, greater<Pair>> pq;
    for (int si = 0; si < numShelters; ++si)
    {
        int c = shelterCity[si];
        distKm[c] = 0;
        assignedShelter[c] = si;
        pq.emplace(0, c);
    }
    while (!pq.empty())
    {
        auto [d, u] = pq.top();
        pq.pop();
        if (d != distKm[u])
            continue;
        for (auto &pr : adjacency[u])
        {
            int v = pr.first, eid = pr.second;
            ll nd = d + ll(roads[eid].lenKm);
            if (nd < distKm[v])
            {
                distKm[v] = nd;
                parentEdge[v] = eid;
                assignedShelter[v] = assignedShelter[u];
                pq.emplace(nd, v);
            }
        }
    }

    // 3) Reconstruct per-populated-city paths (forward direction)
    //    cityPaths[i] = list of city IDs from popCityId[i] → shelter
    //    edgePaths[i] = list of road IDs along that route
    vector<vector<int>> cityPaths(numPopCities), edgePaths(numPopCities);
    for (int i = 0; i < numPopCities; ++i)
    {
        int curCity = popCityId[i];
        int shelterC = shelterCity[assignedShelter[curCity]];
        // Build edge path forward
        while (curCity != shelterC)
        {
            int eid = parentEdge[curCity];
            edgePaths[i].push_back(eid);
            // step to next city
            curCity = (roads[eid].u == curCity ? roads[eid].v : roads[eid].u);
        }
        // Build city path by walking same edges
        curCity = popCityId[i];
        cityPaths[i].push_back(curCity);
        for (int eid : edgePaths[i])
        {
            curCity = (roads[eid].u == curCity ? roads[eid].v : roads[eid].u);
            cityPaths[i].push_back(curCity);
        }
    }

    // 4) Flatten edgePaths into arrays for GPU
    vector<int> pathOffsets(numPopCities + 1, 0), flatEdgeIds;
    flatEdgeIds.reserve(pathOffsets.back());
    for (int i = 0; i < numPopCities; ++i)
    {
        pathOffsets[i + 1] = pathOffsets[i] + int(edgePaths[i].size());
        flatEdgeIds.insert(flatEdgeIds.end(),
                           edgePaths[i].begin(), edgePaths[i].end());
    }

    // 5) Allocate & copy data to GPU
    int *d_pathOffsets, *d_edgeSeq, *d_roadLenKm, *d_roadCap;
    ll *d_primeCnt, *d_elderCnt, *d_roadReady, *d_arrivalMin,
        *d_savedPrime, *d_savedElderly;
    cudaMalloc(&d_pathOffsets, sizeof(int) * (numPopCities + 1));
    cudaMalloc(&d_edgeSeq, sizeof(int) * flatEdgeIds.size());
    cudaMalloc(&d_roadLenKm, sizeof(int) * numRoads);
    cudaMalloc(&d_roadCap, sizeof(int) * numRoads);
    cudaMalloc(&d_primeCnt, sizeof(ll) * numPopCities);
    cudaMalloc(&d_elderCnt, sizeof(ll) * numPopCities);
    cudaMalloc(&d_roadReady, sizeof(ll) * numRoads);
    cudaMalloc(&d_arrivalMin, sizeof(ll) * numPopCities);
    cudaMalloc(&d_savedPrime, sizeof(ll) * numPopCities);
    cudaMalloc(&d_savedElderly, sizeof(ll) * numPopCities);

    vector<int> hostLen(numRoads), hostCap(numRoads);
    for (int i = 0; i < numRoads; ++i)
    {
        hostLen[i] = roads[i].lenKm;
        hostCap[i] = roads[i].cap;
    }
    cudaMemcpy(d_pathOffsets, pathOffsets.data(),
               sizeof(int) * (numPopCities + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeSeq, flatEdgeIds.data(),
               sizeof(int) * flatEdgeIds.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_roadLenKm, hostLen.data(),
               sizeof(int) * numRoads, cudaMemcpyHostToDevice);
    cudaMemcpy(d_roadCap, hostCap.data(),
               sizeof(int) * numRoads, cudaMemcpyHostToDevice);
    cudaMemcpy(d_primeCnt, primeCount.data(),
               sizeof(ll) * numPopCities, cudaMemcpyHostToDevice);
    cudaMemcpy(d_elderCnt, elderlyCount.data(),
               sizeof(ll) * numPopCities, cudaMemcpyHostToDevice);
    cudaMemset(d_roadReady, 0, sizeof(ll) * numRoads);

    // 6) Launch CUDA kernel
    int threadsPerBlock = 256;
    int blocks = (numPopCities + threadsPerBlock - 1) / threadsPerBlock;
    evacuationKernel<<<blocks, threadsPerBlock>>>(
        numPopCities,
        d_pathOffsets,
        d_edgeSeq,
        d_roadLenKm,
        d_roadCap,
        d_primeCnt,
        d_elderCnt,
        d_roadReady,
        d_arrivalMin,
        d_savedPrime,
        d_savedElderly);
    cudaDeviceSynchronize();

    // 7) Copy results back to host
    vector<ll> arrivalMin(numPopCities),
        savedPrime(numPopCities),
        savedElderly(numPopCities);
    cudaMemcpy(arrivalMin.data(), d_arrivalMin,
               sizeof(ll) * numPopCities, cudaMemcpyDeviceToHost);
    cudaMemcpy(savedPrime.data(), d_savedPrime,
               sizeof(ll) * numPopCities, cudaMemcpyDeviceToHost);
    cudaMemcpy(savedElderly.data(), d_savedElderly,
               sizeof(ll) * numPopCities, cudaMemcpyDeviceToHost);

    // 8) Enforce elderly distance limit on host
    for (int i = 0; i < numPopCities; ++i)
    {
        if (distKm[popCityId[i]] > maxElderlyDistanceKm)
        {
            savedElderly[i] = 0;
        }
    }

    // 9) Enforce shelter capacities (elderly-first, by arrival order)
    vector<vector<int>> citiesByShelter(numShelters);
    for (int i = 0; i < numPopCities; ++i)
    {
        int sidx = assignedShelter[popCityId[i]];
        citiesByShelter[sidx].push_back(i);
    }

    ll totalSavedPrime = 0, totalSavedElderly = 0;
    for (int s = 0; s < numShelters; ++s)
    {
        auto &list = citiesByShelter[s];
        sort(list.begin(), list.end(),
             [&](int a, int b)
             { return arrivalMin[a] < arrivalMin[b]; });
        ll usedCap = 0, cap = shelterCapacity[s];
        for (int idx : list)
        {
            ll takeE = min(savedElderly[idx], cap - usedCap);
            usedCap += takeE;
            totalSavedElderly += takeE;
            ll takeP = min(savedPrime[idx], cap - usedCap);
            usedCap += takeP;
            totalSavedPrime += takeP;
            if (usedCap == cap)
                break;
        }
    }

    // 10) Write output
    ofstream fout(argv[2]);
    // Evacuation paths
    for (int i = 0; i < numPopCities; ++i)
    {
        for (int city : cityPaths[i])
            fout << city << " ";
        fout << "\n";
    }
    // Drop events
    for (int i = 0; i < numPopCities; ++i)
    {
        int shCity = shelterCity[assignedShelter[popCityId[i]]];
        fout << shCity << " "
             << savedPrime[i] << " "
             << savedElderly[i] << "\n";
    }
    // Summary
    fout << "Saved_primes " << totalSavedPrime << "\n";
    fout << "New Saved_elderly " << totalSavedElderly << "\n";
    fout.close();

    // 11) Cleanup GPU memory
    cudaFree(d_pathOffsets);
    cudaFree(d_edgeSeq);
    cudaFree(d_roadLenKm);
    cudaFree(d_roadCap);
    cudaFree(d_primeCnt);
    cudaFree(d_elderCnt);
    cudaFree(d_roadReady);
    cudaFree(d_arrivalMin);
    cudaFree(d_savedPrime);
    cudaFree(d_savedElderly);

    return 0;
}
