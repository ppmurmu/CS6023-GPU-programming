// evac_cuda_descriptive_complete.cu
// CUDA-accelerated evacuation simulator with dynamic multi-shelter routing
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
    const int *pathOffsets,
    const int *edgeSequence,
    const int *roadLengthsKm,
    const int *roadCapacities,
    const ll *primeCounts,
    const ll *elderlyCounts,
    ll *roadReadyMinute,
    ll *arrivalMinute,
    ll *savedPrime,
    ll *savedElderly)
{
    int cityIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cityIdx >= numPopCities)
        return;

    ll groupSize = primeCounts[cityIdx] + elderlyCounts[cityIdx];
    ll currentMin = 0;

    int startOff = pathOffsets[cityIdx];
    int endOff = pathOffsets[cityIdx + 1];
    for (int off = startOff; off < endOff; ++off)
    {
        int roadId = edgeSequence[off];
        int km = roadLengthsKm[roadId];
        int capacity = roadCapacities[roadId];
        int batches = (int)((groupSize + capacity - 1) / capacity);
        ll travelMin = ll(km) * 12LL * batches; // 12 = 60/5 km/h

        ll prevReady = atomicMax(&roadReadyMinute[roadId], currentMin);
        ll departMin = max(currentMin, prevReady);
        ll finishMin = departMin + travelMin;

        atomicMax(&roadReadyMinute[roadId], finishMin);
        currentMin = finishMin;
    }

    arrivalMinute[cityIdx] = currentMin;
    savedPrime[cityIdx] = primeCounts[cityIdx];
    savedElderly[cityIdx] = elderlyCounts[cityIdx];
}

// -----------------------------------------------------------------------------
// Helper: Dijkstra from a single source to build parent pointers
// -----------------------------------------------------------------------------
void dijkstra_from(int source,
                   int numCities,
                   const vector<vector<pair<int, int>>> &adj,
                   const vector<int> &roadLen,
                   vector<int> &parent)
{
    const ll INF = LLONG_MAX / 4;
    vector<ll> dist(numCities, INF);
    parent.assign(numCities, -1);

    using PII = pair<ll, int>;
    priority_queue<PII, vector<PII>, greater<PII>> pq;
    dist[source] = 0;
    pq.emplace(0, source);

    while (!pq.empty())
    {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u])
            continue;
        for (auto &pr : adj[u])
        {
            int v = pr.first, eid = pr.second;
            ll nd = d + ll(roadLen[eid]);
            if (nd < dist[v])
            {
                dist[v] = nd;
                parent[v] = u;
                pq.emplace(nd, v);
            }
        }
    }
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
        cerr << "Usage: " << argv[0] << " <input> <output>\n";
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
    vector<vector<pair<int, int>>> adj(numCities);
    for (int i = 0; i < numRoads; ++i)
    {
        fin >> roads[i].u >> roads[i].v >> roads[i].lenKm >> roads[i].cap;
        adj[roads[i].u].emplace_back(roads[i].v, i);
        adj[roads[i].v].emplace_back(roads[i].u, i);
    }

    int numShelters;
    fin >> numShelters;
    vector<int> shelterCity(numShelters);
    vector<ll> shelterCap(numShelters);
    for (int i = 0; i < numShelters; ++i)
        fin >> shelterCity[i] >> shelterCap[i];

    int numPop;
    fin >> numPop;
    vector<int> popCityId(numPop);
    vector<ll> primeCount(numPop), elderlyCount(numPop);
    for (int i = 0; i < numPop; ++i)
        fin >> popCityId[i] >> primeCount[i] >> elderlyCount[i];

    ll maxElderlyDist;
    fin >> maxElderlyDist;
    fin.close();

    // 2) Multi-source Dijkstra to nearest shelter
    const ll INF = LLONG_MAX / 4;
    vector<ll> distKm(numCities, INF);
    vector<int> parentEdge(numCities, -1), assignedShelter(numCities, -1);
    using PII = pair<ll, int>;
    priority_queue<PII, vector<PII>, greater<PII>> pq;
    for (int s = 0; s < numShelters; ++s)
    {
        int c = shelterCity[s];
        distKm[c] = 0;
        assignedShelter[c] = s;
        pq.emplace(0, c);
    }
    while (!pq.empty())
    {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > distKm[u])
            continue;
        for (auto &pr : adj[u])
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

    // 3) Build initial per-pop city edgePaths
    vector<vector<int>> edgePaths(numPop);
    for (int i = 0; i < numPop; ++i)
    {
        int c = popCityId[i];
        int sc = shelterCity[assignedShelter[c]];
        while (c != sc)
        {
            int eid = parentEdge[c];
            edgePaths[i].push_back(eid);
            c = (roads[eid].u == c ? roads[eid].v : roads[eid].u);
        }
    }

    // 4) Flatten for GPU
    vector<int> pathOffsets(numPop + 1, 0), flatEdges;
    for (int i = 0; i < numPop; ++i)
    {
        pathOffsets[i + 1] = pathOffsets[i] + edgePaths[i].size();
        flatEdges.insert(flatEdges.end(), edgePaths[i].begin(), edgePaths[i].end());
    }

    // 5) GPU alloc & copy
    int *d_pathOff, *d_edgeSeq, *d_len, *d_cap;
    ll *d_pCnt, *d_eCnt, *d_roadReady, *d_arrMin, *d_sP, *d_sE;
    cudaMalloc(&d_pathOff, sizeof(int) * (numPop + 1));
    cudaMalloc(&d_edgeSeq, sizeof(int) * flatEdges.size());
    cudaMalloc(&d_len, sizeof(int) * numRoads);
    cudaMalloc(&d_cap, sizeof(int) * numRoads);
    cudaMalloc(&d_pCnt, sizeof(ll) * numPop);
    cudaMalloc(&d_eCnt, sizeof(ll) * numPop);
    cudaMalloc(&d_roadReady, sizeof(ll) * numRoads);
    cudaMalloc(&d_arrMin, sizeof(ll) * numPop);
    cudaMalloc(&d_sP, sizeof(ll) * numPop);
    cudaMalloc(&d_sE, sizeof(ll) * numPop);

    vector<int> hLen(numRoads), hCap(numRoads);
    for (int i = 0; i < numRoads; ++i)
    {
        hLen[i] = roads[i].lenKm;
        hCap[i] = roads[i].cap;
    }
    cudaMemcpy(d_pathOff, pathOffsets.data(), sizeof(int) * (numPop + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeSeq, flatEdges.data(), sizeof(int) * flatEdges.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_len, hLen.data(), sizeof(int) * numRoads, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cap, hCap.data(), sizeof(int) * numRoads, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pCnt, primeCount.data(), sizeof(ll) * numPop, cudaMemcpyHostToDevice);
    cudaMemcpy(d_eCnt, elderlyCount.data(), sizeof(ll) * numPop, cudaMemcpyHostToDevice);
    cudaMemset(d_roadReady, 0, sizeof(ll) * numRoads);

    // 6) Launch kernel
    int tp = 256, bs = (numPop + tp - 1) / tp;
    evacuationKernel<<<bs, tp>>>(numPop,
                                 d_pathOff, d_edgeSeq, d_len, d_cap,
                                 d_pCnt, d_eCnt,
                                 d_roadReady, d_arrMin,
                                 d_sP, d_sE);
    cudaDeviceSynchronize();

    // 7) Copy back
    vector<ll> arrivalMin(numPop), savedPrime(numPop), savedElderly(numPop);
    cudaMemcpy(arrivalMin.data(), d_arrMin, sizeof(ll) * numPop, cudaMemcpyDeviceToHost);
    cudaMemcpy(savedPrime.data(), d_sP, sizeof(ll) * numPop, cudaMemcpyDeviceToHost);
    cudaMemcpy(savedElderly.data(), d_sE, sizeof(ll) * numPop, cudaMemcpyDeviceToHost);

    // 8) Elderly cap
    for (int i = 0; i < numPop; ++i)
        if (distKm[popCityId[i]] > maxElderlyDist)
            savedElderly[i] = 0;

    // 9) Global greedy fill & record dest shelters
    vector<pair<ll, int>> events;
    events.reserve(numPop);
    for (int i = 0; i < numPop; ++i)
        events.emplace_back(arrivalMin[i], i);
    sort(events.begin(), events.end());

    vector<ll> remP = savedPrime, remE = savedElderly;
    vector<vector<array<ll, 3>>> drops(numPop);
    vector<ll> capLeft = shelterCap;
    ll totalP = 0, totalE = 0;
    unordered_set<int> usedDest;

    for (auto &ev : events)
    {
        int i = ev.second;
        ll p = remP[i], e = remE[i];
        if (p + e == 0)
            continue;
        int pri = assignedShelter[popCityId[i]];
        vector<int> order;
        order.push_back(pri);
        for (int s = 0; s < numShelters; ++s)
            if (s != pri)
                order.push_back(s);
        for (int s : order)
        {
            if (p + e == 0)
                break;
            if (capLeft[s] == 0)
                continue;
            ll de = min(e, capLeft[s]);
            capLeft[s] -= de;
            e -= de;
            remE[i] -= de;
            totalE += de;
            ll dp = min(p, capLeft[s]);
            capLeft[s] -= dp;
            p -= dp;
            remP[i] -= dp;
            totalP += dp;
            drops[i].push_back({ll(shelterCity[s]), dp, de});
            usedDest.insert(shelterCity[s]);
        }
    }

    // 10) Recompute full paths to final shelters
    // map dest city -> parent pointers
    unordered_map<int, vector<int>> parentMap;
    for (int dest : usedDest)
    {
        vector<int> prev;
        dijkstra_from(dest, numCities, adj, hLen, prev);
        parentMap[dest] = move(prev);
    }

    vector<vector<int>> finalPaths(numPop);
    for (int i = 0; i < numPop; ++i)
    {
        if (drops[i].empty())
        {
            finalPaths[i] = {popCityId[i]};
        }
        else
        {
            int destCity = drops[i].back()[0];
            vector<int> path;
            int cur = popCityId[i];
            path.push_back(cur);
            auto &prev = parentMap[destCity];
            while (cur != destCity)
            {
                cur = prev[cur];
                if (cur < 0)
                {
                    break;
                }
                path.push_back(cur);
            }
            finalPaths[i] = move(path);
        }
    }

    // 11) Write output
    ofstream fout(argv[2]);
    // 11a) paths
    for (auto &path : finalPaths)
    {
        for (int c : path)
            fout << c << " ";
        fout << "\n";
    }
    // 11b) drops (count + triples)
    for (auto &dlist : drops)
    {
        fout << dlist.size();
        for (auto &d : dlist)
            fout << " " << d[0] << " " << d[1] << " " << d[2];
        fout << "\n";
    }
    // 11c) summary
    fout << "Saved_primes " << totalP << "\n";
    fout << "New Saved_elderly " << totalE << "\n";
    fout.close();

    // 12) Cleanup
    cudaFree(d_pathOff);
    cudaFree(d_edgeSeq);
    cudaFree(d_len);
    cudaFree(d_cap);
    cudaFree(d_pCnt);
    cudaFree(d_eCnt);
    cudaFree(d_roadReady);
    cudaFree(d_arrMin);
    cudaFree(d_sP);
    cudaFree(d_sE);

    return 0;
}
