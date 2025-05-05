// evac_cuda_descriptive_complete.cu
// CUDA-accelerated evacuation simulator with dynamic multi-shelter routing
// Uses CPU multi-source Dijkstra and CUDA kernel for evacuation simulation
// Compile with: nvcc -O2 evac_cuda_descriptive_complete.cu -o evac_cuda

#include <bits/stdc++.h>
#include <cuda_runtime.h>
using namespace std;
using ll = long long;

// -----------------------------------------------------------------------------
// CUDA kernel: Simulate evacuation along precomputed edgePaths
// -----------------------------------------------------------------------------
__global__ void evacuationKernel(
    int numPop,
    const int *pathOff,
    const int *edgeSeq,
    const int *roadLen,
    const int *roadCap,
    const ll *primeCnt,
    const ll *elderCnt,
    ll *roadReady,
    ll *arrivalMin,
    ll *savedP,
    ll *savedE)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPop)
        return;
    ll group = primeCnt[idx] + elderCnt[idx];
    ll tcur = 0;
    int start = pathOff[idx], end = pathOff[idx + 1];
    for (int off = start; off < end; ++off)
    {
        int r = edgeSeq[off];
        ll km = roadLen[r];
        ll cap = roadCap[r];
        ll batches = (group + cap - 1) / cap;
        ll travel = km * 12LL * batches; // (km/5kmh)*60
        ll prev = atomicMax(&roadReady[r], tcur);
        ll depart = max(tcur, prev);
        ll finish = depart + travel;
        atomicMax(&roadReady[r], finish);
        tcur = finish;
    }
    arrivalMin[idx] = tcur;
    savedP[idx] = primeCnt[idx];
    savedE[idx] = elderCnt[idx];
}

// -----------------------------------------------------------------------------
// Multi-source Dijkstra to compute distKm, parentEdge, assignedShelter
// -----------------------------------------------------------------------------
void multiSourceDijkstra(
    int numCities,
    const vector<vector<pair<int, int>>> &adj,
    const vector<int> &roadLen,
    const vector<int> &shelterCity,
    vector<ll> &distKm,
    vector<int> &parentEdge,
    vector<int> &assignedShelter)
{
    const ll INF = LLONG_MAX / 4;
    distKm.assign(numCities, INF);
    parentEdge.assign(numCities, -1);
    assignedShelter.assign(numCities, -1);
    using PII = pair<ll, int>;
    priority_queue<PII, vector<PII>, greater<PII>> pq;
    int numShel = shelterCity.size();
    for (int s = 0; s < numShel; ++s)
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
        if (d != distKm[u])
            continue;
        for (auto &pr : adj[u])
        {
            int v = pr.first;
            int eid = pr.second;
            ll nd = d + ll(roadLen[eid]);
            if (nd < distKm[v])
            {
                distKm[v] = nd;
                parentEdge[v] = eid;
                assignedShelter[v] = assignedShelter[u];
                pq.emplace(nd, v);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Dijkstra from a single source (for final path reconstruction)
// -----------------------------------------------------------------------------
void dijkstra_from(
    int src,
    int numCities,
    const vector<vector<pair<int, int>>> &adj,
    const vector<int> &roadLen,
    vector<int> &parent)
{
    const ll INF = LLONG_MAX / 4;
    vector<ll> dist(numCities, INF);
    parent.assign(numCities, -1);
    dist[src] = 0;
    using PII = pair<ll, int>;
    priority_queue<PII, vector<PII>, greater<PII>> pq;
    pq.emplace(0, src);
    while (!pq.empty())
    {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u])
            continue;
        for (auto &pr : adj[u])
        {
            int v = pr.first;
            int eid = pr.second;
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
    int numShel;
    fin >> numShel;
    vector<int> shelterCity(numShel);
    vector<ll> shelterCap(numShel);
    for (int i = 0; i < numShel; ++i)
        fin >> shelterCity[i] >> shelterCap[i];
    int numPop;
    fin >> numPop;
    vector<int> popCity(numPop);
    vector<ll> primeCnt(numPop), elderCnt(numPop);
    for (int i = 0; i < numPop; ++i)
        fin >> popCity[i] >> primeCnt[i] >> elderCnt[i];
    ll maxElder;
    fin >> maxElder;
    fin.close();

    // 2) Multi-source Dijkstra on CPU
    vector<ll> distKm;
    vector<int> parentEdge, assignedShelter;
    vector<int> roadLen(numRoads), roadCap(numRoads);
    for (int i = 0; i < numRoads; ++i)
    {
        roadLen[i] = roads[i].lenKm;
        roadCap[i] = roads[i].cap;
    }
    multiSourceDijkstra(numCities, adj, roadLen, shelterCity,
                        distKm, parentEdge, assignedShelter);

    // 3) Reconstruct per-population edge and city paths
    vector<vector<int>> edgePaths(numPop), cityPaths(numPop);
    for (int i = 0; i < numPop; ++i)
    {
        int c = popCity[i];
        int sc = shelterCity[assignedShelter[c]];
        while (c != sc)
        {
            int eid = parentEdge[c];
            edgePaths[i].push_back(eid);
            c = (roads[eid].u == c ? roads[eid].v : roads[eid].u);
        }
        c = popCity[i];
        cityPaths[i].push_back(c);
        for (int eid : edgePaths[i])
        {
            c = (roads[eid].u == c ? roads[eid].v : roads[eid].u);
            cityPaths[i].push_back(c);
        }
    }

    // 4) Flatten edgePaths
    vector<int> pathOffsets(numPop + 1), flatEdges;
    for (int i = 0; i < numPop; ++i)
    {
        pathOffsets[i + 1] = pathOffsets[i] + edgePaths[i].size();
        flatEdges.insert(flatEdges.end(), edgePaths[i].begin(), edgePaths[i].end());
    }

    // 5) Copy data to GPU
    int *d_pathOff, *d_edgeSeq, *d_roadLen, *d_roadCap;
    ll *d_prime, *d_elder, *d_roadReady, *d_arrMin, *d_sP, *d_sE;
    cudaMalloc(&d_pathOff, sizeof(int) * (numPop + 1));
    cudaMalloc(&d_edgeSeq, sizeof(int) * flatEdges.size());
    cudaMalloc(&d_roadLen, sizeof(int) * numRoads);
    cudaMalloc(&d_roadCap, sizeof(int) * numRoads);
    cudaMalloc(&d_prime, sizeof(ll) * numPop);
    cudaMalloc(&d_elder, sizeof(ll) * numPop);
    cudaMalloc(&d_roadReady, sizeof(ll) * numRoads);
    cudaMalloc(&d_arrMin, sizeof(ll) * numPop);
    cudaMalloc(&d_sP, sizeof(ll) * numPop);
    cudaMalloc(&d_sE, sizeof(ll) * numPop);

    cudaMemcpy(d_pathOff, pathOffsets.data(), sizeof(int) * (numPop + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeSeq, flatEdges.data(), sizeof(int) * flatEdges.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_roadLen, roadLen.data(), sizeof(int) * numRoads, cudaMemcpyHostToDevice);
    cudaMemcpy(d_roadCap, roadCap.data(), sizeof(int) * numRoads, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prime, primeCnt.data(), sizeof(ll) * numPop, cudaMemcpyHostToDevice);
    cudaMemcpy(d_elder, elderCnt.data(), sizeof(ll) * numPop, cudaMemcpyHostToDevice);
    cudaMemset(d_roadReady, 0, sizeof(ll) * numRoads);

    // 6) Launch evacuation kernel
    int tpb = 256, bpg = (numPop + tpb - 1) / tpb;
    evacuationKernel<<<bpg, tpb>>>(
        numPop, d_pathOff, d_edgeSeq,
        d_roadLen, d_roadCap,
        d_prime, d_elder,
        d_roadReady, d_arrMin,
        d_sP, d_sE);
    cudaDeviceSynchronize();

    // 7) Copy back results
    vector<ll> arrivalMin(numPop), savedP(numPop), savedE(numPop);
    cudaMemcpy(arrivalMin.data(), d_arrMin, sizeof(ll) * numPop, cudaMemcpyDeviceToHost);
    cudaMemcpy(savedP.data(), d_sP, sizeof(ll) * numPop, cudaMemcpyDeviceToHost);
    cudaMemcpy(savedE.data(), d_sE, sizeof(ll) * numPop, cudaMemcpyDeviceToHost);

    // 8) Enforce elderly distance
    for (int i = 0; i < numPop; ++i)
    {
        if (distKm[popCity[i]] > maxElder)
            savedE[i] = 0;
    }

    // 9) Greedy fill across shelters
    vector<ll> remP = savedP, remE = savedE;
    vector<vector<array<ll, 3>>> drops(numPop);
    // 9a) Record elderly left behind due to distance constraint
    for (int i = 0; i < numPop; ++i)
    {
        if (elderCnt[i] > 0 && distKm[popCity[i]] > maxElder)
        {
            drops[i].push_back({ll(popCity[i]), 0LL, elderCnt[i]});
            remE[i] = 0;
        }
    }
    vector<ll> capLeft = shelterCap;
    ll totalP = 0, totalE = 0;
    vector<pair<ll, int>> events;
    for (int i = 0; i < numPop; ++i)
        events.emplace_back(arrivalMin[i], i);
    sort(events.begin(), events.end());
    for (auto &ev : events)
    {
        int i = ev.second;
        ll p = remP[i], e = remE[i];
        if (p + e == 0)
            continue;
        int pri = assignedShelter[popCity[i]];
        vector<int> order;
        order.push_back(pri);
        for (int s = 0; s < numShel; ++s)
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
            ll dp = min(p, capLeft[s]);
            capLeft[s] -= dp;
            p -= dp;
            remP[i] -= dp;
            drops[i].push_back({ll(shelterCity[s]), dp, de});
            totalP += dp;
            totalE += de;
        }
    }

    // Build parentMap for drop destinations
    unordered_set<int> used;
    for (auto &dlist : drops)
        for (auto &d : dlist)
            used.insert(d[0]);
    unordered_map<int, vector<int>> parentMap;
    for (int dest : used)
    {
        vector<int> par;
        dijkstra_from(dest, numCities, adj, roadLen, par);
        parentMap[dest] = move(par);
    }

    // 10) Reconstruct final paths merging all drop segments
    vector<vector<int>> finalPaths(numPop);
    for (int i = 0; i < numPop; ++i)
    {
        vector<int> path;
        int prev = popCity[i];
        path.push_back(prev);
        for (auto &d : drops[i])
        {
            int dest = d[0];
            auto &par = parentMap[dest];
            int cur = prev;
            while (cur != dest)
            {
                cur = par[cur];
                path.push_back(cur);
            }
            prev = dest;
        }
        finalPaths[i] = move(path);
    }

    // 11) Output
    ofstream fout(argv[2]);
    for (auto &p : finalPaths)
    {
        for (int c : p)
            fout << c << " ";
        fout << "\n";
    }
    for (auto &dlist : drops)
    {
        fout << dlist.size();
        for (auto &ar : dlist)
            fout << " " << ar[0] << " " << ar[1] << " " << ar[2];
        fout << "\n";
    }
    fout << "Saved_primes " << totalP << "\n";
    fout << "New Saved_elderly " << totalE << "\n";
    fout.close();

    // Cleanup
    cudaFree(d_pathOff);
    cudaFree(d_edgeSeq);
    cudaFree(d_roadLen);
    cudaFree(d_roadCap);
    cudaFree(d_prime);
    cudaFree(d_elder);
    cudaFree(d_roadReady);
    cudaFree(d_arrMin);
    cudaFree(d_sP);
    cudaFree(d_sE);
    return 0;
}
