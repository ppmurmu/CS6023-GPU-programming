// evac_cuda_descriptive_complete.cu
// CUDA-accelerated evacuation simulator with dynamic multi-shelter routing
// Uses CPU multi-source Dijkstra and CUDA kernel for evacuation simulation
// Compile with: nvcc -O2 evac_cuda_descriptive_complete.cu -o evac_cuda_descriptive_complete

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
    for (int s = 0; s < (int)shelterCity.size(); ++s)
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
            int v = pr.first, eid = pr.second;
            ll nd = d + roadLen[eid];
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
            int v = pr.first, eid = pr.second;
            ll nd = d + roadLen[eid];
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

    // 2) Compute shortest distances & assigned shelters
    vector<ll> distKm;
    vector<int> parentEdge, assignedShelter;
    vector<int> roadLenVec(numRoads), roadCapVec(numRoads);
    for (int i = 0; i < numRoads; ++i)
    {
        roadLenVec[i] = roads[i].lenKm;
        roadCapVec[i] = roads[i].cap;
    }
    multiSourceDijkstra(numCities, adj, roadLenVec, shelterCity,
                        distKm, parentEdge, assignedShelter);

    // 3) Build per-population paths
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
    }

    // 4) Flatten edgePaths
    vector<int> pathOff(numPop + 1, 0), flatEdges;
    for (int i = 0; i < numPop; ++i)
    {
        pathOff[i + 1] = pathOff[i] + edgePaths[i].size();
        flatEdges.insert(flatEdges.end(), edgePaths[i].begin(), edgePaths[i].end());
    }

    // 5) Zero out elders beyond maxElder
    vector<ll> simElder = elderCnt;
    for (int i = 0; i < numPop; ++i)
        if (distKm[popCity[i]] > maxElder)
            simElder[i] = 0;

    // 6) Allocate and copy to GPU
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

    cudaMemcpy(d_pathOff, pathOff.data(), sizeof(int) * (numPop + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeSeq, flatEdges.data(), sizeof(int) * flatEdges.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_roadLen, roadLenVec.data(), sizeof(int) * numRoads, cudaMemcpyHostToDevice);
    cudaMemcpy(d_roadCap, roadCapVec.data(), sizeof(int) * numRoads, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prime, primeCnt.data(), sizeof(ll) * numPop, cudaMemcpyHostToDevice);
    cudaMemcpy(d_elder, simElder.data(), sizeof(ll) * numPop, cudaMemcpyHostToDevice);
    cudaMemset(d_roadReady, 0, sizeof(ll) * numRoads);

    // 7) Launch kernel
    int tpb = 256, bpg = (numPop + tpb - 1) / tpb;
    evacuationKernel<<<bpg, tpb>>>(numPop, d_pathOff, d_edgeSeq,
                                   d_roadLen, d_roadCap,
                                   d_prime, d_elder,
                                   d_roadReady, d_arrMin,
                                   d_sP, d_sE);
    cudaDeviceSynchronize();

    // 8) Copy back results
    vector<ll> arrivalMin(numPop), savedP(numPop), savedE(numPop);
    cudaMemcpy(arrivalMin.data(), d_arrMin, sizeof(ll) * numPop, cudaMemcpyDeviceToHost);
    cudaMemcpy(savedP.data(), d_sP, sizeof(ll) * numPop, cudaMemcpyDeviceToHost);
    cudaMemcpy(savedE.data(), d_sE, sizeof(ll) * numPop, cudaMemcpyDeviceToHost);

    // 9) Greedy fill & drop recording
    vector<ll> remP = savedP, remE = savedE;
    vector<vector<array<ll, 3>>> drops(numPop);
    // record elders dropped by distance
    for (int i = 0; i < numPop; ++i)
    {
        if (elderCnt[i] > 0 && distKm[popCity[i]] > maxElder)
        {
            drops[i].push_back({(ll)popCity[i], 0LL, elderCnt[i]});
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
        vector<int> order{pri};
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
            drops[i].push_back({(ll)shelterCity[s], dp, de});
            totalP += dp;
            totalE += de;
        }
    }

    // 10) Reconstruct final paths
    unordered_set<int> used;
    for (auto &lst : drops)
        for (auto &d : lst)
            used.insert(d[0]);
    unordered_map<int, vector<int>> parentMap;
    for (int dest : used)
    {
        vector<int> par;
        dijkstra_from(dest, numCities, adj, roadLenVec, par);
        parentMap[dest] = move(par);
    }
    vector<vector<int>> finalPaths(numPop);
    for (int i = 0; i < numPop; ++i)
    {
        vector<int> path;
        int cur = popCity[i];
        path.push_back(cur);
        for (auto &d : drops[i])
        {
            int dst = d[0];
            auto &par = parentMap[dst];
            while (cur != dst)
            {
                cur = par[cur];
                path.push_back(cur);
            }
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
    for (auto &lst : drops)
    {

        for (auto &d : lst)
            fout << d[0] << " " << d[1] << " " << d[2] << " ";
        fout << "\n";
    }
    // fout<<"Saved_primes "<<totalP<<"\n";
    // fout<<"New Saved_elderly "<<totalE<<"\n";
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
