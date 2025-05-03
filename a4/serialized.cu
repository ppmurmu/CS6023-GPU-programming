#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
using namespace std;

struct Edge
{
    int to;
    int length;
};

const long long INF = numeric_limits<long long>::max();

// Run Dijkstra from `src` over `graph`, filling `dist` and `parent`
void dijkstra(int src,
              const vector<vector<Edge>> &graph,
              vector<long long> &dist,
              vector<int> &parent)
{
    int n = graph.size();
    dist.assign(n, INF);
    parent.assign(n, -1);
    dist[src] = 0;
    using P = pair<long long, int>;
    priority_queue<P, vector<P>, greater<P>> pq;
    pq.emplace(0, src);

    while (!pq.empty())
    {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u])
            continue;
        for (auto &e : graph[u])
        {
            long long nd = d + e.length;
            if (nd < dist[e.to])
            {
                dist[e.to] = nd;
                parent[e.to] = u;
                pq.emplace(nd, e.to);
            }
        }
    }
}

// Reconstruct path from `s` to `t` using `parent`
// returns vector of nodes [s, ..., t]
vector<int> build_path(int s, int t, const vector<int> &parent)
{
    vector<int> rev;
    for (int cur = t; cur != -1; cur = parent[cur])
    {
        rev.push_back(cur);
        if (cur == s)
            break;
    }
    reverse(rev.begin(), rev.end());
    return rev;
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return 1;
    }

    //-------------- input --------------------------------
    ifstream infile(argv[1]);
    if (!infile)
    {
        cerr << "Error: Cannot open file " << argv[1] << "\n";
        return 1;
    }

    long long num_cities, num_roads;
    infile >> num_cities >> num_roads;

    vector<array<int, 4>> roads(num_roads);
    for (int i = 0; i < num_roads; i++)
    {
        infile >> roads[i][0] // u
            >> roads[i][1]    // v
            >> roads[i][2]    // length
            >> roads[i][3];   // cap (ignored in sequential)
    }

    int num_shelters;
    infile >> num_shelters;
    vector<int> shelter_city(num_shelters);
    vector<long long> shelter_capacity(num_shelters);
    for (int i = 0; i < num_shelters; i++)
    {
        infile >> shelter_city[i] >> shelter_capacity[i];
    }

    int num_pop;
    infile >> num_pop;
    vector<int> city(num_pop);
    vector<long long> pop_prime(num_pop), pop_elder(num_pop);
    for (int i = 0; i < num_pop; i++)
    {
        infile >> city[i] >> pop_prime[i] >> pop_elder[i];
    }

    long long max_dist_elderly;
    infile >> max_dist_elderly;
    infile.close();

    // Build undirected graph
    vector<vector<Edge>> graph(num_cities);
    for (auto &r : roads)
    {
        int u = r[0], v = r[1], L = r[2];
        graph[u].push_back({v, L});
        graph[v].push_back({u, L});
    }

    // Prepare intermediate storage
    vector<vector<int>> allPaths(num_pop);
    vector<vector<array<long long, 3>>> allDrops(num_pop);

    vector<long long> dist;
    vector<int> parent;

    // Sequential evacuation per populated city
    for (int i = 0; i < num_pop; i++)
    {
        long long remainE = pop_elder[i];
        long long remainP = pop_prime[i];
        int start = city[i];
        int curNode = start;

        // Compute distances from start to all shelters
        dijkstra(start, graph, dist, parent);

        // Order shelters by distance
        vector<pair<long long, int>> order;
        for (int s = 0; s < num_shelters; s++)
        {
            order.emplace_back(dist[shelter_city[s]], s);
        }
        sort(order.begin(), order.end(),
             [&](auto &a, auto &b)
             { return a.first < b.first; });

        // Begin path at origin
        allPaths[i].push_back(curNode);

        // If elders can't reach the nearest shelter, drop them here
        if (remainE > 0)
        {
            long long d0 = order.front().first;
            if (d0 > max_dist_elderly)
            {
                allDrops[i].push_back({(long long)curNode, 0LL, remainE});
                remainE = 0;
            }
        }

        // Visit shelters in order, drop elders then primes
        for (auto &o : order)
        {
            if (remainE == 0 && remainP == 0)
                break;
            int si = o.second;
            int scity = shelter_city[si];
            long long d = o.first;
            if (d == INF)
                continue;
            if (remainE > 0 && d > max_dist_elderly)
                continue;
            if (shelter_capacity[si] == 0)
                continue;

            // Travel from curNode to shelter city
            if (curNode != scity)
            {
                dijkstra(curNode, graph, dist, parent);
                auto seg = build_path(curNode, scity, parent);
                for (int k = 1; k < (int)seg.size(); k++)
                    allPaths[i].push_back(seg[k]);
                curNode = scity;
            }

            // Drop elders first
            long long dropE = min(remainE, shelter_capacity[si]);
            shelter_capacity[si] -= dropE;
            remainE -= dropE;

            // Then drop primes
            long long dropP = min(remainP, shelter_capacity[si]);
            shelter_capacity[si] -= dropP;
            remainP -= dropP;

            allDrops[i].push_back({(long long)scity, dropP, dropE});
        }

        // If anyone remains, drop at current node
        if (remainE > 0 || remainP > 0)
        {
            allDrops[i].push_back({(long long)curNode, remainP, remainE});
        }
    }

    // Allocate output arrays matching original signatures
    long long *path_size = new long long[num_pop];
    long long **paths = new long long *[num_pop];
    long long *num_drops = new long long[num_pop];
    long long ***drops = new long long **[num_pop];

    // Fill them
    for (int i = 0; i < num_pop; i++)
    {
        // Paths
        long long P = allPaths[i].size();
        path_size[i] = P;
        paths[i] = new long long[P];
        for (int j = 0; j < P; j++)
        {
            paths[i][j] = allPaths[i][j];
        }

        // Drops
        long long D = allDrops[i].size();
        num_drops[i] = D;
        drops[i] = new long long *[D];
        for (int j = 0; j < D; j++)
        {
            drops[i][j] = new long long[3];
            drops[i][j][0] = allDrops[i][j][0]; // city
            drops[i][j][1] = allDrops[i][j][1]; // primes
            drops[i][j][2] = allDrops[i][j][2]; // elders
        }
    }

    //------------ output -----------------
    ofstream outfile(argv[2]);
    if (!outfile)
    {
        cerr << "Error: Cannot open file " << argv[2] << "\n";
        return 1;
    }

    // Print paths
    for (int i = 0; i < num_pop; i++)
    {
        for (int j = 0; j < path_size[i]; j++)
            outfile << paths[i][j] << " ";
        outfile << "\n";
    }
    // Print drops
    for (int i = 0; i < num_pop; i++)
    {
        for (int j = 0; j < num_drops[i]; j++)
        {
            outfile << drops[i][j][0] << " "
                    << drops[i][j][1] << " "
                    << drops[i][j][2] << " ";
        }
        outfile << "\n";
    }
    outfile.close();

    return 0;
}
