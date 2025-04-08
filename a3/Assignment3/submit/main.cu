
#include <chrono>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <math.h>

#define MOD 1000000007

using std::cin;
using std::cout;
struct Edge {
    int src, dest, weight;
};

int main() {
    // Create a sample graph
    int V;
    cin >> V;
    int E;
    cin >> E;
    vector<Edge> edges;

    while (E--) {
        int u, v, wt;
        string s;
        cin >> u >> v >> wt;
        cin >> s;
    }
    
    // Answer should be calculated in Kernel. No operations should be performed here.
    // Only copy data to device, kernel call, copy data back to host, and print the answer.
    auto start = std::chrono::high_resolution_clock::now();
    // Kernel call(s) here

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    // Print only the total MST weight


    // cout << elapsed1.count() << " s\n";
    return 0;
}