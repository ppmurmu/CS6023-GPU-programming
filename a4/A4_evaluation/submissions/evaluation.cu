

#include <iostream>
#include <fstream>
#include <string>
#include <cuda.h>
#include <bits/stdc++.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;
using namespace std;

#define MAX_NEIGHBORS 10000

typedef struct {
    int u, v;
    int length;
    int capacity;
} Edge;

typedef struct {
    int cityId;
    int capacity;
} Shelter;

typedef struct {
    int id;
    int primeAge;
    int elderly;
    bool isShelter;
    int shelterCapacity;
} City;

typedef struct {
    int numVertices;
    int numEdges;
    Edge* edges;

    int** adjList;
    int* adjSize;
} Graph;


vector<int> getNums(string line) {

    vector<int> res(0);
    string curr = "";
    for(int i = 0; i < line.size(); i++) {
        if(line[i] == ' ') {

            res.push_back(stoi(curr));
            curr = "";
        } else curr += line[i];
    }
    if(curr.size() > 0)
     res.push_back(stoi(curr));

    return res;
}


Graph* createGraph(int n, int numEdges) {
    Graph* g = (Graph*)malloc(sizeof(Graph));
    g->numVertices = n;
    g->numEdges = 0;

    g->edges = (Edge*)malloc(sizeof(Edge) * numEdges);

    g->adjList = (int**)malloc(sizeof(int*) * n);
    for (int i = 0; i < n; ++i) {
        g->adjList[i] = (int*)malloc(sizeof(int) * MAX_NEIGHBORS);  // Make it Dynamic
    }

    g->adjSize = (int*)calloc(n, sizeof(int));

    return g;
}

void addEdge(Graph* g, int u, int v, int length, int capacity) {
    int idx = g->numEdges;
    g->edges[idx] = (Edge){u, v, length, capacity};

    g->adjList[u][g->adjSize[u]++] = idx;
    g->adjList[v][g->adjSize[v]++] = idx;

    g->numEdges++;
}


Graph* copyGraphToGPU(Graph* hostGraph) {
    Graph* deviceGraph;

    // Allocate and copy edges
    Edge* deviceEdges;
    cudaMalloc(&deviceEdges, sizeof(Edge) * hostGraph->numEdges);
    cudaMemcpy(deviceEdges, hostGraph->edges, sizeof(Edge) * hostGraph->numEdges, cudaMemcpyHostToDevice);

    // Allocate and copy adjSize
    int* deviceAdjSize;
    cudaMalloc(&deviceAdjSize, sizeof(int) * hostGraph->numVertices);
    cudaMemcpy(deviceAdjSize, hostGraph->adjSize, sizeof(int) * hostGraph->numVertices, cudaMemcpyHostToDevice);

    // Allocate and copy adjList (2D as array of pointers)
    int** deviceAdjList;
    cudaMalloc(&deviceAdjList, sizeof(int*) * hostGraph->numVertices);

    for (int i = 0; i < hostGraph->numVertices; ++i) {
        int* deviceList;
        cudaMalloc(&deviceList, sizeof(int) * MAX_NEIGHBORS);
        cudaMemcpy(deviceList, hostGraph->adjList[i], sizeof(int) * MAX_NEIGHBORS, cudaMemcpyHostToDevice);
        cudaMemcpy(&deviceAdjList[i], &deviceList, sizeof(int*), cudaMemcpyHostToDevice);
    }

    // Allocate Graph struct itself
    cudaMalloc(&deviceGraph, sizeof(Graph));

    // Copy fields to device Graph struct
    cudaMemcpy(&deviceGraph->numVertices, &hostGraph->numVertices, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&deviceGraph->numEdges, &hostGraph->numEdges, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&deviceGraph->edges, &deviceEdges, sizeof(Edge*), cudaMemcpyHostToDevice);
    cudaMemcpy(&deviceGraph->adjList, &deviceAdjList, sizeof(int**), cudaMemcpyHostToDevice);
    cudaMemcpy(&deviceGraph->adjSize, &deviceAdjSize, sizeof(int*), cudaMemcpyHostToDevice);

    return deviceGraph;
}



//function used to flatten the path vector of vectors
void flattenVectorOfVectors(const vector<vector<int>>& input,
                            int*& flatArray,
                            int*& offsetArray,
                            int& totalSize,
                            int& numVectors) {
    numVectors = input.size();
    totalSize = 0;

    // First pass to compute total size
    for (const auto& vec : input) {
        totalSize += vec.size();
    }

    // Allocate raw arrays
    flatArray = new int[totalSize];
    offsetArray = new int[numVectors + 1]; // +1 for end offset

    int offset = 0;
    offsetArray[0] = 0; // first offset is always 0
    int idx = 0;

    for (const auto& vec : input) {
        memcpy(flatArray + offset, vec.data(), vec.size() * sizeof(int));
        offset += vec.size();
        idx++;
        offsetArray[idx] = offset; // mark end of current vector
    }
}

// Kernel parameters:
//  - numCities        : number of populated cities (threads you launch)
//  - primePop          : [num_cities] number of prime‑age evacuees
//  - pathOffset       : [num_cities+1] offsets into path_data
//  - pathData         : flattened array of all city‑paths
//  - isShelter         : [num_vertices] boolean (0/1) marker for shelters
//  - shelterCapacity  : [num_vertices] remaining capacity per shelter
//  - maxSteps         : maximum path length (in edges) across all cities

__global__ void wavefrontEvacKernel(
    Graph*       dGraph,
    int          numCities,
    int*         dPopCityIds,
    City*        dCities,
    int*         pathOffset,
    int*         pathData,
    int          elderlyMax,
    int*         dEdgeWinner,
    long long*   dEdgeBestKey,
    long long*   dEdgeNextFree,
    int*         dflatDropCityInfo,
    int*         dflatDroppedPrimeAgeInfo,
    int*         dflatDroppedElderlyInfo,
    int*         doffsetDroppedInfoArray,
    int*         dshelterDrops,
    int*         valid,
    long long*   dTimeTaken,
    int* done
) {
    grid_group grid = this_grid();
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numCities) return;

    int local_done = 0;




    // Load evac info
    int cityId = dPopCityIds[t];
    int evacRem = dCities[cityId].primeAge;
    int evacElderly = dCities[cityId].elderly;

    // Unpack path indices
    int startIdx = pathOffset[t];
    int endIdx   = pathOffset[t+1];
    int pathLen  = endIdx - startIdx;

    // Per-thread travel state
    int  stepIndex           = 0;   // index in pathData
    int  holdingEdge         = -1;  // current edge index or -1 if waiting
    long long remainingKm    = 0;   // km left for current batch
    int batchPeopleLeft      = 0;   // people left on this edge batch
    long long currentTime    = 0;   // in minutes
    long long arrivalTime    = 0;

    int dropsStep = 0;
    int dropsStartIdx = doffsetDroppedInfoArray[t];
    int dropsEndIdx = doffsetDroppedInfoArray[t + 1];
    int dropsLen = dropsEndIdx - dropsStartIdx;



    int distanceTravelled = 0;

    //people dropped at the start node
    int vNode =  pathData[startIdx + stepIndex];
        if(dropsStep < dropsLen && vNode == dflatDropCityInfo[dropsStartIdx + dropsStep]) {

        atomicAdd(&dshelterDrops[vNode], dflatDroppedPrimeAgeInfo[dropsStartIdx + dropsStep]);
        if(distanceTravelled <= elderlyMax) {
            atomicAdd(&dshelterDrops[vNode], dflatDroppedElderlyInfo[dropsStartIdx + dropsStep]);
        }
        evacRem -= dflatDroppedPrimeAgeInfo[dropsStartIdx + dropsStep];
        evacElderly -= dflatDroppedElderlyInfo[dropsStartIdx + dropsStep];
        dropsStep++;

        if(evacRem + evacElderly <= 0) {
          stepIndex = pathLen;
          local_done = 1;
          atomicAdd(done, 1);
        }
    }

    grid.sync();



    // Attempt to acquire the first edge (no batch state yet)
    int edge = -1;
    if (pathLen > 1) {
        int u = pathData[startIdx], v = pathData[startIdx+1];
        int deg = dGraph->adjSize[u];
        for (int i = 0; i < deg; ++i) {
            int cand = dGraph->adjList[u][i];
            Edge e    = dGraph->edges[cand];
            if ((e.u==u && e.v==v)||(e.u==v && e.v==u)) { edge = cand; break; }
        }

        if(edge == -1) {
          *valid = 0;
        }

        // contend for it
        long long tick   = currentTime;
        long long MAXPOP = 10000 + 5;
        long long key    = tick * (MAXPOP+1) + (MAXPOP - evacRem);
        long long prev   = atomicMin(&dEdgeBestKey[edge], key);
        if (key < prev) atomicExch(&dEdgeWinner[edge], t);



    }

    grid.sync();

    if(pathLen > 1) {

      if (dEdgeWinner[edge] == t) {
            holdingEdge      = edge;
            remainingKm      = dGraph->edges[edge].length;
            batchPeopleLeft  = evacRem;  // entire remaining population enters
        }
    }



    // Main per-km loop
    while (*done < numCities ) {
        

        if (holdingEdge < 0 && stepIndex < pathLen - 1) {
            // Waiting at node
            currentTime += 12;

            // Try to acquire next edge
            int u = pathData[startIdx + stepIndex];
            int v = pathData[startIdx + stepIndex + 1];
            int deg = dGraph->adjSize[u], edge = -1;
            for (int i = 0; i < deg; ++i) {
                int cand = dGraph->adjList[u][i];
                Edge e    = dGraph->edges[cand];

                if ((e.u==u && e.v==v)||(e.u==v && e.v==u)) { edge = cand; break; }
            }





            if(edge == -1) {
              *valid = 0;
              stepIndex = pathLen;
              local_done = 1;
              atomicAdd(done, 1);

              
            } else {
                long long tick   = arrivalTime;
                long long MAXPOP = 10000;
                long long key    = (tick * (MAXPOP)) + (MAXPOP - evacRem);
                key = (key * 10000) + cityId;

                


                long long prev   = atomicMin(&dEdgeBestKey[edge], key);
                if (key < prev) atomicExch(&dEdgeWinner[edge], t);

                
                
            }

            
        } else {
            // Traveling 1 km for current batch

            currentTime += 12;
            remainingKm--;
            distanceTravelled++;

            // If this batch completes the edge
            if (remainingKm <= 0) {

                int edge = holdingEdge;

                // Decrement batch population
                int cap = dGraph->edges[edge].capacity;
                batchPeopleLeft -= cap;

                

                // If more people remain for this edge, reset distance but keep holding
                if (batchPeopleLeft > 0) {
                    remainingKm     = dGraph->edges[edge].length;
                } else {
                    arrivalTime = currentTime;

                    // Last batch finished: mark next-free and release
                    if (dEdgeWinner[edge] == t) {

                        dEdgeWinner[edge]  = -1;
                        dEdgeBestKey[edge] = (long long) INT64_MAX;
                    }

                    // Arrive at node: drop at shelter if any
                    int vNode = pathData[startIdx + stepIndex + 1];

                    if ((evacRem + (distanceTravelled <= elderlyMax ? evacElderly : 0)) > 0 && vNode == dflatDropCityInfo[dropsStartIdx + dropsStep]) {
                        // store the dropped people on each shelter

                        if(dropsStep < dropsLen && vNode == dflatDropCityInfo[dropsStartIdx + dropsStep]) {


                            atomicAdd(&dshelterDrops[vNode], dflatDroppedPrimeAgeInfo[dropsStartIdx + dropsStep]);
                            if(distanceTravelled <= elderlyMax) {
                                atomicAdd(&dshelterDrops[vNode], dflatDroppedElderlyInfo[dropsStartIdx + dropsStep]);
                            }
                            evacRem -= dflatDroppedPrimeAgeInfo[dropsStartIdx + dropsStep];
                            evacElderly -= dflatDroppedElderlyInfo[dropsStartIdx + dropsStep];

                            if((evacRem + (distanceTravelled <= elderlyMax ? evacElderly : 0)) <= 0) {


                              stepIndex = pathLen;
                              atomicAdd(done, 1);
                              local_done = 1;

                                
                            }

                            dropsStep++;
                        }
                    }

                    stepIndex++;
                    holdingEdge = -1;
                }
            }
        }






        // Unconditional sync so all threads stay in lockstep
        grid.sync();




        int edge = -1;
        // If we just released and still have more path, acquire next edge & init batch
        if (holdingEdge < 0 && stepIndex < pathLen - 1) {
            int u = pathData[startIdx + stepIndex];
            int v = pathData[startIdx + stepIndex + 1];
            int deg = dGraph->adjSize[u];
            for (int i = 0; i < deg; ++i) {
                int cand = dGraph->adjList[u][i];
                Edge e    = dGraph->edges[cand];
                if ((e.u==u && e.v==v)||(e.u==v && e.v==u)) { edge = cand; break; }
            }

            if(edge == -1) {
              *valid = 0;
              stepIndex = pathLen;
              local_done = 1;
              atomicAdd(done, 1);
            } else {
                long long tick   = arrivalTime;
                long long MAXPOP = 10000;
                long long key    = (tick * (MAXPOP)) + (MAXPOP - evacRem);
                key = (key * 10000) + cityId;


                long long prev   = atomicMin(&dEdgeBestKey[edge], key);
                if (key < prev) atomicExch(&dEdgeWinner[edge], t);
            }
        }

        grid.sync();

        if(holdingEdge < 0 && stepIndex < pathLen - 1) {
          if (dEdgeWinner[edge] == t) {


                holdingEdge      = edge;
                remainingKm      = dGraph->edges[edge].length;
                batchPeopleLeft  = evacRem;  // entire remaining population enters
            }
        }

        grid.sync();
    }

    if(evacRem > 0) *valid = 0;
    if(evacElderly > 0) *valid = 0;

    // Final report
    atomicMax(dTimeTaken, currentTime);
    //printf("Thread %d (city %d): total time = %lld minutes\n", t, cityId, currentTime);
    //printf("Evacs Remaining: %d %d %d\n", cityId, evacRem, evacElderly);
}




int main(int argc, char *argv[]) {
    ifstream inputFile(argv[1]);
    ifstream outputFile(argv[2]);


    //=============================================FILE OPENING SCRAP CODE======================================//

    if (!inputFile.is_open()) {
        cerr << "Error: Could not open input/input.txt\n";
        return 1;
    }

    if (!outputFile.is_open()) {
        cerr << "Error: Could not open output/output.txt\n";
        return 1;
    }

    //==========================================================================================================//

    //============================================= READING INPUT FILE =========================================//

    string line;

    int numCities = 0;
    inputFile >> numCities;

    int numRoads = 0;
    inputFile >> numRoads;

    int* isPopulated = new int[numCities + 10]();
    int* isShelter = new int[numCities + 10]();

    //create graph and add edges
    Graph* g = createGraph(numCities, numRoads);

    for(int i = 0; i < numRoads; i++) {
        int u = 0, v = 0, length = 0, capacity = 0;
        inputFile >> u >> v >> length >> capacity;
        addEdge(g, u, v, length, capacity);
    }


    City* cityInformation = (City*)malloc(sizeof(City) * numCities);

    //create shelters and store shelter information
    int numShelters;
    inputFile >> numShelters;
    Shelter* shelters = (Shelter*)malloc(sizeof(Shelter) * numShelters);

    for(int i = 0; i < numShelters; i++) {

        int city = 0, capacity = 0;
        inputFile >> city >> capacity;


        isShelter[city] = 1;
        shelters[i].cityId = city;
        shelters[i].capacity = capacity;


        cityInformation[city].id = city;
        cityInformation[city].isShelter = 1;
        cityInformation[city].shelterCapacity = capacity;
    }

    int numPopulatedCities = 0;
    inputFile >> numPopulatedCities;

    int* primePopIds = new int[numPopulatedCities + 10]();

    for(int i = 0; i < numPopulatedCities; i++) {
        int city = 0, primeAge = 0, elderly = 0;
        inputFile >> city >> primeAge >> elderly;

        primePopIds[i] = city;
        isPopulated[city] = 1;
        cityInformation[city].id = city;
        cityInformation[city].primeAge = primeAge;
        cityInformation[city].elderly = elderly;


    }

    int elderlyMax;
    inputFile >> elderlyMax;

    //==========================================================================================================//

    //============================================= READING OUTPUT FILE =========================================//
    vector<vector<int>> paths(0, vector<int>());

    for(int i = 0; i < numPopulatedCities; i++) {
        string line;


        getline(outputFile, line);
        vector<int> steps = getNums(line);

        paths.push_back(steps);
    }


    vector<vector<int>> dropCityInfo;
    vector<vector<int>> droppedPrimeAgeInfo;
    vector<vector<int>> droppedElderlyInfo;

    for(int i = 0; i < numPopulatedCities; i++) {
        string line;
        getline(outputFile, line);
        vector<int> allDrops = getNums(line);

        int interLoop = allDrops.size()/3;

        vector<int> dropCity(interLoop, 0);
        vector<int> droppedPrimeAge(interLoop, 0);
        vector<int> droppedElderly(interLoop, 0);


        for(int j = 0; j < interLoop; j++) {
          dropCity[j] = allDrops[(3 * j)];
          droppedPrimeAge[j] = allDrops[(3 * j) + 1];
          droppedElderly[j] = allDrops[(3 * j) + 2];
        }

        dropCityInfo.push_back(dropCity);
        droppedPrimeAgeInfo.push_back(droppedPrimeAge);
        droppedElderlyInfo.push_back(droppedElderly);
    }

    int* flatArray = nullptr;
    int* offsetArray = nullptr;
    int totalSize = 0;
    int numVectors = 0;


    flattenVectorOfVectors(paths, flatArray, offsetArray, totalSize, numVectors);

    int* flatDropCityInfo = nullptr;
    int* flatDroppedPrimeAgeInfo = nullptr;
    int* flatDroppedElderlyInfo = nullptr;
    int* offsetDroppedInfoArray = nullptr;
    int dropCityInfoTotalSize = 0;
    int dropCityInfoNumvectors = 0;

    flattenVectorOfVectors(dropCityInfo, flatDropCityInfo, offsetDroppedInfoArray, dropCityInfoTotalSize, dropCityInfoNumvectors);
    flattenVectorOfVectors(droppedPrimeAgeInfo, flatDroppedPrimeAgeInfo, offsetDroppedInfoArray, dropCityInfoTotalSize, dropCityInfoNumvectors);
    flattenVectorOfVectors(droppedElderlyInfo, flatDroppedElderlyInfo, offsetDroppedInfoArray, dropCityInfoTotalSize, dropCityInfoNumvectors);


    //=========================================================================================================//



    //============================================= KERNEL LAUNCH =========================================//

    Graph* dGraph = copyGraphToGPU(g);

    City* dCities;
    cudaMalloc(&dCities, sizeof(City) * numCities);
    cudaMemcpy(dCities, cityInformation, sizeof(City) * numCities, cudaMemcpyHostToDevice);

    int* dPopCityIds;
    cudaMalloc(&dPopCityIds, sizeof(int) * numPopulatedCities);
    cudaMemcpy(dPopCityIds, primePopIds, sizeof(int) * numPopulatedCities, cudaMemcpyHostToDevice);

    int* dPathOffset;
    cudaMalloc(&dPathOffset, sizeof(int) * (numPopulatedCities + 1));
    cudaMemcpy(dPathOffset, offsetArray, sizeof(int) * (numPopulatedCities + 1), cudaMemcpyHostToDevice);

    int* dPathData;
    cudaMalloc(&dPathData, sizeof(int) * totalSize);
    cudaMemcpy(dPathData, flatArray, sizeof(int) * totalSize, cudaMemcpyHostToDevice);

    int* dEdgeWinner;
    long long* dEdgeMaxPop;
    long long* dEdgeNextFree;
    cudaMalloc(&dEdgeWinner, sizeof(int) * numRoads);
    cudaMemset(dEdgeWinner, -1, sizeof(int) * numRoads);
    cudaMalloc(&dEdgeMaxPop, sizeof(long long) * numRoads);
    cudaMemset(dEdgeMaxPop, 1e9 + 7, sizeof(long long) * numRoads);
    cudaMalloc(&dEdgeNextFree, sizeof(long long) * numRoads);
    cudaMemset(dEdgeNextFree, 0, sizeof(long long) * numRoads);

    // initialize both arrays to -1





    int* dflatDropCityInfo;
    int* dflatDroppedPrimeAgeInfo;
    int* dflatDroppedElderlyInfo;
    int* doffsetDroppedInfoArray;
    int* dshelterDrops;

    cudaMalloc(&dflatDropCityInfo, sizeof(int) * dropCityInfoTotalSize);
    cudaMemcpy(dflatDropCityInfo, flatDropCityInfo, sizeof(int) * dropCityInfoTotalSize, cudaMemcpyHostToDevice);
    cudaMalloc(&dflatDroppedPrimeAgeInfo, sizeof(int) * dropCityInfoTotalSize);
    cudaMemcpy(dflatDroppedPrimeAgeInfo, flatDroppedPrimeAgeInfo, sizeof(int) * dropCityInfoTotalSize, cudaMemcpyHostToDevice);
    cudaMalloc(&dflatDroppedElderlyInfo, sizeof(int) * dropCityInfoTotalSize);
    cudaMemcpy(dflatDroppedElderlyInfo, flatDroppedElderlyInfo, sizeof(int) * dropCityInfoTotalSize, cudaMemcpyHostToDevice);
    cudaMalloc(&dshelterDrops, sizeof(int)* numCities);
    cudaMemset(dshelterDrops, 0, sizeof(int)* numCities);

    cudaMalloc(&doffsetDroppedInfoArray, sizeof(int) * (numPopulatedCities + 1));
    cudaMemcpy(doffsetDroppedInfoArray, offsetDroppedInfoArray, sizeof(int) * (numPopulatedCities + 1), cudaMemcpyHostToDevice);


    dim3 blockDim(128);
    dim3 gridDim((numPopulatedCities + blockDim.x - 1) / blockDim.x);












    int valid = 1;
    int* dValid;
    cudaMalloc(&dValid, sizeof(int));
    cudaMemcpy(dValid, &valid, sizeof(int), cudaMemcpyHostToDevice);
    long long timeTaken = 0;
    long long* dTimeTaken;
    cudaMalloc(&dTimeTaken, sizeof(long long));
    cudaMemset(&dTimeTaken, 0, sizeof(long long));

    int* done;
    cudaMalloc(&done, sizeof(int));
    cudaMemset(&done, 0, sizeof(int));


    void* kernelArgs[] = {
        &dGraph,
        &numPopulatedCities,
        &dPopCityIds,
        &dCities,
        &dPathOffset,
        &dPathData,
        &elderlyMax,
        &dEdgeWinner,
        &dEdgeMaxPop,
        &dEdgeNextFree,
        &dflatDropCityInfo,
        &dflatDroppedPrimeAgeInfo,
        &dflatDroppedElderlyInfo,
        &doffsetDroppedInfoArray,
        &dshelterDrops,
        &dValid,
        &dTimeTaken,
        &done
    };

    // Check cooperative launch support
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (!deviceProp.cooperativeLaunch) {
        cerr << "ERROR: Device does not support cooperative launch!" << endl;
        return 0;
    }

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)wavefrontEvacKernel,
        gridDim,
        blockDim,
        kernelArgs
    );

    if (err != cudaSuccess) {
        cerr << "Kernel launch failed: " << cudaGetErrorString(err) << endl;
        return 0;
    }

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    cudaMemcpy(&valid, dValid, sizeof(int), cudaMemcpyDeviceToHost);


    int *shelterDrops = (int*) malloc(sizeof(int) * numCities);
    memset(shelterDrops, 0, sizeof(int) * numCities);
    cudaMemcpy(shelterDrops, dshelterDrops, sizeof(int)* numCities, cudaMemcpyDeviceToHost);
    cudaMemcpy(&timeTaken, dTimeTaken, sizeof(long long), cudaMemcpyDeviceToHost);





    long long saved = 0;
    for(int i = 0; i < numCities; i++) {
      if(cityInformation[i].isShelter && shelterDrops[i] > cityInformation[i].shelterCapacity) {

        int extra = shelterDrops[i] - cityInformation[i].shelterCapacity;
        shelterDrops[i] = cityInformation[i].shelterCapacity;
        shelterDrops[i] -= extra;

      }

      if(cityInformation[i].isShelter) {

        saved += shelterDrops[i];
      }
    }


    if(valid == 1) {
        printf("%lld %lld\n", saved, timeTaken);
    } else {
        printf("%d %d\n", -1, -1);
    }





    //=========================================================================================================//





    //============================================= GRAPH CHECK ================================================//

    // cout << "\nAdjacency List:\n";
    // for (int i = 0; i < numCities; ++i) {
    //     cout << "City " << i << ": ";
    //     for (int j = 0; j < g->adjSize[i]; ++j) {
    //         int edge_idx = g->adjList[i][j];
    //         Edge e = g->edges[edge_idx];
    //         std::cout << "(" << e.u << " <-> " << e.v
    //                   << ", len=" << e.length << ", cap=" << e.capacity << ") ";
    //     }
    //     std::cout << "\n";
    // }

    // for(int i = 0; i < totalSize; i++) {
    //   cout << flatArray[i] << " ";
    // }
    // cout << endl;

    // for(int i = 0; i <= numPopulatedCities; i++) {
    //     cout << offsetArray[i] << " ";
    // }
    // cout << endl;



    // for(int i = 0; i < dropCityInfoTotalSize; i++) {
    //     cout << flatDropCityInfo[i] << " ";
    // }
    // cout << endl;



    // for(int i = 0; i < dropCityInfoTotalSize; i++) {
    //     cout << flatDroppedPrimeAgeInfo[i] << " ";
    // }
    // cout << endl;

    // for(int i = 0; i < dropCityInfoTotalSize; i++) {
    //     cout << flatDroppedElderlyInfo[i] << " ";
    // }
    // cout << endl;

    // for(int i = 0; i <= numPopulatedCities + 1; i++) {
    //     cout << offsetDroppedInfoArray[i] << " ";
    // }
    // cout << endl;


    inputFile.close();
    outputFile.close();

    return 0;
}