#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <algorithm>
#include <climits>
#include <cmath>
#include <unordered_map>
#include <utility>
#include <set>

using namespace std;

// Structure to represent a road between cities
struct Road
{
    int to;
    int length;
    int capacity;
};

// Structure for Dijkstra's algorithm
struct CityDistance
{
    int city;
    int distance;

    bool operator>(const CityDistance &other) const
    {
        return distance > other.distance;
    }
};

// Class to handle the evacuation strategy
class EvacuationStrategy
{
private:
    int num_cities;
    int num_roads;
    int max_distance_elderly;
    vector<vector<Road>> adjacencyList;

    // Maps for shelters and populated cities
    unordered_map<int, int> originalShelterCapacity;      // Initial capacities (for reference)
    unordered_map<int, int> shelterCapacity;              // Current capacities (gets updated)
    unordered_map<int, pair<int, int>> populatedCityInfo; // city_id -> (prime_age, elderly)
    vector<int> populatedCities;                          // List of populated city IDs
    vector<int> shelterCities;                            // List of shelter city IDs

    // Shortest distance and path tracking
    vector<vector<int>> distances;             // distances[from][to] = shortest distance
    vector<vector<vector<int>>> shortestPaths; // shortestPaths[from][to] = path array

    // Function to find shortest paths using Dijkstra's algorithm
    void calculateShortestPaths()
    {
        // Initialize distances matrix
        distances.resize(num_cities, vector<int>(num_cities, INT_MAX));
        shortestPaths.resize(num_cities, vector<vector<int>>(num_cities));

        for (int source = 0; source < num_cities; source++)
        {
            // Distance to self is 0
            distances[source][source] = 0;
            shortestPaths[source][source] = {source};

            vector<bool> visited(num_cities, false);
            priority_queue<CityDistance, vector<CityDistance>, greater<CityDistance>> pq;

            pq.push({source, 0});

            while (!pq.empty())
            {
                int current = pq.top().city;
                int currentDist = pq.top().distance;
                pq.pop();

                if (visited[current])
                    continue;
                visited[current] = true;

                for (const Road &road : adjacencyList[current])
                {
                    int next = road.to;
                    int newDist = currentDist + road.length;

                    if (newDist < distances[source][next])
                    {
                        distances[source][next] = newDist;

                        // Update path
                        shortestPaths[source][next] = shortestPaths[source][current];
                        shortestPaths[source][next].push_back(next);

                        pq.push({next, newDist});
                    }
                }
            }
        }
    }

    // Verify that there's a direct road between consecutive cities in a path
    bool verifyPathConnectivity(const vector<int> &path)
    {
        for (int i = 0; i < path.size() - 1; i++)
        {
            int from = path[i];
            int to = path[i + 1];

            bool connected = false;
            for (const Road &road : adjacencyList[from])
            {
                if (road.to == to)
                {
                    connected = true;
                    break;
                }
            }

            if (!connected)
            {
                cout << "No direct road between cities " << from << " and " << to << " in path" << endl;
                return false;
            }
        }
        return true;
    }

    // Calculate evacuation time for a path considering batching
    double calculatePathEvacuationTime(const vector<int> &path, int numPeople)
    {
        double totalTime = 0;

        for (int i = 0; i < path.size() - 1; i++)
        {
            int from = path[i];
            int to = path[i + 1];

            // Find the road
            int roadCapacity = 0;
            int roadLength = 0;
            for (const Road &road : adjacencyList[from])
            {
                if (road.to == to)
                {
                    roadCapacity = road.capacity;
                    roadLength = road.length;
                    break;
                }
            }

            if (roadCapacity == 0)
            {
                // No direct road found (should not happen if path is valid)
                return numeric_limits<double>::max();
            }

            // Calculate traversal time
            double timePerBatch = static_cast<double>(roadLength) / 5.0; // Hours
            int numBatches = ceil(static_cast<double>(numPeople) / roadCapacity);
            totalTime += timePerBatch * numBatches * 60; // Convert to minutes
        }

        return totalTime;
    }

    // Find the closest shelter for evacuees
    int findClosestShelter(int sourceCity, int peopleCount, bool forElderly)
    {
        int closestShelter = -1;
        int shortestDistance = INT_MAX;

        for (int shelter : shelterCities)
        {
            // Skip if shelter is full
            if (shelterCapacity[shelter] <= 0)
                continue;

            int dist = distances[sourceCity][shelter];

            // Skip if no path exists
            if (dist == INT_MAX)
                continue;

            // For elderly, check distance limit
            if (forElderly && dist > max_distance_elderly)
                continue;

            // Check if path is valid (has direct connections)
            if (!verifyPathConnectivity(shortestPaths[sourceCity][shelter]))
                continue;

            // Update closest shelter
            if (dist < shortestDistance)
            {
                shortestDistance = dist;
                closestShelter = shelter;
            }
        }

        return closestShelter;
    }

    // Calculate time for people to move along a path
    double calculateMovementTime(const vector<int> &path, int numPeople)
    {
        double totalTime = 0;

        for (int i = 0; i < path.size() - 1; i++)
        {
            int from = path[i];
            int to = path[i + 1];

            // Find road capacity and length
            int capacity = 0;
            int length = 0;
            for (const Road &road : adjacencyList[from])
            {
                if (road.to == to)
                {
                    capacity = road.capacity;
                    length = road.length;
                    break;
                }
            }

            if (capacity == 0)
            {
                // This should not happen with valid paths
                cout << "Error: No direct road between " << from << " and " << to << endl;
                return numeric_limits<double>::max();
            }

            // Calculate time with batching
            double timePerBatch = (double)length / 5.0; // Hours (5 km/h)
            int numBatches = ceil((double)numPeople / capacity);
            totalTime += timePerBatch * numBatches * 60; // Minutes
        }

        return totalTime;
    }

    // Check if elderly can travel the given distance
    bool canElderlyTravel(int distance)
    {
        return distance <= max_distance_elderly;
    }

public:
    EvacuationStrategy(
        int num_cities,
        int num_roads,
        int *roads,
        int num_shelters,
        long long *shelter_city,
        long long *shelter_capacity,
        int num_populated_cities,
        long long *city,
        long long *pop,
        int max_distance_elderly) : num_cities(num_cities), num_roads(num_roads), max_distance_elderly(max_distance_elderly)
    {

        // Initialize adjacency list
        adjacencyList.resize(num_cities);

        // Add roads to adjacency list (bidirectional)
        for (int i = 0; i < num_roads; i++)
        {
            int u = roads[4 * i];
            int v = roads[4 * i + 1];
            int length = roads[4 * i + 2];
            int capacity = roads[4 * i + 3];

            adjacencyList[u].push_back({v, length, capacity});
            adjacencyList[v].push_back({u, length, capacity});
        }

        // Add shelters
        for (int i = 0; i < num_shelters; i++)
        {
            int city_id = shelter_city[i];
            int capacity = shelter_capacity[i];

            originalShelterCapacity[city_id] = capacity;
            shelterCapacity[city_id] = capacity;
            shelterCities.push_back(city_id);
        }

        // Add populated cities
        for (int i = 0; i < num_populated_cities; i++)
        {
            int city_id = city[i];
            int prime_age = pop[2 * i];
            int elderly = pop[2 * i + 1];

            populatedCityInfo[city_id] = {prime_age, elderly};
            populatedCities.push_back(city_id);
        }

        // Precompute shortest paths
        calculateShortestPaths();
    }

    // Generate evacuation paths for all populated cities
    void generatePaths(
        long long *path_size,
        long long **paths,
        long long *num_drops,
        long long ***drops,
        int num_populated_cities)
    {
        // Process each populated city
        for (int i = 0; i < num_populated_cities; i++)
        {
            int sourceCity = populatedCities[i];
            int prime_age = populatedCityInfo[sourceCity].first;
            int elderly = populatedCityInfo[sourceCity].second;

            vector<int> evacPath;
            vector<vector<long long>> evacDrops;

            // Start with the source city
            evacPath.push_back(sourceCity);

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

            // Remaining evacuees
            int remainingElderly = elderly;
            int remainingPrime = prime_age;
            int currentCity = sourceCity;

            // Process elderly evacuation first (they have distance restrictions)
            while (remainingElderly > 0)
            {
                // Find closest shelter for elderly
                int targetShelter = findClosestShelter(currentCity, remainingElderly, true);

                if (targetShelter == -1)
                {
                    // No accessible shelter found, drop elderly at current city
                    evacDrops.push_back({(long long)currentCity, 0, (long long)remainingElderly});
                    remainingElderly = 0;
                }
                else
                {
                    // Get path to shelter
                    vector<int> pathToShelter = shortestPaths[currentCity][targetShelter];

                    // Skip first node if it's the current city
                    int startIdx = (pathToShelter.size() > 0 && pathToShelter[0] == currentCity) ? 1 : 0;

                    // Check if elderly can reach shelter (verify total distance)
                    int totalDistance = 0;
                    bool canReach = true;

                    for (int j = 0; j < pathToShelter.size() - 1; j++)
                    {
                        int from = pathToShelter[j];
                        int to = pathToShelter[j + 1];

                        for (const Road &road : adjacencyList[from])
                        {
                            if (road.to == to)
                            {
                                totalDistance += road.length;
                                break;
                            }
                        }

                        if (totalDistance > max_distance_elderly)
                        {
                            canReach = false;
                            break;
                        }
                    }

                    if (!canReach)
                    {
                        // Elderly can't reach shelter, drop at current city
                        evacDrops.push_back({(long long)currentCity, 0, (long long)remainingElderly});
                        remainingElderly = 0;
                    }
                    else
                    {
                        // Move to shelter
                        for (int j = startIdx; j < pathToShelter.size(); j++)
                        {
                            int nextCity = pathToShelter[j];

                            // Add to evacuation path if not already there
                            if (evacPath.empty() || evacPath.back() != nextCity)
                            {
                                evacPath.push_back(nextCity);
                            }

                            // Update current city
                            currentCity = nextCity;

                            // If at shelter, drop elderly
                            if (currentCity == targetShelter)
                            {
                                int capacity = shelterCapacity[currentCity];
                                int elderlyToDrop = min(remainingElderly, capacity);

                                // Update counts
                                remainingElderly -= elderlyToDrop;
                                shelterCapacity[currentCity] -= elderlyToDrop;

                                // Record drop
                                if (elderlyToDrop > 0)
                                {
                                    evacDrops.push_back({(long long)currentCity, 0, (long long)elderlyToDrop});
                                }

                                break;
                            }
                        }
                    }
                }
            }

            // Process prime-age evacuation
            while (remainingPrime > 0)
            {
                // Find closest shelter
                int targetShelter = findClosestShelter(currentCity, remainingPrime, false);

                if (targetShelter == -1)
                {
                    // No accessible shelter found, drop at current city
                    evacDrops.push_back({(long long)currentCity, (long long)remainingPrime, 0});
                    remainingPrime = 0;
                }
                else
                {
                    // Get path to shelter
                    vector<int> pathToShelter = shortestPaths[currentCity][targetShelter];

                    // Skip first node if it's the current city
                    int startIdx = (pathToShelter.size() > 0 && pathToShelter[0] == currentCity) ? 1 : 0;

                    // Move to shelter
                    for (int j = startIdx; j < pathToShelter.size(); j++)
                    {
                        int nextCity = pathToShelter[j];

                        // Add to evacuation path if not already there
                        if (evacPath.empty() || evacPath.back() != nextCity)
                        {
                            evacPath.push_back(nextCity);
                        }

                        // Update current city
                        currentCity = nextCity;

                        // If at shelter, drop prime-age
                        if (currentCity == targetShelter)
                        {
                            int capacity = shelterCapacity[currentCity];
                            int primeToDrop = min(remainingPrime, capacity);

                            // Update counts
                            remainingPrime -= primeToDrop;
                            shelterCapacity[currentCity] -= primeToDrop;

                            // Record drop
                            if (primeToDrop > 0)
                            {
                                evacDrops.push_back({(long long)currentCity, (long long)primeToDrop, 0});
                            }

                            break;
                        }
                    }
                }
            }

            // Ensure path has no duplicates
            vector<int> uniquePath;
            for (int city : evacPath)
            {
                if (uniquePath.empty() || uniquePath.back() != city)
                {
                    uniquePath.push_back(city);
                }
            }

            // Double check path connectivity
            if (!verifyPathConnectivity(uniquePath))
            {
                cout << "Path connectivity issue detected for city " << sourceCity << endl;
                // If there's a connectivity issue, use only the source city
                uniquePath = {sourceCity};
                evacDrops = {{(long long)sourceCity, (long long)prime_age, (long long)elderly}};
            }

            // Double check shelter capacities
            for (const auto &drop : evacDrops)
            {
                int city_id = drop[0];
                int prime_dropped = drop[1];
                int elderly_dropped = drop[2];

                // Skip cities that aren't shelters
                if (originalShelterCapacity.find(city_id) == originalShelterCapacity.end())
                    continue;

                // Check if over capacity
                int totalDropped = prime_dropped + elderly_dropped;
                if (totalDropped > originalShelterCapacity[city_id])
                {
                    cout << "Error: Shelter at city " << city_id << " is over capacity" << endl;
                    cout << "Capacity: " << originalShelterCapacity[city_id] << " Dropped: " << totalDropped
                         << " (Prime: " << prime_dropped << ", Elderly: " << elderly_dropped << ")" << endl;
                }
            }

            // Set the path size
            path_size[i] = uniquePath.size();

            // Allocate and copy the path
            paths[i] = new long long[uniquePath.size()];
            for (int j = 0; j < uniquePath.size(); j++)
            {
                paths[i][j] = uniquePath[j];
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
};

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

    // set your answer to these variables
    long long *path_size = new long long[num_populated_cities];
    long long **paths = new long long *[num_populated_cities];
    long long *num_drops = new long long[num_populated_cities];
    long long ***drops = new long long **[num_populated_cities];

    // Create evacuation strategy
    EvacuationStrategy strategy(
        num_cities,
        num_roads,
        roads,
        num_shelters,
        shelter_city,
        shelter_capacity,
        num_populated_cities,
        city,
        pop,
        max_distance_elderly);

    // Generate paths and drops
    strategy.generatePaths(path_size, paths, num_drops, drops, num_populated_cities);

    //------------output-----------------

    ofstream outfile(argv[2]); // Write to output file from command-line argument
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