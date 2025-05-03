#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <algorithm>
#include <climits>
#include <cmath>
#include <unordered_map>
#include <utility>

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

// Structure to evaluate shelter choices
struct ShelterEvaluation
{
    int shelterCity;
    double score;
    int distance;
    int capacity;

    bool operator<(const ShelterEvaluation &other) const
    {
        return score < other.score;
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
    unordered_map<int, int> shelterCapacity;              // city_id -> capacity
    unordered_map<int, pair<int, int>> populatedCityInfo; // city_id -> (prime_age, elderly)
    vector<int> populatedCities;                          // List of populated city IDs
    vector<int> shelterCities;                            // List of shelter city IDs

    // Shortest distance and path tracking
    vector<vector<int>> distances;             // distances[from][to] = shortest distance
    vector<vector<vector<int>>> shortestPaths; // shortestPaths[from][to] = path array

    // Road traversal time calculation
    double calculateTraversalTime(int numPeople, int roadCapacity, int roadLength)
    {
        // Speed is 5 km/h
        double timePerBatch = static_cast<double>(roadLength) / 5.0; // Time in hours
        int numBatches = ceil(static_cast<double>(numPeople) / roadCapacity);
        return timePerBatch * numBatches;
    }

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

    // Evaluate and score shelters based on capacity, distance, and potential evacuation time
    vector<ShelterEvaluation> evaluateShelters(int sourceCity, int peopleToEvacuate, bool forElderly = false)
    {
        vector<ShelterEvaluation> shelterScores;

        for (int shelter : shelterCities)
        {
            if (shelterCapacity[shelter] <= 0)
                continue; // Skip full shelters

            int dist = distances[sourceCity][shelter];

            // Skip if distance exceeds elderly limit and we're checking for elderly
            if (forElderly && dist > max_distance_elderly)
                continue;

            // Skip if no path exists
            if (dist == INT_MAX)
                continue;

            int capacity = shelterCapacity[shelter];

            // Calculate people we can save
            int peopleSaved = min(peopleToEvacuate, capacity);

            // Estimate evacuation time (simplified - assuming all roads have same capacity)
            // We'll use the average road capacity as a simplification
            double avgRoadCapacity = 0;
            int roadCount = 0;

            for (int city = 0; city < num_cities; city++)
            {
                for (const Road &road : adjacencyList[city])
                {
                    avgRoadCapacity += road.capacity;
                    roadCount++;
                }
            }

            avgRoadCapacity = roadCount > 0 ? avgRoadCapacity / roadCount : 100; // Default if no roads

            // Estimate time (simplification)
            double estimatedTime = calculateTraversalTime(peopleToEvacuate, avgRoadCapacity, dist);

            // Score formula prioritizes people saved but also considers time
            // We use 1.0 and 0.4 weights as in the formula
            // For simplicity: higher peopleSaved/dist ratio is better
            double score = peopleSaved / (1.0 + 0.1 * dist);

            shelterScores.push_back({shelter, score, dist, capacity});
        }

        // Sort by score (highest first)
        sort(shelterScores.begin(), shelterScores.end(), [](const ShelterEvaluation &a, const ShelterEvaluation &b)
             { return a.score > b.score; });

        return shelterScores;
    }

    // Check if the elderly can reach from current to next city
    bool canElderlyReach(int current, int next, int distanceSoFar)
    {
        for (const Road &road : adjacencyList[current])
        {
            if (road.to == next)
            {
                return (distanceSoFar + road.length) <= max_distance_elderly;
            }
        }
        return false;
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
        // For each populated city
        for (int i = 0; i < num_populated_cities; i++)
        {
            int sourceCity = populatedCities[i];
            int prime_age = populatedCityInfo[sourceCity].first;
            int elderly = populatedCityInfo[sourceCity].second;

            vector<int> evacPath;
            vector<vector<long long>> evacDrops;

            // Start with the source city
            evacPath.push_back(sourceCity);
            int currentCity = sourceCity;
            int distanceTraveled = 0;

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

            // Continue evacuation until all people are dropped
            while (prime_age > 0 || elderly > 0)
            {
                // First handle elderly (they have distance restrictions)
                int nextShelterForElderly = -1;
                vector<ShelterEvaluation> elderlyShelters;

                if (elderly > 0)
                {
                    elderlyShelters = evaluateShelters(currentCity, elderly, true);
                    if (!elderlyShelters.empty())
                    {
                        nextShelterForElderly = elderlyShelters[0].shelterCity;
                    }
                }

                // If no shelter for elderly within range, drop them at current city
                if (elderly > 0 && nextShelterForElderly == -1)
                {
                    evacDrops.push_back({(long long)currentCity, 0, (long long)elderly});
                    elderly = 0;
                }

                // Now handle prime-age people
                vector<ShelterEvaluation> primeShelters;
                int nextShelterForPrime = -1;

                if (prime_age > 0)
                {
                    primeShelters = evaluateShelters(currentCity, prime_age, false);
                    if (!primeShelters.empty())
                    {
                        nextShelterForPrime = primeShelters[0].shelterCity;
                    }
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
                vector<int> pathToShelter = shortestPaths[currentCity][targetShelter];

                // Skip first city if it's the current city
                int startIdx = (pathToShelter.size() > 0 && pathToShelter[0] == currentCity) ? 1 : 0;

                // Travel along the path to the shelter
                for (int j = startIdx; j < pathToShelter.size(); j++)
                {
                    int nextCity = pathToShelter[j];

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

                // If we've reached the shelter but still have people, continue to next iteration
                // to find another shelter
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

    ofstream outfile(argv[2]); // Read input file from command-line argument
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