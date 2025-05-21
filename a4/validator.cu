// %%writefile validator.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <climits>
#include <sstream>
#include <string>

using namespace std;

// Structure to represent a road
struct Road
{
    int src;
    int dest;
    int length;
    int capacity;
};

// Structure to represent a city
struct City
{
    int id;
    int prime;
    int elderly;
    bool isPopulated;
    bool isShelter;
    int capacity;        // Shelter capacity, 0 if not a shelter
    int initialCapacity; // For validation
};

// Structure to represent a drop
struct Drop
{
    int city;
    int prime;
    int elderly;
};

// Hash function for pair
struct pair_hash
{
    template <class T1, class T2>
    size_t operator()(const pair<T1, T2> &p) const
    {
        auto h1 = hash<T1>{}(p.first);
        auto h2 = hash<T2>{}(p.second);
        return h1 ^ h2;
    }
};

// Class to validate evacuation planning
class EvacuationValidator
{
private:
    int num_cities;
    int num_roads;
    int num_shelters;
    int num_populated_cities;
    int max_distance_elderly;

    vector<Road> roads;
    vector<City> cities;
    vector<int> shelterCities;
    vector<int> populatedCities;

    // Maps for quick look-up
    unordered_map<pair<int, int>, Road, pair_hash> roadMap;

    // Input paths and drops
    vector<vector<int>> inputPaths;
    vector<vector<Drop>> inputDrops;

    // Stats
    int totalPopulation = 0;
    int totalElderly = 0;
    int totalPrime = 0;
    int totalShelterCap = 0;
    int savedElderly = 0;
    int savedPrime = 0;
    int totalSaved = 0;
    int droppedElderly = 0;
    int droppedPrime = 0;
    int totalDropped = 0;
    float simulationTime = 0.0f;

    // Validation flags
    bool pathsValid = true;
    bool dropsValid = true;
    bool elderlyDistanceValid = true;

public:
    EvacuationValidator() {}

    // Read input file
    bool readInputFile(const string &filename)
    {
        ifstream infile(filename);
        if (!infile)
        {
            cerr << "Error: Cannot open input file " << filename << "\n";
            return false;
        }

        infile >> num_cities;
        infile >> num_roads;

        // Read roads
        roads.resize(num_roads);
        for (int i = 0; i < num_roads; i++)
        {
            infile >> roads[i].src >> roads[i].dest >> roads[i].length >> roads[i].capacity;
            // Add to map for quick lookup
            roadMap[{roads[i].src, roads[i].dest}] = roads[i];
            roadMap[{roads[i].dest, roads[i].src}] = roads[i]; // Bidirectional
        }

        // Read shelters
        infile >> num_shelters;
        cities.resize(num_cities);
        for (int i = 0; i < num_cities; i++)
        {
            cities[i].id = i;
            cities[i].prime = 0;
            cities[i].elderly = 0;
            cities[i].isPopulated = false;
            cities[i].isShelter = false;
            cities[i].capacity = 0;
            cities[i].initialCapacity = 0;
        }

        for (int i = 0; i < num_shelters; i++)
        {
            int cityId, capacity;
            infile >> cityId >> capacity;
            cities[cityId].isShelter = true;
            cities[cityId].capacity = capacity;
            cities[cityId].initialCapacity = capacity;
            shelterCities.push_back(cityId);
            totalShelterCap += capacity;
        }

        // Read populated cities
        infile >> num_populated_cities;
        populatedCities.resize(num_populated_cities);
        for (int i = 0; i < num_populated_cities; i++)
        {
            int cityId, prime, elderly;
            infile >> cityId >> prime >> elderly;
            cities[cityId].isPopulated = true;
            cities[cityId].prime = prime;
            cities[cityId].elderly = elderly;
            populatedCities[i] = cityId;

            totalPrime += prime;
            totalElderly += elderly;
            totalPopulation += prime + elderly;
        }

        // Read max distance for elderly
        infile >> max_distance_elderly;

        infile.close();
        return true;
    }

    // Read paths and drops from output
    bool readOutputData(istream &in)
    {
        inputPaths.resize(num_populated_cities);
        inputDrops.resize(num_populated_cities);

        string line;

        // Read paths
        for (int i = 0; i < num_populated_cities; i++)
        {
            if (!getline(in, line))
            {
                cerr << "Error: Not enough path lines in output\n";
                return false;
            }

            stringstream ss(line);
            int city;
            while (ss >> city)
            {
                inputPaths[i].push_back(city);
            }

            // Validate path starts from populated city
            if (!inputPaths[i].empty() && inputPaths[i][0] != populatedCities[i])
            {
                cerr << "Error: Path " << i << " doesn't start from the correct populated city\n";
                pathsValid = false;
            }
        }

        // Read drops
        for (int i = 0; i < num_populated_cities; i++)
        {
            if (!getline(in, line))
            {
                cerr << "Error: Not enough drop lines in output\n";
                return false;
            }

            stringstream ss(line);
            int city, prime, elderly;
            while (ss >> city >> prime >> elderly)
            {
                inputDrops[i].push_back({city, prime, elderly});
            }
        }

        return true;
    }

    // Validate connected paths
    bool validatePaths()
    {
        for (int i = 0; i < num_populated_cities; i++)
        {
            const vector<int> &path = inputPaths[i];

            for (size_t j = 0; j < path.size() - 1; j++)
            {
                int u = path[j];
                int v = path[j + 1];

                // Check if there's a road between u and v
                if (roadMap.find({u, v}) == roadMap.end())
                {
                    cerr << "Error: No direct road between cities " << u << " and " << v
                         << " in path " << i << "\n";
                    pathsValid = false;
                }
            }
        }

        return pathsValid;
    }

    // Validate drops with updated penalty calculation
    bool validateDrops()
    {
        vector<int> cityPrimeEvacuated(num_cities, 0);
        vector<int> cityElderlyEvacuated(num_cities, 0);

        for (int i = 0; i < num_populated_cities; i++)
        {
            int originCity = populatedCities[i];
            int totalPrime = cities[originCity].prime;
            int totalElderly = cities[originCity].elderly;
            int droppedPrime = 0;
            int droppedElderly = 0;

            for (const Drop &drop : inputDrops[i])
            {
                // Check if drop city is in the path
                bool cityInPath = false;
                for (int pathCity : inputPaths[i])
                {
                    if (pathCity == drop.city)
                    {
                        cityInPath = true;
                        break;
                    }
                }

                if (!cityInPath)
                {
                    cerr << "Error: Drop at city " << drop.city
                         << " which is not in the path for populated city " << originCity << "\n";
                    dropsValid = false;
                }

                droppedPrime += drop.prime;
                droppedElderly += drop.elderly;

                // Update city drops for capacity validation
                cityPrimeEvacuated[drop.city] += drop.prime;
                cityElderlyEvacuated[drop.city] += drop.elderly;
            }

            // Check if all people are accounted for
            if (droppedPrime != totalPrime || droppedElderly != totalElderly)
            {
                cerr << "Error: Not all people from city " << originCity << " are accounted for in drops\n";
                cerr << "  Expected: " << totalPrime << " prime, " << totalElderly << " elderly\n";
                cerr << "  Dropped: " << droppedPrime << " prime, " << droppedElderly << " elderly\n";
                dropsValid = false;
            }
        }

        // Reset saved and dropped counts
        savedPrime = 0;
        savedElderly = 0;
        droppedPrime = 0;
        droppedElderly = 0;

        // Validate shelter capacities and apply penalty
        for (int city = 0; city < num_cities; city++)
        {
            if (cities[city].isShelter)
            {
                int capacity = cities[city].initialCapacity;
                int totalDropped = cityPrimeEvacuated[city] + cityElderlyEvacuated[city];

                // Calculate excess and apply penalty according to PDF rules
                int excess = max(0, totalDropped - capacity);

                // Penalty is same as excess, but capped by capacity
                int penalty = min(excess, capacity);

                // Actual saved = totalDropped - excess - penalty
                int actualSaved = totalDropped - excess - penalty;

                // Split saved between prime and elderly proportionally
                if (totalDropped > 0)
                {
                    float primeProportion = (float)cityPrimeEvacuated[city] / totalDropped;
                    float elderlyProportion = (float)cityElderlyEvacuated[city] / totalDropped;

                    int savedPrimeHere = round(actualSaved * primeProportion);
                    int savedElderlyHere = actualSaved - savedPrimeHere; // Ensure total adds up exactly

                    savedPrime += savedPrimeHere;
                    savedElderly += savedElderlyHere;

                    // Update dropped counts (those not saved)
                    droppedPrime += cityPrimeEvacuated[city] - savedPrimeHere;
                    droppedElderly += cityElderlyEvacuated[city] - savedElderlyHere;
                }

                // For validation purposes, log if shelter is over capacity, but don't mark as error
                if (totalDropped > capacity)
                {
                    cerr << "Note: Shelter at city " << city << " is over capacity\n";
                    cerr << "  Capacity: " << capacity << "\n";
                    cerr << "  Dropped: " << totalDropped << " (Prime: " << cityPrimeEvacuated[city]
                         << ", Elderly: " << cityElderlyEvacuated[city] << ")\n";
                    cerr << "  Excess: " << excess << ", Penalty: " << penalty << "\n";
                    cerr << "  Saved: " << actualSaved << " people\n";
                }
            }
            else
            {
                // For non-shelter cities, all dropped people count as "dropped" (not saved)
                droppedPrime += cityPrimeEvacuated[city];
                droppedElderly += cityElderlyEvacuated[city];
            }
        }

        totalSaved = savedPrime + savedElderly;
        totalDropped = droppedPrime + droppedElderly;

        return dropsValid;
    }

    // Validate elderly distance constraints
    bool validateElderlyDistance()
    {
        for (int i = 0; i < num_populated_cities; i++)
        {
            int originCity = populatedCities[i];
            const vector<int> &path = inputPaths[i];

            // FIXED: Calculate distance for each position in the path
            vector<int> pathPositionDistance(path.size(), 0);

            // Calculate cumulative distance along the path
            for (size_t j = 1; j < path.size(); j++)
            {
                int u = path[j - 1];
                int v = path[j];

                if (roadMap.find({u, v}) != roadMap.end())
                {
                    int roadLength = roadMap[{u, v}].length;
                    pathPositionDistance[j] = pathPositionDistance[j - 1] + roadLength;
                }
            }

            // Match drops to path positions and check distances
            size_t dropIndex = 0;
            for (size_t pathIdx = 0; pathIdx < path.size() && dropIndex < inputDrops[i].size(); pathIdx++)
            {
                int currentCity = path[pathIdx];

                // Check if this position matches a drop city
                while (dropIndex < inputDrops[i].size() && inputDrops[i][dropIndex].city == currentCity)
                {
                    const Drop &drop = inputDrops[i][dropIndex];

                    if (drop.elderly > 0)
                    {
                        int dropDistance = pathPositionDistance[pathIdx];
                        if (dropDistance > max_distance_elderly)
                        {
                            cerr << "Error: Elderly from city " << originCity
                                 << " dropped at city " << drop.city
                                 << " which is " << dropDistance
                                 << " distance away (max allowed: " << max_distance_elderly << ")\n";
                            elderlyDistanceValid = false;
                        }
                    }
                    dropIndex++;
                }
            }
        }

        return elderlyDistanceValid;
    }

    // Calculate travel time for a batch of people
    float calculateTravelTime(int roadLength, int roadCapacity, int people)
    {
        float timePerBatch = roadLength / 5.0f; // hours
        int batches = ceil((float)people / roadCapacity);
        return timePerBatch * batches;
    }

    // Simulate evacuation to calculate total time
    float simulateEvacuation()
    {
        // Track road usage times
        unordered_map<pair<int, int>, float, pair_hash> roadEndTimes;

        float maxTime = 0.0f;

        for (int i = 0; i < num_populated_cities; i++)
        {
            int originCity = populatedCities[i];
            const vector<int> &path = inputPaths[i];
            const vector<Drop> &cityDrops = inputDrops[i];

            int remainingPrime = cities[originCity].prime;
            int remainingElderly = cities[originCity].elderly;
            float currentTime = 0.0f;

            // Process each segment of the path
            for (size_t j = 0; j < path.size() - 1; j++)
            {
                int u = path[j];
                int v = path[j + 1];

                // Find the road
                auto roadKey = make_pair(min(u, v), max(u, v));
                if (roadMap.find({u, v}) == roadMap.end())
                {
                    cerr << "Error: No road found from " << u << " to " << v << "\n";
                    continue;
                }

                int roadLength = roadMap[{u, v}].length;
                int roadCapacity = roadMap[{u, v}].capacity;

                // Check if road is being used
                if (roadEndTimes.find(roadKey) != roadEndTimes.end())
                {
                    currentTime = max(currentTime, roadEndTimes[roadKey]);
                }

                int totalPeople = remainingPrime + remainingElderly;
                if (totalPeople > 0)
                {
                    float travelTime = calculateTravelTime(roadLength, roadCapacity, totalPeople);
                    float endTime = currentTime + travelTime;
                    roadEndTimes[roadKey] = endTime;
                    currentTime = endTime;
                }

                // Process drops at the next city
                for (const Drop &drop : cityDrops)
                {
                    if (drop.city == v)
                    {
                        remainingPrime -= drop.prime;
                        remainingElderly -= drop.elderly;
                    }
                }
            }

            maxTime = max(maxTime, currentTime);
        }

        simulationTime = maxTime;
        return maxTime;
    }

    // Print statistics
    void printStats(ostream &out) const
    {
        out << "TotalPopulation: " << totalPopulation << "\n";
        out << "TotalElderly: " << totalElderly << "\n";
        out << "TotalPrime: " << totalPrime << "\n";
        out << "TotalShelterCap: " << totalShelterCap << "\n";
        out << "SavedElderly: " << savedElderly << "\n";
        out << "SavedPrime: " << savedPrime << "\n";
        out << "TotalSaved: " << totalSaved << "\n";
        out << "DroppedElderly: " << droppedElderly << "\n";
        out << "DroppedPrime: " << droppedPrime << "\n";
        out << "TotalDropped: " << totalDropped << "\n";
        out << "SimulationTime: " << simulationTime * 60 << " minutes\n"; // Convert hours to minutes
    }

    // Run all validations
    bool validate()
    {
        bool pathsValid = validatePaths();
        bool dropsValid = validateDrops();
        bool elderlyValid = validateElderlyDistance();
        simulateEvacuation();

        return pathsValid && dropsValid && elderlyValid;
    }

    // Overall validation status
    bool isValid() const
    {
        return pathsValid && dropsValid && elderlyDistanceValid;
    }
};

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file> [stats_file]\n";
        return 1;
    }

    string inputFile = argv[1];
    string outputFile = argv[2];
    string statsFile = (argc > 3) ? argv[3] : "validation_stats.txt";

    EvacuationValidator validator;

    // Read input file
    if (!validator.readInputFile(inputFile))
    {
        return 1;
    }

    // Read output file
    ifstream outfile(outputFile);
    if (!outfile)
    {
        cerr << "Error: Cannot open output file " << outputFile << "\n";
        return 1;
    }

    if (!validator.readOutputData(outfile))
    {
        return 1;
    }

    outfile.close();

    // Validate the solution
    bool valid = validator.validate();

    // Write statistics
    ofstream statsOut(statsFile);
    validator.printStats(statsOut);
    statsOut.close();

    cout << "Validation " << (valid ? "PASSED" : "FAILED") << "\n";
    cout << "Statistics written to " << statsFile << "\n";

    return valid ? 0 : 1;
}