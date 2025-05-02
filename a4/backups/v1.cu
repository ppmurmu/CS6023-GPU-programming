#include <iostream>
#include <fstream>
using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return 1;
    }

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

    // set your answer to these variables
    long long *path_size;
    long long **paths;
    long long *num_drops;
    long long ***drops;

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

    return 0;
}
