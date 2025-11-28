#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include <numeric>
#include <iomanip>
#include <vector>
#include <mpi.h>
#include <omp.h>

using namespace std;

#define NUM_ITER 100

const int PAD_SIZE = 1;
const double PAD_VAL = 0.0;
const int RSIZE = 4000;
const int CSIZE = 4000;

const double D = 1000.0;
const double k = 1e-4;
const double lambda = 3e-5;
const double ux = 3.3;
const double uy = 1.4;

const double dx = 10.0;
const double dy = dx;
const double dt = 1;

const double inv_dx = 1.0 / dx;
const double inv_dy = 1.0 / dy;
const double inv_dx2 = 1.0 / (dx * dx);
const double inv_dy2 = 1.0 / (dy * dy);
const double decayRate = lambda + k;

double parallel(vector<vector<double>> &mat, int rsize, int csize, int world_rank, int world_size)
{
    double start = MPI_Wtime();

    // Split rows among processes
    int base_rows = rsize / world_size;
    int remainder = rsize % world_size;

    vector<int> sendcounts(world_size), displs(world_size);
    int offset = 0;
    for (int i = 0; i < world_size; ++i)
    {
        int rows = base_rows + (i < remainder);
        sendcounts[i] = rows * csize;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    int local_rows = base_rows + (world_rank < remainder);

    // Allocate local grid with 2 placeholder rows for neighbor
    vector<vector<double>> local(local_rows + 2, vector<double>(csize, 0.0));
    vector<vector<double>> new_local(local_rows + 2, vector<double>(csize, 0.0));

    // Prepare send buffer only on root
    vector<double> flat;
    if (world_rank == 0)
    {
        flat.resize(rsize * csize);
        for (int i = 0; i < rsize; ++i)
            memcpy(&flat[i * csize], mat[i].data(), csize * sizeof(double));
    }

    // Local receive buffer
    vector<double> recvbuf(local_rows * csize);
    MPI_Scatterv(flat.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 recvbuf.data(), recvbuf.size(), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Copy received block into local grid, leaving one boundary row on each side
    for (int i = 0; i < local_rows; ++i)
        for (int j = 0; j < csize; ++j)
            local[i + 1][j] = recvbuf[i * csize + j];

    int stop_signal = 0;

    for (int step = 1; step <= NUM_ITER; ++step)
    {
        // Exchange boundary values with its neighboring processes
        MPI_Request recv_reqs[2];
        int recv_count = 0;
        MPI_Request send_reqs[2];
        int send_count = 0;

        // Receive from top neighbor
        if (world_rank > 0)
            MPI_Irecv(local[0].data(), csize, MPI_DOUBLE, world_rank - 1, 0, MPI_COMM_WORLD, &recv_reqs[recv_count++]);

        // Receive from bottom neighbor
        if (world_rank < world_size - 1)
            MPI_Irecv(local[local_rows + 1].data(), csize, MPI_DOUBLE, world_rank + 1, 0, MPI_COMM_WORLD, &recv_reqs[recv_count++]);

        // Send top row to upper neighbor
        if (world_rank > 0)
            MPI_Isend(local[1].data(), csize, MPI_DOUBLE, world_rank - 1, 0, MPI_COMM_WORLD, &send_reqs[send_count++]);

        // Send bottom row to lower neighbor
        if (world_rank < world_size - 1)
            MPI_Isend(local[local_rows].data(), csize, MPI_DOUBLE, world_rank + 1, 0, MPI_COMM_WORLD, &send_reqs[send_count++]);

// Compute new radioactive values
#pragma omp parallel for collapse(2)
        for (int i = 2; i < local_rows; ++i)
        {
            for (int j = 0; j < csize; ++j)
            {
                double center = local[i][j];
                double up = (i > 0) ? local[i - 1][j] : 0;
                double down = (i < local_rows + 1) ? local[i + 1][j] : 0;
                double left = (j > 0) ? local[i][j - 1] : 0;
                double right = (j < csize - 1) ? local[i][j + 1] : 0;

                double adv = ux * (center - up) * inv_dx + uy * (center - left) * inv_dy;
                double lap = (down - 2 * center + up) * inv_dx2 + (right - 2 * center + left) * inv_dy2;
                new_local[i][j] = center + dt * (D * lap - decayRate * center - adv);
            }
        }

        // Wait for all requests to complete
        MPI_Waitall(recv_count, recv_reqs, MPI_STATUSES_IGNORE);
        MPI_Waitall(send_count, send_reqs, MPI_STATUSES_IGNORE);

#pragma omp parallel for collapse(2)
        for (int i = 1; i <= local_rows; i += local_rows - 1)
        {
            for (int j = 0; j < csize; ++j)
            {
                double center = local[i][j];
                double up = (i > 0) ? local[i - 1][j] : 0;
                double down = (i < local_rows + 1) ? local[i + 1][j] : 0;
                double left = (j > 0) ? local[i][j - 1] : 0;
                double right = (j < csize - 1) ? local[i][j + 1] : 0;

                double adv = ux * (center - up) * inv_dx + uy * (center - left) * inv_dy;
                double lap = (down - 2 * center + up) * inv_dx2 + (right - 2 * center + left) * inv_dy2;
                new_local[i][j] = center + dt * (D * lap - decayRate * center - adv);
            }
        }

        // Swap grids
        local.swap(new_local);

        // Count uncontaminated blocks (value == 0)
        long int local_uncontaminated = 0;

#pragma omp parallel for collapse(2) reduction(+ : local_uncontaminated)
        for (int i = 1; i <= local_rows; ++i)
        {
            for (int j = 0; j < csize; ++j)
            {
                if (local[i][j] == 0)
                    ++local_uncontaminated;
            }
        }

        long int total_uncontaminated = 0;
        MPI_Reduce(&local_uncontaminated, &total_uncontaminated, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if (world_rank == 0)
            cout << "Iteration " << step << ": uncontaminated block = " << total_uncontaminated << '\n';

        // synchronizes with other processes before moving to the next step
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Stop signal is distributed to all processes using MPI Broadcast
    if (world_rank == 0)
        stop_signal = 1;

    MPI_Bcast(&stop_signal, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (stop_signal == 1 && world_rank == 0)
        cout << "Broadcasted stop signal to all processes.\n";

    // Gather final map
    for (int i = 0; i < local_rows; ++i)
        for (int j = 0; j < csize; ++j)
            recvbuf[i * csize + j] = local[i + 1][j];

    vector<double> final_flat;
    if (world_rank == 0)
        final_flat.resize(rsize * csize);

    MPI_Gatherv(recvbuf.data(), recvbuf.size(), MPI_DOUBLE,
                final_flat.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    return MPI_Wtime() - start;
}

void load_file(vector<vector<double>> &mat, string filename, int &rsize, int &csize)
{
    fstream file(filename);
    string line;

    if (!file.is_open())
    {
        cout << "Error open file\n";
        exit(1);
    }

    while (getline(file, line))
    {
        stringstream ss(line);
        string cell_value;
        int col_count = 0;
        mat.push_back(vector<double>());

        while (getline(ss, cell_value, ','))
        {
            mat[rsize].push_back(stod(cell_value));
            col_count++;
        }

        if (csize == 0)
            csize = col_count;
        rsize++;
    }
}

int main(int argc, char **argv)
{
    string filename = "radioactive_matrix.csv";

    vector<vector<double>> mat;
    int rsize = 0;
    int csize = 0;

    int world_rank = 0, world_size = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank == 0)
    {
        cout << "Loading " << filename << "...\n";
        load_file(mat, filename, rsize, csize);
        cout << "Loaded completely\n";

        cout << "\nProcessing Parallel method...\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double t_parallel = parallel(mat, RSIZE, CSIZE, world_rank, world_size);

    cout << "Rank " << world_rank << " executed in " << t_parallel << "s\n";

    double max_secs;

    MPI_Reduce(&t_parallel, &max_secs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        cout << "Asynchronous radioactive diffusion finished. Max time across ranks: " << t_parallel << "s\n";
    }

    MPI_Finalize();

    return 0;
}
