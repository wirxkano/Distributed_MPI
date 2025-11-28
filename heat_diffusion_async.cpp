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
#include <cstring>

using namespace std;

#define NUM_ITER 100

const int PAD_SIZE = 1;
const double PAD_VAL = 30.0;
const int K_SIZE = 3;

const double K[K_SIZE][K_SIZE] = {
    {0.05, 0.1, 0.05},
    {0.1, 0.4, 0.1},
    {0.05, 0.1, 0.05}};

double conv(vector<vector<double>> &mat, int r, int c)
{
    double sum = 0.0;

    for (int u = 0; u < K_SIZE; ++u)
    {
        for (int v = 0; v < K_SIZE; ++v)
        {
            sum += mat[r + u][c + v] * K[u][v];
        }
    }

    return sum;
}

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
    vector<vector<double>> local(local_rows + 2, vector<double>(csize + 2 * PAD_SIZE, PAD_VAL));
    vector<vector<double>> conv_local(local_rows, vector<double>(csize, PAD_VAL));

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

    // Copy received block into local grid, leaving one boundary row/col on each side
    for (int i = 0; i < local_rows; ++i)
        for (int j = 0; j < csize; ++j)
            local[i + PAD_SIZE][j + PAD_SIZE] = recvbuf[i * csize + j];

    int stop_signal = 0;

    for (int step = 1; step <= NUM_ITER; ++step)
    {
        // Exchange boundary values with its neighboring processes
        MPI_Request recv_reqs[2], send_reqs[2];
        int recv_count = 0, send_count = 0;

        if (world_rank > 0)
        {
            MPI_Irecv(local[0].data() + PAD_SIZE, csize, MPI_DOUBLE,
                      world_rank - 1, 0, MPI_COMM_WORLD, &recv_reqs[recv_count++]);
            MPI_Isend(local[PAD_SIZE].data() + PAD_SIZE, csize, MPI_DOUBLE,
                      world_rank - 1, 0, MPI_COMM_WORLD, &send_reqs[send_count++]);
        }

        if (world_rank < world_size - 1)
        {
            MPI_Irecv(local[local_rows + PAD_SIZE].data() + PAD_SIZE, csize, MPI_DOUBLE,
                      world_rank + 1, 0, MPI_COMM_WORLD, &recv_reqs[recv_count++]);
            MPI_Isend(local[local_rows].data() + PAD_SIZE, csize, MPI_DOUBLE,
                      world_rank + 1, 0, MPI_COMM_WORLD, &send_reqs[send_count++]);
        }

// Compute Convolution
#pragma omp parallel for collapse(2)
        for (int r = 1; r < local_rows - 1; ++r)
        {
            for (int c = 0; c < csize; ++c)
            {
                conv_local[r][c] = conv(local, r + PAD_SIZE - (K_SIZE / 2), c + PAD_SIZE - (K_SIZE / 2));
            }
        }

        MPI_Waitall(recv_count, recv_reqs, MPI_STATUSES_IGNORE);
        // MPI_Waitall(send_count, send_reqs, MPI_STATUSES_IGNORE);

#pragma omp parallel for
        for (int c = 0; c < csize; ++c)
        {
            conv_local[0][c] = conv(local, PAD_SIZE - (K_SIZE / 2), c + PAD_SIZE - (K_SIZE / 2));
            conv_local[local_rows - 1][c] = conv(local, local_rows - 1 + PAD_SIZE - (K_SIZE / 2), c + PAD_SIZE - (K_SIZE / 2));
        }

// copy back to local mat for next iteration
#pragma omp parallel for collapse(2)
        for (int r = 0; r < local_rows; ++r)
        {
            for (int c = 0; c < csize; ++c)
            {
                local[r + PAD_SIZE][c + PAD_SIZE] = conv_local[r][c];
            }
        }
    }

    // Stop signal is distributed to all processes using MPI Broadcast
    if (world_rank == 0)
        stop_signal = 1;

    if (stop_signal == 1 && world_rank == 0)
        cout << "Broadcasted stop signal to all processes.\n";

    // Gather final map
    for (int i = 0; i < local_rows; ++i)
        for (int j = 0; j < csize; ++j)
            recvbuf[i * csize + j] = local[i + PAD_SIZE][j + PAD_SIZE];

    vector<double> final_flat;
    if (world_rank == 0)
        final_flat.resize(rsize * csize);

    MPI_Gatherv(recvbuf.data(), recvbuf.size(), MPI_DOUBLE,
                world_rank == 0 ? final_flat.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
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
    string filename = "heat_matrix.csv";

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

    MPI_Bcast(&rsize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&csize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    double t_parallel = parallel(mat, rsize, csize, world_rank, world_size);

    cout << "Rank " << world_rank << " executed in " << t_parallel << "s\n";

    double max_secs;

    MPI_Reduce(&t_parallel, &max_secs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        cout << "Asynchronous heat diffusion finished. Max time across ranks: " << max_secs << "s\n";
    }

    MPI_Finalize();

    return 0;
}
