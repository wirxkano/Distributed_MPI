#include <iostream>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <functional>
#include <condition_variable>
#include <chrono>
#include <math.h>
#include <mpi.h>

using namespace std;
using Task = std::function<void()>;

#define NUM_THREAD 4

const int N = 4000;
const double CELL_SIZE = 10.0;
const double SOUND_SPEED = 343.0;
const double YIELD = 5000 * 1.0e6; // kilotons to kg TNT

class TaskQueue
{
private:
    std::mutex mtx;
    std::queue<Task> queue;
    std::condition_variable cv;
    bool shutting_down = false;
    int unfinished_tasks = 0;

public:
    static TaskQueue *get()
    {
        static TaskQueue instance;
        return &instance;
    }

    void enqueue(const Task &task)
    {
        {
            std::unique_lock<std::mutex> lock(mtx);
            queue.push(task);
            ++unfinished_tasks;
        }
        cv.notify_one();
    }

    Task dequeue()
    {
        std::unique_lock<std::mutex> lock(mtx);

        cv.wait(lock, [&]()
                { return shutting_down || !queue.empty(); });

        if (shutting_down)
            return nullptr;
        if (queue.empty())
            return nullptr;

        Task task = queue.front();
        queue.pop();
        return task;
    }

    void task_done()
    {
        std::unique_lock<std::mutex> lock(mtx);
        --unfinished_tasks;
        if (unfinished_tasks == 0)
        {
            cv.notify_all();
        }
    }

    void wait_all()
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&]()
                { return unfinished_tasks == 0; });
    }

    void shutdown()
    {
        {
            std::unique_lock<std::mutex> lock(mtx);
            shutting_down = true;
        }
        cv.notify_all();
    }
};

class Worker
{
private:
    bool stop;
    std::thread t;
    int id;

    void run()
    {
        while (!stop)
        {
            Task task = TaskQueue::get()->dequeue();
            if (!task)
                continue;
            task();
            TaskQueue::get()->task_done();
        }
    }

public:
    Worker(int id) : id(id), stop(false)
    {
        t = std::thread(&Worker::run, this);
    }

    void exit()
    {
        stop = true;
        TaskQueue::get()->shutdown();
        if (t.joinable())
            t.join();
    }
};

double compute_Z(double R, double W)
{
    return R * pow(W, -1.0 / 3.0);
}

double compute_peak_overpressure(double Z)
{
    double U = -0.21436 + 1.35034 * log10(Z);

    const double c[9] = {
        2.611369, -1.690128, 0.00805, 0.336743, -0.005162,
        -0.080923, -0.004785, 0.007930, 0.000768};

    double sum = 0;
    double Ui = 1.0;

    for (int i = 0; i < 9; i++)
    {
        sum += c[i] * Ui;
        Ui *= U;
    }

    return pow(10, sum);
}

void parallel_compute(int rstart, int rend, vector<double> &local_grid, vector<double> &global_grid, vector<int> &recvcounts, vector<int> &displs)
{

    int rows_local = rend - rstart;
    int cx = N / 2;
    int cy = N / 2;

    vector<Worker *> workers;
    for (int i = 0; i < NUM_THREAD; i++)
        workers.push_back(new Worker(i));

    auto start = chrono::high_resolution_clock::now();

    for (int t = 1; t <= 100; t++)
    {
        double Rmax = t * SOUND_SPEED;

        for (int row = rstart; row < rend; row++)
        {
            TaskQueue::get()->enqueue([&, row, Rmax]()
                                      {
                int local_r = row - rstart;

                for (int j = 0; j < N; j++) {

                    if (local_grid[local_r * N + j] != 0) continue;

                    double dx = (row - cx) * CELL_SIZE;
                    double dy = (j - cy) * CELL_SIZE;
                    double R = sqrt(dx * dx + dy * dy);

                    if (R <= Rmax) {
                        double Z = compute_Z(R, YIELD);
                        local_grid[local_r * N + j] = compute_peak_overpressure(Z);
                    }
                } });
        }
        TaskQueue::get()->wait_all();

        // Synchronous across all ranks at each time step
        MPI_Barrier(MPI_COMM_WORLD);

        // Assemble a global map after each time step (stimulation map at each time step)
        MPI_Allgatherv(local_grid.data(), (rend - rstart) * N, MPI_DOUBLE,
                       global_grid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
    }

    TaskQueue::get()->shutdown();
    for (Worker *w : workers)
    {
        w->exit();
        delete w;
    }

    auto end = chrono::high_resolution_clock::now();
    double sec = chrono::duration<double>(end - start).count();

    cout << "Rank finished local parallel time: " << sec << " sec\n";
}

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Partition rows among ranks
    int rows_per_rank = N / size;
    int rstart = rank * rows_per_rank;
    int rend = (rank == size - 1) ? N : rstart + rows_per_rank;

    vector<int> recvcounts(size), displs(size);
    for (int r = 0; r < size; r++)
    {
        int rows_r = (r == size - 1) ? N - r * rows_per_rank : rows_per_rank;
        recvcounts[r] = rows_r * N;
        displs[r] = r * rows_per_rank * N;
    }

    vector<double> local_grid((rend - rstart) * N, 0.0);
    vector<double> global_grid(N * N, 0.0);

    if (rank == 0)
    {
        printf("Processing...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = chrono::high_resolution_clock::now();

    parallel_compute(rstart, rend, local_grid, global_grid, recvcounts, displs);

    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = chrono::high_resolution_clock::now();

    double local_time = chrono::duration<double>(t1 - t0).count();
    double max_secs;

    MPI_Reduce(&local_time, &max_secs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        cout << "Synchronous shockwave simulation finished. Max time across ranks: " << max_secs << "s\n";
    }

    MPI_Finalize();
    return 0;
}