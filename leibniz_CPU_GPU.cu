#include <iostream>
#include <math.h>
#include <sstream>
#include <omp.h>
#include <chrono>
#include <pthread.h>
#include <stdio.h>
#include <cuda.h>

#define NUMBER_OF_CORES 4

using namespace std::chrono;
#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double *address, double val)
{
    unsigned long long int *address_as_ull =
        (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

struct argsP
{
    int ini;
    int fin;
    double *sum;
};

__global__ void pi_elem(double *values)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    values[i] = pow((double)-1, (double)i);
    values[i] /= (2 * i + 1);
}

// Poxis Function
void *suma(void *input)
{
    for (int i = ((struct argsP *)input)->ini; i < ((struct argsP *)input)->fin; i++)
    {
        *((struct argsP *)input)->sum += (pow(-1, i) / (2 * i + 1));
    }
    return NULL;
}

int main()
{
    int max;
    double sum, sum2;
    std::string ent;
    max = 1000;

    std::cout << "Insert n iterations" << std::endl;

    std::cin >> ent;

    max = stoi(ent);
    std::cout << "\n---- Using " << NUMBER_OF_CORES << " cores ----" << std::endl;


    // Pthread
    pthread_t tids[NUMBER_OF_CORES - 1];
    struct argsP *argumentosArray[NUMBER_OF_CORES - 1];
    auto start1 = high_resolution_clock::now();

    for (int i = 0; i < NUMBER_OF_CORES - 1; i++)
    {
        argumentosArray[i] = (struct argsP *)malloc(sizeof(struct argsP));
        argumentosArray[i]->ini = (int)i * (double)max / NUMBER_OF_CORES;
        argumentosArray[i]->fin = (int)(i + 1) * (double)max / NUMBER_OF_CORES;
        argumentosArray[i]->sum = new double(0);
        pthread_create(&tids[i], NULL, suma, (void *)argumentosArray[i]);
    }

    double total = 0;
    struct argsP *argumentos1 = (struct argsP *)malloc(sizeof(struct argsP));
    argumentos1->ini = (int)NUMBER_OF_CORES - 2 * (double)max / NUMBER_OF_CORES;
    argumentos1->fin = (int)NUMBER_OF_CORES - 1 * (double)max / NUMBER_OF_CORES;
    argumentos1->sum = new double(0);
    suma((void *)argumentos1);
    total += *argumentos1->sum;
    for (int i = 0; i < NUMBER_OF_CORES - 1; i++)
    {
        pthread_join(tids[i], NULL);
        total += *(argumentosArray[i]->sum);
    }
    total *= 4;
    auto stop1 = high_resolution_clock::now();
    auto duration1 = duration_cast<microseconds>(stop1 - start1);

    std::cout << "\nWith POXIS Threads: " << std::endl;
    std::cout << "Aproximation: " << total << std::endl;
    std::cout << "Duracion:" << duration1.count() << "[uS] \n"
              << std::endl;
    std::cout << "With OMP:" << std::endl;

    // Parallel OMP
    auto start = high_resolution_clock::now();
#pragma omp parallel for reduction(+ \
                                   : sum) num_threads(NUMBER_OF_CORES)
    for (int i = 0; i < max; i++)
    {
        sum = sum + (pow(-1, i) / (2 * i + 1));
    }
    sum = sum * 4;
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Aproximation: " << sum << std::endl;
    std::cout << "Duracion:" << duration.count() << "[uS]" << std::endl;

    std::cout << "\n\nSingle-core:" << std::endl;

    // One-core
    auto start2 = high_resolution_clock::now();
    for (int i = 0; i < max; i++)
    {
        sum2 = sum2 + (pow(-1, i) / (2 * i + 1));
    }
    sum2 = sum2 * 4;
    auto stop2 = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop2 - start2);
    std::cout << "Aproximation: " << sum2 << std::endl;
    std::cout << "Duracion:" << duration2.count() << "[uS]" << std::endl;

    std::cout << "\n\nWith CUDA:" << std::endl;


    // CUDA

    long N = (32 * max);
    double pi = 0;
    int i;
    double *d_values, *h_values;

    cudaMalloc((void **)&d_values, N * sizeof(double));
    h_values = (double *)malloc(N * sizeof(double));
    auto startCuda = high_resolution_clock::now();
    pi_elem<<<N / 32, 32>>>(d_values);

    cudaMemcpy(h_values, d_values, N * sizeof(double), cudaMemcpyDeviceToHost);
    auto stopCuda = high_resolution_clock::now();
    printf("%f\n", h_values[0]);
    for (i = 0; i < N; i++)
    {
        pi += h_values[i];
    }
    printf("Aproximation: %f\n", 4 * pi);

    auto durationCuda = duration_cast<microseconds>(stopCuda - startCuda);
    // std::cout << "Aproximation: " << *sumaCuda << std::endl;
    std::cout << "Duracion:" << durationCuda.count() << "[uS]" << std::endl;

    return 0;
}