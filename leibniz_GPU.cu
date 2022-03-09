#include <iostream>
#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <chrono>


// If the architecture is less than 600, it does not include the atomicAdd function and must define
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

__global__ void pi_elem(double *values)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    values[i] = pow((double)-1, (double)i);
    values[i] /= (2 * i + 1);
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
    std::cout << "Duracion:" << durationCuda.count() << "[uS]" << std::endl;

    return 0;
}