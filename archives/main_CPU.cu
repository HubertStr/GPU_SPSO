
#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>


#define max_iters 2048              //number of iterations

#define inf 9999.99f                //infinity


int main(int argc, char**argv){

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ------------------------------------

    printf("\nSetting up the problem.."); fflush(stdout);
    startTime(&timer);

    // main function arguments
    if (argc!=9)
    {
        printf("\n     Invalid number of arguments!");
    }

    unsigned int N;
    unsigned int D;
    float xmax;
    float xmin;
    float c_1;
    float c_2;
    float inertia;
    float vmax;

    N           = atoi(argv[1]);
    D           = atoi(argv[2]);
    xmax        = atoi(argv[3]);
    xmin        = atoi(argv[4]);
    c_1         = atoi(argv[5]);
    c_2         = atoi(argv[6]);
    inertia     = atoi(argv[7]);
    vmax        = atoi(argv[8]);

    // other calculated variables	
    float chi;

    chi = 2/abs(2-c_1 - c_2 - sqrt((c_2+c_1)*(c_2+c_1)-(4*c_2+4*c_1)));

    // array variables
    unsigned int i;
    int M = N*D;

    float* particle_position = (float*) malloc( sizeof(float) * M );
    for (i=0; i < M; i++){
    	 particle_position[i] = (xmin + (rand()/(float) RAND_MAX) * (xmax - xmin)); 
	 };

    float* particle_velocity = (float*) malloc( sizeof(float) * M );
    for (i=0; i < M; i++){ 
    	particle_velocity[i] = (((rand()/(float) RAND_MAX) * (2 * vmax)) - vmax); 
	};

    float* p_best_pos = (float*) malloc( sizeof(float) * M);
    for (i=0; i< M; i++){ p_best_pos[i] = particle_position[i]; };

    float* p_best_fitness = (float*) malloc( sizeof(float)*N);
    for (i=0; i<N; i++){ p_best_fitness[i] = inf; }

    int* local_best_index  = (int*) malloc( sizeof(int)*N);
    for (i=0; i<N; i++){ local_best_index[i] = i; }

    int* best_index  = (int*) malloc( sizeof(int)*N);
    for (i=0; i<N; i++){ best_index[i] = i; };

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));


    // Allocate device variables ----------------------------------------------

    printf("\nAllocating device variables..."); fflush(stdout);
   startTime(&timer);


    float* particle_position_d;
    cuda_ret = cudaMalloc((void**) &particle_position_d, sizeof(float)*N*D);
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory particle_position");

    float* particle_velocity_d;
    cuda_ret = cudaMalloc((void**) &particle_velocity_d, sizeof(float)*N*D);
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory particle_velocity");

    float* p_best_pos_d;
    cuda_ret = cudaMalloc((void**) &p_best_pos_d, sizeof(float)*N*D);
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");


    float* p_best_fitness_d;
    cuda_ret = cudaMalloc((void**) &p_best_fitness_d, sizeof(float)*N);
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    float* local_best_index_d;
    cuda_ret = cudaMalloc((void**) &local_best_index_d, sizeof(int)*N);
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");


    float* best_index_d;
    cuda_ret = cudaMalloc((void**) &best_index_d, sizeof(int)*N);
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("\n Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(particle_position_d, particle_position, sizeof(float)*N*D, cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device particle_position");

    cuda_ret = cudaMemcpy(particle_velocity_d, particle_velocity, sizeof(float)*N*D, cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device particle_velocity");

    cuda_ret = cudaMemcpy(p_best_pos_d, p_best_pos, sizeof(float)*N*D, cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device p_best_pos_d");

    cuda_ret = cudaMemcpy(p_best_fitness_d, p_best_fitness, sizeof(float)*N, cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device p_best_fitness");

    cuda_ret = cudaMemcpy(local_best_index_d, local_best_index, sizeof(int)*N, cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device local_best_index");

    cuda_ret = cudaMemcpy(best_index_d, best_index, sizeof(int)*N, cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device best_index");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));


    // Launch kernel ----------------------------------------------------------

    printf("\n Launching kernel..."); fflush(stdout);
    startTime(&timer);

    const unsigned int THREADS_PER_BLOCK = 100;
    const unsigned int numBlocks = N/THREADS_PER_BLOCK;
    dim3 gridDim(numBlocks, 1, 1), blockDim(THREADS_PER_BLOCK, 1, 1);
    
    //INSERT CODE HERE to call kernel
    Iterate<<<gridDim, blockDim>>>(particle_position, particle_velocity, p_best_pos,  p_best_fitness, local_best_index, best_index, N, D, xmax, xmin, c_1, c_2, inertia, vmax, chi);
    

    cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("\nCopying data from device to host..."); fflush(stdout);
    startTime(&timer);
  
    cuda_ret = cudaMemcpy(p_best_fitness, p_best_fitness_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

    // Verify correctness -----------------------------------------------------

    printf("\nVerifying results..."); fflush(stdout);
    for (i=0; i<N; i++) { printf("%.2f",p_best_fitness[i]); printf("\n"); };

    // Free memory ------------------------------------------------------------
    
    printf("\nFreeing memory");
    free(particle_position);
    free(particle_velocity);
    free(p_best_pos);
    free(p_best_fitness);
    free(local_best_index);
    free(best_index);

    //INSERT CODE HERE to free device matrices
    cudaFree(particle_position_d);
    cudaFree(particle_velocity_d);
    cudaFree(p_best_pos_d);
    cudaFree(p_best_fitness_d);
    cudaFree(local_best_index_d);
    cudaFree(best_index_d);

    return(0);
}