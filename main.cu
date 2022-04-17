// Let this script use GPU generated random numbers


#include <stdio.h>
#include <stdlib.h>
#include "support.h"
#include "kernel.cu"
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>


#define max_iters 2048              //number of iterations

#define inf 9999.99f                //infinity

int main(int argc, char**argv){
    Timer timer;
    unsigned int N;
    unsigned int D;
    float xmax;
    float xmin;
    float c_1;
    float c_2;
    float inertia;
    float vmax;  
    int *l_best_index, *best_index;
    float *particle_position, *particle_velocity;
    float *p_best_pos, *p_best_fitness;
    curandState *states;
    float *g_best_pos;
    float *g_best;
    cudaError_t err;

    startTime(&timer); 

    if (argc!=9)
    {
        printf("\n     Invalid number of arguments!");
    }

    N           = atoi(argv[1]);
    D           = atoi(argv[2]);
    xmax        = atoi(argv[3]);
    xmin        = atoi(argv[4]);
    c_1         = atoi(argv[5]);
    c_2         = atoi(argv[6]);
    inertia     = atoi(argv[7]);
    vmax        = atoi(argv[8]);

    // Calculated variables	
    float chi;
    chi = 2/abs(2-c_1 - c_2 - sqrt((c_2+c_1)*(c_2+c_1)-(4*c_2+4*c_1)));

//  MEMORY ALLOCATION

    // Allocating memory for all variables that do not come from input
//    printf("Allocating memory for all other variables that do not come from input ..."); fflush(stdout);

    //Dynamically allocating memory for results
    g_best = new float;
    g_best_pos = new float[D];
    
    //Particle position array
    cudaMalloc((void**)&particle_position, N * D * sizeof(float));
    err = cudaGetLastError();        // Get error code
   if ( err != cudaSuccess )
   {
      printf("CUDA Error in particle position array memory allocation: %s\n", cudaGetErrorString(err));
      exit(-1);
   }

    //Velocity array
    cudaMalloc((void**)&particle_velocity, N * D * sizeof(float));
    err = cudaGetLastError();        // Get error code
   if ( err != cudaSuccess )
   {
      printf("CUDA Error in velocity array memory allocation: %s\n", cudaGetErrorString(err));
      exit(-1);
   }
   
    //Particle best position array
    cudaMalloc((void**)&p_best_pos, N * D * sizeof(float));
    err = cudaGetLastError();        // Get error code
   if ( err != cudaSuccess )
   {
      printf("CUDA Error in Particle best position array memory allocation: %s\n", cudaGetErrorString(err));
      exit(-1);
   }
   
    //Particle best fitness value
    cudaMalloc((void**)&p_best_fitness, N * sizeof(float));
    err = cudaGetLastError();        // Get error code
   if ( err != cudaSuccess )
   {
      printf("CUDA Error in Particle best fitness value memory allocation: %s\n", cudaGetErrorString(err));
      exit(-1);
   }

    //Local best index value (from a fintess point of view)
    cudaMalloc((void**)&l_best_index, N * sizeof(int));
    err = cudaGetLastError();        // Get error code
   if ( err != cudaSuccess )   
   {
      printf("CUDA Error in Local best index value memory allocation: %s\n", cudaGetErrorString(err));
      exit(-1);
   }

    //Global Best index value (from a fintess point of view)
    cudaMalloc((void**)&best_index, N * sizeof(int));
    err = cudaGetLastError();        // Get error code
   if ( err != cudaSuccess )
   {
      printf("CUDA Error in Global best index value memory allocation: %s\n", cudaGetErrorString(err));
      exit(-1);
   }
   
   // Cuda Rand memory allocation
    cudaMalloc((void**)&states, N * D * sizeof(curandState));
    err = cudaGetLastError();        // Get error code
    if ( err != cudaSuccess )
    {
      printf("Cuda Rand memory allocation: %s\n", cudaGetErrorString(err));
      exit(-1);
    }   
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

//  Initialization of random numbers on GPU
//    printf("Initialization of random numbers on GPU for particle's velocity and position ..."); fflush(stdout);
    startTime(&timer); 
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, time(NULL));

//  Initialize the particle's velocity (values of the array within the bounds): 
    curandGenerateUniform(generator, particle_velocity, N * D);
    err = cudaGetLastError();        // Get error code
   if ( err != cudaSuccess )
   {
      printf("CUDA Error in random generation for intial particle velocity: %s\n", cudaGetErrorString(err));
      exit(-1);
   }
//  Initialize the particle's position with a uniformly distributed random vector (values within the bounds)
    curandGenerateUniform(generator, particle_position, N * D);
    err = cudaGetLastError();        // Get error code
   if ( err != cudaSuccess )
   {
      printf("CUDA Error in random generation for intial particle position: %s\n", cudaGetErrorString(err));
      exit(-1);
   }
    cudaDeviceSynchronize();
//    stopTime(&timer); printf("%f s\n", elapsedTime(timer));


// Scale velocity and position values to be between the max and min (and not between 0 and 1 anymore)
// AND initialize particle best (for each particle) + local best 
//    printf("Launch kernel to scale and initialize ..."); fflush(stdout);
//    startTime(&timer); 
    const unsigned int THREADS_PER_BLOCK = 20;
    const unsigned int numBlocks = N/THREADS_PER_BLOCK +1 ;
    dim3 gridDim(numBlocks, 1, 1), blockDim(THREADS_PER_BLOCK, 1, 1);    
    Scale_Init <<< gridDim, blockDim >>>(xmax, xmin, particle_position, particle_velocity, p_best_fitness, l_best_index, best_index, states);
    err = cudaGetLastError();        // Get error code
   if ( err != cudaSuccess )
   {
      printf("CUDA Error Scale Init: %s\n", cudaGetErrorString(err));
      exit(-1);
   }
    cudaDeviceSynchronize();
//    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

//  Kernel for iterations
//    printf("Launch kernel to compute iterations ..."); fflush(stdout);
//    startTime(&timer); 
    for (int i = 0; i < max_iters; i++){
        Iterations<<< gridDim, blockDim >>>(xmax, xmin, particle_position, particle_velocity, p_best_pos, p_best_fitness, l_best_index, best_index, states, c_1, c_2, inertia, vmax, chi, N, D);
    }
   err = cudaGetLastError();        // Get error code
   if ( err != cudaSuccess )
   {
      printf("CUDA Error in iterations: %s\n", cudaGetErrorString(err));
      exit(-1);
   }
    cudaDeviceSynchronize();
    
//    stopTime(&timer); printf("%f s\n", elapsedTime(timer)); 

//  Kernel for min computation - ReduceKernel 1
//    printf("Run kernel to compute reduction - step 1..."); fflush(stdout);
//    startTime(&timer); 
    ReduceKernel1<<< gridDim, blockDim >>>(p_best_fitness, best_index);
    err = cudaGetLastError();        // Get error code
    if ( err != cudaSuccess )
    {
        printf("CUDA Error in ReduceKernel1: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    cudaDeviceSynchronize();
    
//    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    
//  Kernel for min computation - ReduceKernel 2
//    printf("Run kernel to compute reduction - step 2..."); fflush(stdout);
//    startTime(&timer); 
    ReduceKernel2<<< 1, (N / 20) + 1 >>>(p_best_pos, p_best_fitness, best_index, D);
    err = cudaGetLastError();        // Get error code
    if ( err != cudaSuccess )
    {
       printf("CUDA Error in ReduceKernel2: %s\n", cudaGetErrorString(err));
       exit(-1);
    }
    cudaDeviceSynchronize();
    
//    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    cudaMemcpy((void*)g_best, (void*)p_best_fitness, sizeof(float), cudaMemcpyDeviceToHost);
    if ( err != cudaSuccess )
    {
       printf("CUDA Error in global min: %s\n", cudaGetErrorString(err));
       exit(-1);
    }
    
    //Copy co-ordinates of global minimum
    cudaMemcpy((void*)g_best_pos, (void*)p_best_pos, D * sizeof(float), cudaMemcpyDeviceToHost);
    if ( err != cudaSuccess )
    {
       printf("CUDA Error in coordinates global min: %s\n", cudaGetErrorString(err));
       exit(-1);
    }
    
    printf("%f \n", g_best);

//    printf("Freeing memory");
    
    //Free device matrices
    cudaFree(particle_position);
    cudaFree(particle_velocity);
    cudaFree(p_best_pos);
    cudaFree(p_best_fitness);
    cudaFree(l_best_index);
    cudaFree(best_index);
    
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("\n");


}
