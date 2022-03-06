// Let this script use GPU generated random numbers


#include <stdio.h>
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
    printf("Allocating memory for all other variables that do not come from input ..."); fflush(stdout);
    startTime(&timer); 

    //Particle position array (=Position in Git)
    cudaMalloc((void**)&particle_position, N * D * sizeof(float));
    cudaError_t err = cudaGetLastError();        // Get error code
   if ( err != cudaSuccess )
   {
      printf("CUDA Error in particle position array memory allocation: %s\n", cudaGetErrorString(err));
      exit(-1);
   }

    //Velocity array (=Velocity in Git)
    cudaMalloc((void**)&particle_velocity, N * D * sizeof(float));
    cudaError_t err = cudaGetLastError();        // Get error code
   if ( err != cudaSuccess )
   {
      printf("CUDA Error in velocity array memory allocation: %s\n", cudaGetErrorString(err));
      exit(-1);
   }
   
    //Particle best position array (=PBestX in git)
    cudaMalloc((void**)&p_best_pos, N * D * sizeof(float));
    cudaError_t err = cudaGetLastError();        // Get error code
   if ( err != cudaSuccess )
   {
      printf("CUDA Error in Particle best position array memory allocation: %s\n", cudaGetErrorString(err));
      exit(-1);
   }
   
    //Particle best fitness value (=PBestY Array in Git)
    cudaMalloc((void**)&p_best_fitness, N * sizeof(float));
    cudaError_t err = cudaGetLastError();        // Get error code
   if ( err != cudaSuccess )
   {
      printf("CUDA Error in Particle best fitness value memory allocation: %s\n", cudaGetErrorString(err));
      exit(-1);
   }

    //Local best index value (from a fintess point of view) (=LBestIndex in Git)
    cudaMalloc((void**)&l_best_index, N * sizeof(int));
    cudaError_t err = cudaGetLastError();        // Get error code
   if ( err != cudaSuccess )   
   {
      printf("CUDA Error in Local best index value memory allocation: %s\n", cudaGetErrorString(err));
      exit(-1);
   }

    //Global Best index value (from a fintess point of view)(=GBestIndex in Git)
    cudaMalloc((void**)&best_index, N * sizeof(int));
    cudaError_t err = cudaGetLastError();        // Get error code
   if ( err != cudaSuccess )
   {
      printf("CUDA Error in Global best index value memory allocation: %s\n", cudaGetErrorString(err));
      exit(-1);
   }
   
   // Cuda Rand memory allocation
    cudaMalloc((void**)&states, N * D * sizeof(curandState));
    cudaError_t err = cudaGetLastError();        // Get error code
    if ( err != cudaSuccess )
    {
      printf("Cuda Rand memory allocation: %s\n", cudaGetErrorString(err));
      exit(-1);
    }   
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

//  Initialization of random numbers on GPU
    printf("Initialization of random numbers on GPU for particle's velocity and position ..."); fflush(stdout);
    startTime(&timer); 
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, time(NULL));

//  Initialize the particle's velocity (values of the array within the bounds): 
    curandGenerateUniform(generator, particle_velocity, N * D);
    cudaError_t err = cudaGetLastError();        // Get error code
   if ( err != cudaSuccess )
   {
      printf("CUDA Error in random generation for intial particle velocity: %s\n", cudaGetErrorString(err));
      exit(-1);
   }
//  Initialize the particle's position with a uniformly distributed random vector (values within the bounds)
    curandGenerateUniform(generator, particle_position, N * D);
    cudaError_t err = cudaGetLastError();        // Get error code
   if ( err != cudaSuccess )
   {
      printf("CUDA Error in random generation for intial particle position: %s\n", cudaGetErrorString(err));
      exit(-1);
   }
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));


// Scale velocity and position values to be between the max and min (and not between 0 and 1 anymore)
// AND initialize particle best (for each particle) + local best 
    printf("Launch kernel to scale and initialize ..."); fflush(stdout);
    startTime(&timer); 
    const unsigned int THREADS_PER_BLOCK = 200;
    const unsigned int numBlocks = N/THREADS_PER_BLOCK;
    dim3 gridDim(numBlocks, 1, 1), blockDim(THREADS_PER_BLOCK, 1, 1);    
    Scale_Init <<< gridDim, blockDim >>>(xmax, xmin, particle_position, particle_velocity, p_best_fitness, l_best_index, best_index, states);
    cudaError_t err = cudaGetLastError();        // Get error code
   if ( err != cudaSuccess )
   {
      printf("CUDA Error Scale Init: %s\n", cudaGetErrorString(err));
      exit(-1);
   }
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

//  Kernel for iterations
    printf("Launch kernel to compute iterations ..."); fflush(stdout);
    startTime(&timer); 
    for (int i = 0; i < max_iters; i++){
        Iterations<<< gridDim, blockDim >>>(xmax, xmin, particle_position, particle_velocity, p_best_pos, p_best_fitness, l_best_index, best_index, states, c_1, c_2, inertia, vmax, chi, N, D);
    }
   cudaError_t err = cudaGetLastError();        // Get error code
   if ( err != cudaSuccess )
   {
      printf("CUDA Error in iterations: %s\n", cudaGetErrorString(err));
      exit(-1);
   }
    cudaDeviceSynchronize();
    
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("\nFreeing memory");

    //INSERT CODE HERE to free device matrices
    cudaFree(particle_position);
    cudaFree(particle_velocity);
    cudaFree(p_best_pos);
    cudaFree(p_best_fitness);
    cudaFree(l_best_index);
    cudaFree(best_index);
}
