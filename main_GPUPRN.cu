// Let this script use GPU generated random numbers

#include <stdio.h>
#include "support.h"
#include "kernel.cu"
#include <curand.h>
#include <curand_kernel.h>

#define c_1 1    //nonnegative constants
#define c_2 1    //nonnegative constants
#define inertia 0.5    //within [0,1]

#define max_iters 2048              //number of iterations
#define phi c_1+c_2

// #define chi 0.72984f                //chi (constriction factor)
// #define pi 3.14159265f
// #define inf 9999.99f                //infinity


int main(int argc, char**argv){
    
    float *global_best;
    float *global_best_pos;
    float *pos, *velocity;
    float *p_best_x, *p_best_y;
    int *local_best_index, *best_local_index;
    curandState *states;
    unsigned int N;
    unsigned int D;
    unsigned float xmax;
    unsigned float xmin;
    unsigned float c_1;
    unsigned float c_2;
    unsigned float inertia;


    if (argc<2)
    {
        printf("You need add 2 parameters\n");
        return 1;
    }
    for (int i=1; i<argc; i++){
        N= atoi(argv[i]);
        D= atoi(argv[++i]);
        xmax= atoi(argv[++i]);
        xmin = atoi(argv[++i]);
        c_1 = atoi(argv[++i]);
        c_2 = atoi(argv[++i]);
        inertia =  atoi(argv[++i]);
    }

//  MEMORY ALLOCATION

    // Host and device meory allocation of input variables
    malloc((void**)&global_best_h, D * sizeof(float));
    malloc((void**)&global_best_index_h, sizeof(float));
    malloc((void**)&N_h, sizeof(float));
    malloc((void**)&D_h, sizeof(float));
    malloc((void**)&xmax_h, sizeof(float));
    malloc((void**)&xmin_h, sizeof(float));
    malloc((void**)&c_1_h, sizeof(float));
    malloc((void**)&c_2_h, sizeof(float));
    malloc((void**)&inertia_h, sizeof(float));

    cudaMalloc((void**)&global_best, D * sizeof(float));
    cudaMalloc((void**)&global_best_index, sizeof(float));
    cudaMalloc((void**)&N, sizeof(float));
    cudaMalloc((void**)&D, sizeof(float));
    cudaMalloc((void**)&xmax, sizeof(float));
    cudaMalloc((void**)&xmin, sizeof(float));
    cudaMalloc((void**)&c_1, sizeof(float));
    cudaMalloc((void**)&c_2, sizeof(float));
    cudaMalloc((void**)&inertia, sizeof(float));

    // Migrate from host to device the inputs
    cuda_ret = cudaMemcpy(global_best, global_best_h, D * sizeof(float), cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess)
    {
      printf("CUDA Error in global_best memory allocation on device: %s\n", cudaGetErrorString(err));
      exit(-1);
    }

    cuda_ret = cudaMemcpy(global_best_index, global_best_index_h, sizeof(float), cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess)
    {
      printf("CUDA Error in global_best_index memory allocation on device: %s\n", cudaGetErrorString(err));
      exit(-1);
    }

    cuda_ret = cudaMemcpy(N, N_h, sizeof(float), cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess)
    {
      printf("CUDA Error in N memory allocation on device: %s\n", cudaGetErrorString(err));
      exit(-1);
    }

    cuda_ret = cudaMemcpy(D, D_h, sizeof(float), cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess)
    {
      printf("CUDA Error in D memory allocation on device: %s\n", cudaGetErrorString(err));
      exit(-1);
    }

    cuda_ret = cudaMemcpy(xmax, xmax_h, sizeof(float), cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess)
    {
      printf("CUDA Error in xmax memory allocation on device: %s\n", cudaGetErrorString(err));
      exit(-1);
    }

    cuda_ret = cudaMemcpy(xmin, xmin_h, sizeof(float), cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess)
    {
      printf("CUDA Error in xmin memory allocation on device: %s\n", cudaGetErrorString(err));
      exit(-1);
    }

    cuda_ret = cudaMemcpy(c_1, c_1_h, sizeof(float), cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess)
    {
      printf("CUDA Error in c_1 memory allocation on device: %s\n", cudaGetErrorString(err));
      exit(-1);
    }

    cuda_ret = cudaMemcpy(c_2, c_2_h, sizeof(float), cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess)
    {
      printf("CUDA Error in c_2 memory allocation on device: %s\n", cudaGetErrorString(err));
      exit(-1);
    }

    cuda_ret = cudaMemcpy(inertia, inertia_h, sizeof(float), cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess)
    {
      printf("CUDA Error in inertia memory allocation on device: %s\n", cudaGetErrorString(err));
      exit(-1);
    }


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

    printf("ALL GOOD FOR MEMORY ALLOCATION");

//  Initialization of random numbers on GPU
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

// Scale velocity and position values to be between the max and min (and not between 0 and 1 anymore)
// AND initialize particle best (for each particle) + local best 

    Scale_Init <<< N, D >>>(particle_position, particle_velocity, p_best_fitness, l_best_index, best_index, states, xmax, xmin);
    cudaError_t err = cudaGetLastError();        // Get error code
   if ( err != cudaSuccess )
   {
      printf("CUDA Error Scale Init: %s\n", cudaGetErrorString(err));
      exit(-1);
   }   
}
