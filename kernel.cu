#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

#define PI 3.14159265
#define inf 9999.99f

__global__ void Scale_Init(float xmax, float xmin, float *pos, float *velocity, float *p_best_y, int *l_best_index, int *best_index, curandState *states){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int t_index = threadIdx.x;
    
    //Rescale velocity so that it is within the bounds
    velocity[index] =(xmax - xmin) * (2.0f * velocity[index] - 1.0f);
    
    //Rescale pos so that it is within the bounds
    pos[index] = xmax * (2.0f * pos[index] - 1.0f);
    
    //Initializing p_best_y to infinity and local best to self
    if (t_index == 0)
    {
        p_best_y[blockIdx.x] = inf;
        l_best_index[blockIdx.x] = blockIdx.x;
        best_index[blockIdx.x] = blockIdx.x;
    }

    //call of Curand_init on a specific curandState, seed and no offset for each thread
    curand_init(index, index, 0, &states[index]);
}

// Kernel to compute the actual iterations of the updates 
__global__ void Iterations(float xmax, float xmin, float *pos, float *velocity, float *p_best_pos,float *p_best_y, int *l_best_index, int *best_index, curandState *states, float c_1, float c_2, float inertia, float vmax, float chi, int N, int D){
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    
    float r1, r2;
    
    float personal_best;
    int local_best;
    curandState local_state = states[index];

  
    //Calculate fitness of particle
    float fitness = 0.0f;
//    float fitness2 = 1.0f;
//    float fitness1 = 0.0f;    

// For f1
    for (int i = 0; i < D; i++)
        fitness += pos[index * D + i]*pos[index * D + i];
        
// For f2
//    for (int i = 0; i < D ; i++)
//        fitness += pos[index * D + i]*pos[index * D + i] - 10*cos(2*PI*pos[index * D + i]) +10;

// For f3
//    for (int i = 0; i < D; i++)
//        fitness1 += pos[index * D + i]*pos[index * D + i];
//        fitness2 *= cos(pos[index * D + i]/sqrt((float) index * D + j)
//    fitness = (1/4000)*fitness1 - fitness2 + 1
        
// For f4
//    for (int i = 0; i < D -1; i++)
//        fitness += (100*(pos[index * D + i + 1] - pos[index * D + i]*pos[index * D + i])*(pos[index * D + i + 1] 
//    - pos[index * D + i]*pos[index * D + i]) + (pos[index * D + i] - 1)*(pos[index * D + i] - 1)) ;
        
        
  
   
    //If fitness is better, change particle best
    if (p_best_y[index] > fitness)
    {
      p_best_y[index] = fitness;
      for (int i = 0; i < D; i++)
        p_best_pos[index * D + i] = pos[index * D + i];
    }
    personal_best = p_best_y[index];
    
    //Look up for left and right neighbours
    int left = (N + index - 1) % N;
    int right = (1 + index) % N;
    
    //Set the local best index
    if (p_best_y[left] < personal_best)
      l_best_index[index] = left;
    if (p_best_y[right] < personal_best)
      l_best_index[index] = right;
    local_best = l_best_index[index];
    
    //Compute and update particle velocity and position
    for (int i = 0; i < D; i++)
    {
      int j = index * D + i;
      r1 = curand_uniform(&local_state);
      r2 = curand_uniform(&local_state);
      
      // Compute the velocity
      velocity[j] = chi * (velocity[j] + (c_1 * r1 * (p_best_pos[j] - pos[j])) + (c_1 * r2 * (p_best_pos[local_best] - pos[j])));
      
      //Ensure velocity values are within range
      if (velocity[j] > (xmax - xmin) )
        velocity[j] = (xmax - xmin);
      if (velocity[j] < -(xmax - xmin))
        velocity[j] = -(xmax - xmin);
      
      //Update the position ensuring all values are within the xmin to xmax range
      pos[j] = pos[j] + velocity[j];
      if (pos[j] > xmax)
        pos[j] = xmax;
      if (pos[j] < xmin)
        pos[j] = xmin;
    }
    
    //Set the current state of the PRNG
    states[index] = local_state;
      
}

__global__ void ReduceKernel1(float *p_best_fitness, int* best_index) {

    // Calculate global thread index based on the block and thread indices ----

    int i = threadIdx.x + blockDim.x * blockIdx.x; 
    int tx = threadIdx.x;

    __shared__ float stage[512];
    __shared__ int best[512];
    
    best[tx] = best_index[i];
    stage[tx] = p_best_fitness[i];

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s>0; s >>=1){
    if (tx < s){
        if (stage[tx] > stage[tx + s]){
            stage[tx] = stage[tx + s];
            best[tx] = best[tx + s];
            }
        }
    __syncthreads();
    }

    if (tx == 0){
       p_best_fitness[blockIdx.x] = stage[0];
       best_index[blockIdx.x] = best[0];   
    }
    
}

__global__ void ReduceKernel2(float* p_best_pos, float *p_best_fitness, int* best_index, int D) {

    // Calculate global thread index based on the block and thread indices ----

    int i = threadIdx.x + blockDim.x * blockIdx.x; 
    int tx = threadIdx.x;
    
    __shared__ float stage[512];
    __shared__ int best[512];
    
    best[tx] = best_index[i];
    stage[tx] = p_best_fitness[i];

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s>0; s >>=1){
    if (tx < s){
        if (stage[tx] > stage[tx + s]){
            stage[tx] = stage[tx + s];
            best[tx] = best[tx + s];
            }
        }
    __syncthreads();
    }

    if (tx == 0){
       p_best_fitness[blockIdx.x] = stage[0];
       best_index[blockIdx.x] = best[0];   
    }

    for (int j = 0; j < D; j++){
        p_best_pos[j] = p_best_pos[best[0] * D + j];}
     
   
}
