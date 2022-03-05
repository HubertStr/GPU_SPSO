//Kernel to Scale velocity and position values to be between the max and min (and not between 0 and 1 anymore)
// AND initialize particle best (for each particle) + local best 

__global__void Scale_Init(int *xmax, int *xmin, float *pos, float *velocity, float *p_best_y, int *l_best_index, int *best_index, curandState *states)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int t_index = threadIdx.x;

    //Rescale pos between xmin and xmax
    pos[index] = x_max * (2.0f * pos[index] - 1.0f);

    //Rescale velocity
    velocity[index] =(x_max - x_min) * (2.0f * velocity[index] - 1.0f);
    
    //Set PBest to infinity and LBest to self
    //Initialize array of best indices
    if (t_index == 0)
    {
        p_best_y[blockIdx.x] = inf;
        l_best_index[blockIdx.x] = blockIdx.x;
        best_index[blockIdx.x] = blockIdx.x;
    }

    //Initializing up cuRAND
    //Each thread gets a different seed, different sequence number and no offset
    curand_init(index, index, 0, &states[index]);
}
// 1 kernel to compute the actual iterations of the updates 

// 1 Kernel to compute the global best after the iterations

