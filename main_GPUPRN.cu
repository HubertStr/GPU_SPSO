// Let this script use GPU generated random numbers


#include <stdio.h>
#include "support.h"
#include "kernel.cu"
#include <curand.h>
#include <curand_kernel.h>

#define dim1 50                     //number of dimensions in fitness function
#define dim2 100                    //number of dimensions in fitness function
#define dim3 150                    //number of dimensions in fitness function
#define dim4 200                    //number of dimensions in fitness function

#define x_min_f1 -100                //minimum x
#define x_max_f1 100                //minimum x
#define x_min_f2_f4 -10                //minimum x
#define x_max_f2_f4 10                //minimum x
#define x_min_f3 -600                //minimum x
#define x_max_f3 600                //minimum x

#define max_particles1 400
#define max_particles2 1200
#define max_particles3 2000
#define max_particles4 2800


#define max_iters 2048              //number of iterations

#define chi 0.72984f                //chi (constriction factor)
#define pi 3.14159265f
#define inf 9999.99f                //infinity


