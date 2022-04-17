#include <stdio.h>
//#include "support.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>

#define max_iters 2048              //number of iterations
#define inf 9999.99f                //infinity


void Iterations(float xmax, float xmin, float *pos, float *velocity, float *p_best_pos,float *p_best_fitness, int *l_best_index, int *best_index, float c_1, float c_2, float inertia, float vmax, float chi, int N, int D);

int arg_min(float v[], int n);
float squareOfNumber(float num);

int main(int argc, char**argv){
  clock_t t;
  t = clock();
    
  //Timer timer;
  unsigned int N;
  unsigned int D;
  float xmax;
  float xmin;
  float c_1;
  float c_2;
  float inertia;
  float vmax;

  unsigned int obj_func = 2;

  if (argc!=9)
    {
      printf("\n     Invalid number of arguments!");
    }

  N           = atoi(argv[1]);
  D           = atoi(argv[2]);
  xmax        = atof(argv[3]);
  xmin        = atof(argv[4]);
  c_1         = atof(argv[5]);
  c_2         = atof(argv[6]);
  inertia     = atof(argv[7]);
  vmax        = atof(argv[8]);

  // Calculated variables
  float chi;
  chi = 2/fabs(2-c_1 - c_2 - sqrt((c_2+c_1)*(c_2+c_1)-(4*c_2+4*c_1)));

    
  //  MEMORY ALLOCATION

  // Allocating memory for all variables that do not come from input
  // printf("Allocating memory for all variables that do not come from input ..."); fflush(stdout);
  // startTime(&timer);

  //Particle position array
  float *particle_position; //IF ** then, you can't use the functions defined above
  particle_position = (float*)malloc( N * D * sizeof(float));

  //Velocity array
  float *particle_velocity;
  particle_velocity = (float*)malloc( N * D * sizeof(float));

  //Particle best position array
  float *p_best_pos;
  p_best_pos = (float*)malloc( N * D * sizeof(float));


  //Particle best fitness value
  float *p_best_fitness;
  p_best_fitness = (float*)malloc( N * sizeof(float));

  //Local best index value (from a fitness point of view)

  int *l_best_index;
  l_best_index = (int*)malloc( N * sizeof(int));

  //Global Best index value (from a fintess point of view)
  int *best_index;
  best_index = (int*)malloc( N * sizeof(int));
   
  //stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    
  //INITIALIZATION
    
  //Initialization of all variables...
  //printf("\nInitialization of random numbers on CPU for particle's velocity and position ... "); fflush(stdout);
  //startTime(&timer);
    for (unsigned int i=0; i < (N*D); i++) {
        particle_position[i] = ((((float) rand()) / ((float) RAND_MAX))*(xmax-xmin))-xmin;
        particle_velocity[i] = ((((float) rand()) / ((float) RAND_MAX))*(2*vmax)) - vmax;
    }
    
   //printf("\nInitialization of random numbers on CPU for particle's best index and fitness ... "); fflush(stdout);
    for (unsigned int i = 0; i <N; i++ ){
        p_best_fitness[i] = inf;
        l_best_index[i] = i;
        best_index[i] = i;
    }
    
  //stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    
  //ITERATIONS
    
  //printf("\nRunning iterations of SPSO ... "); fflush(stdout); fflush(stdout);
  //startTime(&timer);
  for (unsigned int i = 0; i < max_iters; i++){
    Iterations(xmax, xmin, particle_position, particle_velocity, p_best_pos, p_best_fitness, l_best_index, best_index,  c_1, c_2, inertia, vmax, chi, N, D);
  }
    
  //stopTime(&timer); printf("%f s\n", elapsedTime(timer));

  /*
   //Print best fintess of each particle
   for(int index = 0; index < N; index++){
      printf( "\n%f", p_best_fitness[index] );
      //printf( "\n" );
   }
  */

 
  //REDUCE
    
  //printf("\nCompute reduction"); fflush(stdout);
  //startTime(&timer);

  int a;
  a = arg_min(p_best_fitness,N);
  //a = p_best_pos[arg_min(p_best_fitness, N)] ;
  //p_best_pos[a];
  //printf("\nThe best particle index is %i", a);

  t = clock() - t;
  double time_taken = ((double)t)/CLOCKS_PER_SEC;

  //printf("\nThe process took %f seconds to execute (CPU ONLY)", time_taken);fflush(stdout);
  //printf("\nFreeing memory\n");fflush(stdout);
  
  printf("f_%i; %i; %i; %f; %f; %f; %f; %f; %f; %f \n ", obj_func, N, D, xmax, xmin, c_1, c_2, inertia, vmax, time_taken);
  //INSERT CODE HERE to free device matrices
  free(particle_position);
  free(particle_velocity);
  free(p_best_pos);
  free(p_best_fitness);
  free(l_best_index);
  free(best_index);
}

float squareOfNumber(float num)
{
  return (num*num);
}

void Iterations(float xmax, float xmin, float *pos, float *velocity, float *p_best_pos,float *p_best_fitness, int *l_best_index, int *best_index, float c_1, float c_2, float inertia, float vmax, float chi, int N, int D){

  for (int index = 0; index <N; index++){

    float r1, r2;
        
    float personal_best;
    int local_best;


    //Calculate fitness of particle
    float fitness = 0.0f;
    
    for (int i = 0; i < D; i++){
      fitness += (pos[index * D + i] * pos[index * D + i]) - cos( ( 2 * M_PI) * pos[index * D + i] ) + 10;
    }

    
    //float fitness_a = 0.0f;
    //float fitness_b = 1.0f;

    //for  (int i = 0; i < D; i++){
    //  fitness_a += squareOfNumber(pos[index * D + i]);
    //  fitness_b *= cos( pos[index * D + i]/(sqrt(i)));
    //}
    //fitness = fitness_a + fitness_b + 1;
    
    //-----F_4
    //for (int i = 0; i < D - 1; i++){
    //   fitness += squareOfNumber(100 * (pos[index * D + i + 1] - squareOfNumber(pos[index * D + i]))) + squareOfNumber(pos[index * D + i] - 1);
    //}

    //f2 = (pos[index * D +i] * pos[index * D + i]) - cos( ( 2 * M_PI) * pos[index * D + i] ) + 10;
    //f1 = pos[index * D + i] * pos[index * D +i]; //xi^2
   
    
    //If fitness is better, change particle best
    if (p_best_fitness[index] > fitness){
        p_best_fitness[index] = fitness;
        for (int i = 0; i < D; i++){
            p_best_pos[index * D + i] = pos[index * D + i];
        }
    }
    personal_best = p_best_fitness[index];
        
    //Look up for left and right neighbours
    int left = (N + index - 1) % N;
    int right = (1 + index) % N;
        
    //Set the local best index
    if (p_best_fitness[left] < personal_best)
      l_best_index[index] = left;
    if (p_best_fitness[right] < personal_best)
      l_best_index[index] = right;
    local_best = l_best_index[index];
        
    //Compute and update particle velocity and position
    for (int i = 0; i < D; i++)
      {
        int j = index * D + i;
        r1 = ((float) rand()) / ((float) RAND_MAX); //random_uniform
        r2 = ((float) rand()) / ((float) RAND_MAX);
        
        // Compute the velocity
        velocity[j] = chi * (velocity[j] + (c_1 * r1 * (p_best_pos[j] - pos[j])) + (c_1 * r2 * (p_best_pos[local_best] - pos[j])));
        
        //Ensure velocity values are within range
          if (velocity[j] > (xmax - xmin) ){
              velocity[j] = (xmax - xmin);
          };
          if (velocity[j] < -(xmax - xmin)){
              velocity[j] = -(xmax - xmin);
          };
        
        //Update the position ensuring all values are within the xmin to xmax range
        pos[j] = pos[j] + velocity[j];
          if (pos[j] > xmax){
              pos[j] = xmax;
          }
          if (pos[j] < xmin){
              pos[j] = xmin;
                
          }
      }
        
    //Set the current state of the PRNG
  }
}

int arg_min(float v[], int n){
  int i, position;
  float sval;
  sval = v[0];
  position = 0;

  for(i = 0; i < n; i++) {
    if(sval > v[i]) {
      sval = v[i];
      position = i;
    }
  }
  return position;
}
