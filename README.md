# GPU_SPSO
GPU-based Parallel Particle Swarm Optimization

## Link of the paper
Our project is based on the following paper:<br/>
![Link to paper](https://ieeexplore.ieee.org/document/4983119).

## Objective functions:
f1:![This is an image](./images/f1.png).

f2:![This is an image](./images/f2.png).

f3:![This is an image](./images/f3.png).

f4:![This is an image](./images/f4.png).


## Constraints
#### Dimensions
In this project, we will look at 4 dimensions:<br/>
50, 100, 150, 200

#### Swarm population
In this project, we will look at 4 dimensions:<br/>
400, 1200, 2000, 2800

#### Domains
f1: (-100, 100)^D<br/>
f2: (-10, 10)^D<br/>
f3: (-600, 600)^D<br/>
f4: (-10, 10)^D<br/>

## First step: Random number generation
Two methods will be implemented:<br/>
- [ ] Method1: M (M >> D∗N) random numbers are generated on CPU before running SPSO. Then they are transported to GPU once for ado and stored in an array R on the global memory.<br/>
- [ ] Method2: Uses cuRAND to generate pseudorandom numbers on the GPU.<br/>    
