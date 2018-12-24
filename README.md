# Overview
The repository contains my implementation of Particle Filters, as part of the Kidnapped Vehicle project for the Udacity's Self-Driving Car Nanodegree.



## Project Introduction
Your robot has been kidnapped and transported to a new location! Luckily it has a map of this location, a (noisy) GPS estimate of its initial location, and lots of (noisy) sensor and control data.

In this project you will implement a 2 dimensional particle filter in C++. Your particle filter will be given a map and some initial localization information (analogous to what a GPS would provide). At each time step your filter will also get observation and control data.


## My Goals
While it would be fairly easy to implement Particle Filters just to make it run and pass, I aim to invest a bit more, to achieve the following:
* Finally make a portfolio entry that showcases my C++ skills. While I have many projects in Python, my [Portfolio Website](https://linasko.github.io/portfolio/) is still missing one in C++.
* Practice using the standard C++ algorithms library. In the past I have worked with functional programming in Python and Haskell, but do not have provable experience of it in C++. More importantly, I currently have doubts on how useful / readable it is and need to get my hands dirty to decide.
* (Optionally) recode my implementation in Eigen. I absolutely love vectorization in Python, but do not have much experience of it in C++.


## Project Solution
The top level controller of the particle filter steps was provided in `src/main.cpp`. The implementation details were left for us to complete and can be found in `src/particle_filter.h` and `src/particle_filter.cpp`. If in doubt about my contribution, feel free to check the commit history.


## Shortcomings
In light of my goals, here are the obvious shortcomings that I see in my implementation:
* **Lack of error handling**. To keep the code concise I skipped on a fair bit of error handling. Normally, I'd have more asserts or make sure everything's airtight with unit tests.
* **Nearest-Neighbour search not optimized** The code can be made faster by using sensor range provided to us specifically to search for landmark-observation associations in a bounded window. However, this sacrifices code clarity, as well as not being the best solution here. An even better approach would be to also use K-D trees to store the landmarks, speeding the seach from roughly O(n<sup>2</sup>) to O(log(n)), especially when the map never changes.


## Success Criteria
The particle filter has to pass accuracy and time requirements. Both are specified in the simulator, which notifies if the filter is accepted.

The things the grading code is looking for are:

1. **Accuracy**: the particle filter should localize vehicle position and yaw to within the values specified ~~in the parameters `max_translation_error` and `max_yaw_error` in `src/main.cpp`~~ in the simulator source code.

2. **Performance**: your particle filter should complete execution within the time of 100 seconds.

Effectively, the task is to build out the methods in `particle_filter.cpp` until the simulator output says:
```
Success! Your particle filter passed!
```



## Running the Code
This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

This repository includes two files that can be used to set up and install uWebSocketIO for either Linux or Mac systems. For windows you can use either Docker, VMware, or even Windows 10 Bash on Ubuntu to install uWebSocketIO. However, before you can use these, please install `cmake`, in whichever way is best for your system.

Once the install for uWebSocketIO is complete, the main program can be built and ran by doing the following from the project top directory.

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./particle_filter

Alternatively some scripts have been included to streamline this process, these can be leveraged by executing the following in the top directory of the project:

1. ./clean.sh
2. ./build.sh
3. ./run.sh



## Communication with the Simulator
Here is the main protocol that main.cpp uses for uWebSocketIO in communicating with the simulator.

INPUT: values provided by the simulator to the c++ program

// sense noisy position data from the simulator

["sense_x"]

["sense_y"]

["sense_theta"]

// get the previous velocity and yaw rate to predict the particle's transitioned state

["previous_velocity"]

["previous_yawrate"]

// receive noisy observation data from the simulator, in a respective list of x/y values

["sense_observations_x"]

["sense_observations_y"]


OUTPUT: values provided by the c++ program to the simulator

// best particle values used for calculating the error evaluation

["best_particle_x"]

["best_particle_y"]

["best_particle_theta"]

//Optional message data used for debugging particle's sensing and associations

// for respective (x,y) sensed positions ID label

["best_particle_associations"]

// for respective (x,y) sensed positions

["best_particle_sense_x"] <= list of sensed x positions

["best_particle_sense_y"] <= list of sensed y positions



## Inputs to the Particle Filter
You can find the inputs to the particle filter in the `data` directory.


#### The Map
`map_data.txt` includes the position of landmarks (in meters) on an arbitrary Cartesian coordinate system. Each row has three columns
1. x position
2. y position
3. landmark id


#### All other data the simulator provides, such as observations and controls.

> * Map data provided by 3D Mapping Solutions GmbH.
