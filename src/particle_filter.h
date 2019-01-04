/*
 * particle_filter.h
 *
 * 2D particle filter class.
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include <array>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "helper_functions.h"


constexpr size_t PARTICLE_COUNT {100u};

class ParticleFilter
{
	public:
		/**
		 * init Initializes particle filter by initializing particles to Gaussian
		 *   distribution around first position and all the weights to 1.
		 * @param x Initial x position [m] (simulated estimate from GPS)
		 * @param y Initial y position [m]
		 * @param theta Initial orientation [rad]
		 * @param std Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
		 *   standard deviation of yaw [rad]]
		 */
		void init(double x, double y, double theta, std::array<double, 3> std);

		/**
		 * prediction Predicts the state for the next time step
		 *   using the process model.
		 * @param delta_t Time between time step t and t+1 in measurements [s]
		 * @param std_pos Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
		 *   standard deviation of yaw [rad]]
		 * @param velocity Velocity of car from t to t+1 [m/s]
		 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
		 */
		void prediction(double delta_t, std::array<double, 3> std_pos, double velocity, double yaw_rate);
		
		/**
		 * updateWeights Updates the weights for each particle based on the likelihood of the 
		 *   observed measurements. 
		 * @param sensor_range Range [m] of sensor
		 * @param std_landmark Array of dimension 2 [Landmark measurement uncertainty [x [m], y [m]]]
		 * @param observations Vector of landmark observations
		 * @param map Map class containing map landmarks
		 */
		void updateWeights(double sensor_range, std::array<double, 2> std_landmark, const std::vector<LandmarkObs>& observations,
						   const Map& map_landmarks);
		
		/**
		 * resample Resamples from the updated set of particles to form
		 *   the new set of particles.
		 */
		void resample();
		
		std::string getAssociations(long index);
		std::string getSenseX(long index);
		std::string getSenseY(long index);

		/**
		* initialized Returns whether particle filter is initialized yet or not.
		*/
		const bool initialized() const
		{
			return m_is_initialized;
		}

		// Setting dynamic size even if number of particles are known, as the number is large and Eigen
		// recommends using Dynamic sizes for sizes > 4.
		Eigen::MatrixX4d m_particles;

	private:
		template <typename T>
		T drawFromNormal(T mean, T std);

		/**
		 * @brief Convert vehicle coordinates to map coordinates
		 * 
		 * @param x_part x of the particle, in map coordinates
		 * @param y_part y of the particle, in map coordinates
		 * @param theta_part theta (orientation) of the particle
		 * @param x_obs x of the observation, in vehicle coordinates
		 * @param y_obs y of the observation, in vehicle coordinates
		 * @return std::pair<double, double> pair (x, y) of map coordinates of the observation
		 */
		std::pair<double, double> vehicleToMapCoord(double x_part, double y_part, double theta_part, double x_obs, double y_obs);

		double multivariateGaussian(double x, double y, double mean_x, double mean_y, double std_dev_x, double std_dev_y);

		bool m_is_initialized {false};
		std::default_random_engine m_generator;

		std::array<Eigen::MatrixX3d, PARTICLE_COUNT> m_associations;
};

#endif /* PARTICLE_FILTER_H_ */
