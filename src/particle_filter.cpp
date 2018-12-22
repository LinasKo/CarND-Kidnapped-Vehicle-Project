/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <tuple>
#include <type_traits>

#include "particle_filter.h"

using namespace std;


constexpr size_t PARTICLE_COUNT {100u};
constexpr double INIT_WEIGHT {1.0};

void ParticleFilter::init(double x, double y, double theta, array<double, 3> std)
{
	m_particles.reserve(PARTICLE_COUNT);
	for (size_t i = 0; i < PARTICLE_COUNT; ++i)
	{
		m_particles.push_back({
			static_cast<int>(i),
			drawFromNormal(x, std[0]),
			drawFromNormal(y, std[1]),
			fmod(drawFromNormal(theta, std[2]), 2 * M_PI),
			INIT_WEIGHT,
			{},
			{},
			{}
		});
	}

	m_is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, array<double, 3> std_pos, double velocity, double yaw_rate)
{
	if (yaw_rate == 0)
	{
		for_each(m_particles.begin(), m_particles.end(), [=](Particle& part)
		{
			part.x += drawFromNormal(velocity * delta_t * cos(part.theta), std_pos[0]);
			part.y += drawFromNormal(velocity * delta_t * sin(part.theta), std_pos[1]);
			// Yaw (theta) remains the same
		});
	}
	else
	{
		for_each(m_particles.begin(), m_particles.end(), [=](Particle& part)
		{
			part.x += drawFromNormal(velocity / yaw_rate * (sin(part.theta + yaw_rate * delta_t) - sin(part.theta)), std_pos[0]);
			part.y += drawFromNormal(-velocity / yaw_rate * (cos(part.theta + yaw_rate * delta_t) - cos(part.theta)), std_pos[1]);
			part.theta += drawFromNormal(yaw_rate * delta_t, std_pos[2]);
			part.theta = fmod(part.theta, 2 * M_PI);
		});
	}

}

void ParticleFilter::updateWeights(double sensor_range, array<double, 2> std_landmark, const std::vector<LandmarkObs>& observations, const Map& map_landmarks)
{
	if (observations.size() == 0)
	{
		return;
	}

	for (auto& part : m_particles)
	{
		part.associations.clear();
		part.sense_x.clear();
		part.sense_y.clear();

		for (const auto& obs : observations)
		{
			double xObsMap, yObsMap;
			tie(xObsMap, yObsMap) = vehicleToMapCoord(part.x, part.y, part.theta, obs.x, obs.y);

			// I can account for sensor range here if more speed is needed.
			const auto& closestLandmark = *min_element(map_landmarks.landmark_list.begin(), map_landmarks.landmark_list.end(),
				[xObsMap, yObsMap](const Map::single_landmark_s& landmark1, const Map::single_landmark_s& landmark2)
			{
				return dist(xObsMap, yObsMap, landmark1.x_f, landmark1.y_f) < dist(xObsMap, yObsMap, landmark2.x_f, landmark2.y_f);
			});

			part.associations.push_back(closestLandmark.id_i);
			part.sense_x.push_back(closestLandmark.x_f);
			part.sense_y.push_back(closestLandmark.y_f);

			part.weight *= multivariateGaussian(xObsMap, yObsMap, closestLandmark.x_f, closestLandmark.y_f, std_landmark[0], std_landmark[1]);;
		}
	}
}

void ParticleFilter::resample()
{
	// Collect weights
	vector<double> weights;
	weights.reserve(m_particles.size());
	transform(m_particles.begin(), m_particles.end(), back_inserter(weights), [](const Particle& part){
		return part.weight;
	});

	// Resample
	discrete_distribution<int> distribution {weights.begin(), weights.end()};
	
	std::vector<Particle> newParticles;
	newParticles.reserve(m_particles.size());
	for (size_t i = 0; i < m_particles.size(); ++i)
	{
		newParticles.push_back(
			m_particles[distribution(m_generator)]
		);
	}
	m_particles = newParticles;

	// Normalize
	auto totalWeight = accumulate(m_particles.begin(), m_particles.end(), 0.0, [](double base, const Particle& part){
		return base + part.weight;
	});
	for_each(m_particles.begin(), m_particles.end(), [totalWeight](Particle& part){
		part.weight /= totalWeight;
	});
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

template <typename T>
T ParticleFilter::drawFromNormal(T mean, T std)
{
	normal_distribution<T> distribution(mean, std);
	return distribution(m_generator);
}

pair<double, double> ParticleFilter::vehicleToMapCoord(double x_part, double y_part, double theta_part, double x_obs, double y_obs)
{
	double x_map = x_part + (cos(theta_part) * x_obs) - (sin(theta_part) * y_obs);
	double y_map = y_part + (sin(theta_part) * x_obs) + (cos(theta_part) * y_obs);
	return make_pair(x_map, y_map);
}

double ParticleFilter::multivariateGaussian(double x, double y, double mean_x, double mean_y, double std_dev_x, double std_dev_y)
{

	double norm = 1.0 / (2 * M_PI * std_dev_x * std_dev_y);
	double exponent = pow(x - mean_x, 2) / (2 * pow(std_dev_x, 2)) +
					  pow(y - mean_y, 2) / (2 * pow(std_dev_y, 2));

	return norm * exp(-exponent);
}
