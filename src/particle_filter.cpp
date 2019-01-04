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


constexpr double INIT_WEIGHT {1.0};

void ParticleFilter::init(double x, double y, double theta, array<double, 3> std)
{
	m_particles = Eigen::MatrixX4d(PARTICLE_COUNT, 4);

	for (auto i = 0; i < m_particles.rows(); ++i)
	{
		m_particles.row(i) <<
			drawFromNormal(x, std[0]),
			drawFromNormal(y, std[1]),
			fmod(drawFromNormal(theta, std[2]), 2 * M_PI),
			INIT_WEIGHT;
	}

	m_is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, array<double, 3> std_pos, double velocity, double yaw_rate)
{
	if (yaw_rate == 0)
	{
		Eigen::VectorXd thetas = m_particles.col(2);
		
		Eigen::VectorXd delta_x = velocity * delta_t * thetas.array().cos();
		Eigen::VectorXd delta_y = velocity * delta_t * thetas.array().sin();

		m_particles.col(0) += delta_x.unaryExpr(
			[this, &std_pos](const double& x) {
				return drawFromNormal(x, std_pos[0]);
			});

		m_particles.col(1) += delta_y.unaryExpr(
			[this, &std_pos](const double& y) {
				return drawFromNormal(y, std_pos[1]);
			});

		// Yaw (theta) remains the same
	}
	else
	{
		Eigen::VectorXd thetas = m_particles.col(2);
		
		Eigen::VectorXd delta_x =   velocity / yaw_rate * ((thetas.array() + yaw_rate * delta_t).sin() - thetas.array().sin());
		Eigen::VectorXd delta_y = - velocity / yaw_rate * ((thetas.array() + yaw_rate * delta_t).cos() - thetas.array().cos());
		Eigen::VectorXd delta_theta = Eigen::VectorXd::Constant(m_particles.rows(), yaw_rate * delta_t);

		m_particles.col(0) += delta_x.unaryExpr(
			[this, &std_pos](const double& x) {
				return drawFromNormal(x, std_pos[0]);
			});

		m_particles.col(1) += delta_y.unaryExpr(
			[this, &std_pos](const double& y) {
				return drawFromNormal(y, std_pos[1]);
			});

		m_particles.col(2) += delta_theta.unaryExpr(
			[this, &std_pos](const double& theta) {
				return drawFromNormal(theta, std_pos[2]);
			});
		m_particles.col(2) = m_particles.col(2).unaryExpr(
			[this, &std_pos](const double& theta) {
				return fmod(theta, 2 * M_PI);
			});
	}
}

void ParticleFilter::updateWeights(double sensor_range, array<double, 2> std_landmark, const std::vector<LandmarkObs>& observations, const Map& map_landmarks)
{
	if (observations.size() == 0)
	{
		return;
	}

	for (auto i = 0; i < m_particles.rows(); ++i)
	{
		Eigen::RowVector4d particle = m_particles.row(i);
		m_associations[i] = Eigen::MatrixX3d(observations.size(), 3);

		for (size_t j = 0; j < observations.size(); ++j)
		{
			const auto& obs = observations[j];

			double xObsMap, yObsMap;
			tie(xObsMap, yObsMap) = vehicleToMapCoord(particle[0], particle[1], particle[2], obs.x, obs.y);

			// I can account for sensor range here if more speed is needed.
			const auto& closestLandmark = *min_element(map_landmarks.landmark_list.begin(), map_landmarks.landmark_list.end(),
				[xObsMap, yObsMap](const Map::single_landmark_s& landmark1, const Map::single_landmark_s& landmark2)
			{
				return dist(xObsMap, yObsMap, landmark1.x_f, landmark1.y_f) < dist(xObsMap, yObsMap, landmark2.x_f, landmark2.y_f);
			});

			m_associations[i].row(j) << closestLandmark.id_i, closestLandmark.x_f, closestLandmark.y_f;

			particle(3) *= multivariateGaussian(xObsMap, yObsMap, closestLandmark.x_f, closestLandmark.y_f, std_landmark[0], std_landmark[1]);
		}

		m_particles.row(i) = particle;
	}
}

void ParticleFilter::resample()
{
	// Collect weights
	Eigen::VectorXd weights = m_particles.col(3);
	std::vector<double> weights_vector(weights.data(), weights.data() + weights.size());

	// Resample
	discrete_distribution<int> distribution {weights_vector.begin(), weights_vector.end()};
	
	auto oldParticles = m_particles;
	for (auto i = 0; i < m_particles.rows(); ++i)
	{
		m_particles.row(i) = oldParticles.row(distribution(m_generator));
	}

	// Normalize
	double totalWeight = m_particles.col(3).sum();
	if (totalWeight == 0)
	{
		m_particles.col(3).array() += INIT_WEIGHT;
	}
	else
	{
		m_particles.col(3) /= totalWeight;
	}

	// Remap to standard particles, to not break stuff.
	m_stlParticles.clear();
	m_stlParticles.reserve(m_particles.rows());
	for (auto i = 0; i < m_particles.rows(); ++i)
	{
		m_stlParticles.push_back(
			{
				m_particles(i, 0),
				m_particles(i, 1),
				m_particles(i, 2),
				m_particles(i, 3)
			}
		);

		Eigen::VectorXi idCol = m_associations[i].col(0).cast<int>();
		Eigen::VectorXd xCol = m_associations[i].col(1);
		Eigen::VectorXd yCol = m_associations[i].col(2);

		m_stlParticles[i].associations = vector<int>(idCol.data(), idCol.data() + idCol.size());
		m_stlParticles[i].sense_x = vector<double>(xCol.data(), xCol.data() + idCol.size());
		m_stlParticles[i].sense_y = vector<double>(yCol.data(), yCol.data() + idCol.size());
	}
}

string ParticleFilter::getAssociations(long index)
{
	vector<int> v = m_stlParticles[index].associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(long index)
{
	vector<double> v = m_stlParticles[index].sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(long index)
{
	vector<double> v = m_stlParticles[index].sense_y;
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
