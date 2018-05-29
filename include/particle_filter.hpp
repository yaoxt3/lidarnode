//
// Created by yxt on 18-5-29.
//

#ifndef PROJECT_PARTICLE_FILTER_H
#define PROJECT_PARTICLE_FILTER_H

#include "../src/lidar_main.cpp"
#include "../../../../../../usr/include/pcl-1.7/pcl/impl/point_types.hpp"

struct particle{
	double x;
	double y;
	double z;
	pcl::PointCloud<pcl::PointXYZI> observed_value;
};

class ParticleFilter{
private:

};
#endif //PROJECT_PARTICLE_FILTER_H
