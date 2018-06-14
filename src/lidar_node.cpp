//
// Created by yxt on 18-6-5.
//
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <vector>
#include <ctime>
#include <sstream>
#include <algorithm>
#include <ros/ros.h>
#include <geometry_msgs/Vector3.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/octree/octree.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_ros/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/cloud_viewer.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
using namespace std;

/*
 * @x,y,z: the current (x,y,z) coordinate
 * @px,py,pz: the previous (x,y,z) coordinate
 * @x0,y0,z0: the original (x,y,z) coordinate
 * @height,width,longth: the original (height,width,longth) of object tracked by particle
 * @likelihood:the similarity between the current object described by particle and the original object
 */
struct particle{
	double x;
	double y;
	double z;
	double px;
	double py;
	double pz;
	double x0;
	double y0;
	double z0;
	double width;  // x
	double height; // y
	double longth; // z
	double likelihood;
	pcl::PointCloud<pcl::PointXYZI> observed_value;

	particle(){
		x = y = z = 0.0;
		px = py = pz = 0.0;
		x0 = y0 = z0 = 0.0;
		height = width = longth = 0.0;
		likelihood = 0.0;
		observed_value.clear();
	}
	particle(const particle& one) {
		x = one.x; y = one.y; z = one.z;
		px = one.px; py = one.py; pz = one.pz;
		x0 = one.x0; y0 = one.y0; z0 = one.z0;
		height = one.height;
		width = one.width;
		longth = one.longth;
		likelihood = one.likelihood;
		observed_value = one.observed_value;
	}
	particle& operator =(const particle& one){
		x = one.x; y = one.y; z = one.z;
		px = one.px; py = one.py; pz = one.pz;
		x0 = one.x0; y0 = one.y0; z0 = one.z0;
		height = one.height;
		width = one.width;
		longth = one.longth;
		likelihood = one.likelihood;
		observed_value = one.observed_value;
	}
};

/*
 * @initialParticle: initialize particle's state.
 * @transition: update the current particle's state by previous state.
 * @normalizeWeights: normalize particle's weights.
 * @resample: resampling the particles to keep the diversity of particles.
 * @getLikelihood: calculate the similarity between the particle's observed value and the tracked object.
 * @compareWeight: compare funtion for sort algorithm in descending order.
 * @objectid: the tracked object id.
 * @std_x,std_y,std_z: standard deviation.
 * @A0,A1,B: coefficient of transition function.
 * @MAX_PARTICLE_NUM: maximum partilce number.
 * @particles: particle set
 * @rng: gsl library variable to generate guassian-distribution number
 */
class ParticleFilter{
public:
	ParticleFilter();
	~ParticleFilter();
	void initialParticle(const pcl::PointCloud<pcl::PointXYZI> *points);
	void transition();
	void normalizeWeights();
	void resample();
//	void getLikelihood(const pcl::search::KdTree<pcl::PointXYZI> *kdtree,const pcl::PointCloud<pcl::PointXYZI> *pointset, const pcl::PointCloud<pcl::PointXYZI> *mypoint);
	void getLikelihood(const pcl::search::KdTree<pcl::PointXYZI> *kdtree,const pcl::PointCloud<pcl::PointXYZI> *pointset,const pcl::PointCloud<pcl::PointXYZI> *mypoint);
	void updateParticle(const particle *point,const pcl::PointXYZ *center_point, const pcl::PointXYZ size);
	pcl::PointXYZ getPosition();
	void printAllParticle();
	void printThisParticle(int);
	int objectid;
	double std_x,std_y,std_z;
	double A0,A1,B;
	static const int MAX_PARTICLE_NUM = 40;
	static const int MAX_INTENSITY = 300;
	particle *particles;
	gsl_rng *rng;
};

ParticleFilter::ParticleFilter(){
	objectid = 0;
	std_x = 0.55;
	std_y = 0.45;
	std_z = 0.35;
	A0 = 2.0;
	A1 = -1.0;
	B = 1.0;
	particles = new particle[MAX_PARTICLE_NUM];
	gsl_rng_env_setup();
	rng = gsl_rng_alloc(gsl_rng_mt19937);
	gsl_rng_set(rng,time(NULL));
}

ParticleFilter::~ParticleFilter()
{
	gsl_rng_free(rng);
}

double Max(double a, double b)
{
	return a >= b ? a : b;
}

double Min(double a, double b)
{
	return a <= b ? a : b;
}

void ParticleFilter::initialParticle(const pcl::PointCloud<pcl::PointXYZI> *points) {
	double maxWidth=-10000.0, minWidth=10000.0;
	double maxHeight=-10000.0, minHeight=10000.0;
	double maxLongth=-10000.0, minLongth=10000.0;
	double mean_x=0.0, mean_y=0.0, mean_z=0.0;
	for (int i = 0; i < points->size(); ++i) {
		maxWidth = Max(maxWidth,points->points[i].x);
		maxLongth = Max(maxLongth, points->points[i].y);
		maxHeight = Max(maxHeight,points->points[i].z);
		minWidth = Min(minWidth,points->points[i].x);
		minLongth = Min(minLongth, points->points[i].y);
		minHeight = Min(minHeight,points->points[i].z);
		mean_x += points->points[i].x;
		mean_y += points->points[i].y;
		mean_z += points->points[i].z;
	}
	// initilize particle's position
	for (int j = 0; j < MAX_PARTICLE_NUM; ++j) {
		particles[j].width = maxWidth - minWidth;
		particles[j].height = maxHeight - minHeight;
		particles[j].longth = maxLongth - minLongth;
		particles[j].x0 = mean_x / points->size();
		particles[j].y0 = mean_y / points->size();
		particles[j].z0 = mean_z / points->size();
		particles[j].x = particles[j].x0;
		particles[j].y = particles[j].y0;
		particles[j].z = particles[j].z0;
		particles[j].px = particles[j].x;
		particles[j].py = particles[j].y;
		particles[j].pz = particles[j].z;
		particles[j].likelihood = 0.0;
	}
}

/*
 * x[t+1] - x[t] = x[t] - x[t-1] + N(0,1)
 * x[t+1] = 2x[t] - x[t-1] + N(0,1)
 */
void ParticleFilter::transition() {
	for (int i = 0; i < MAX_PARTICLE_NUM; ++i) {
		particle p = particles[i];
		double next_x = A0*(p.x-p.x0)+A1*(p.px-p.x0)+B*gsl_ran_gaussian(rng,std_x)+p.x0;
		double next_y = A0*(p.y-p.y0)+A1*(p.py-p.y0)+B*gsl_ran_gaussian(rng,std_y)+p.y0;
		double next_z = A0*(p.z-p.z0)+A1*(p.pz-p.z0)+B*gsl_ran_gaussian(rng,std_z)+p.z0;

		particles[i].px = p.x;
		particles[i].py = p.y;
		particles[i].pz = p.z;
		particles[i].x = next_x;
		particles[i].y = next_y;
		particles[i].z = next_z;
		particles[i].x0 = p.x0;
		particles[i].y0 = p.y0;
		particles[i].z0 = p.z0;
		particles[i].longth = p.longth;
		particles[i].width = p.width;
		particles[i].height = p.height;
		particles[i].likelihood = 0.0;
		particles[i].observed_value.clear();
	}
}

//void ParticleFilter::getLikelihood(const pcl::search::KdTree<pcl::PointXYZI> *kdtree, const pcl::PointCloud<pcl::PointXYZI> *pointset, const pcl::PointCloud<pcl::PointXYZI> *mypoint) {
void ParticleFilter::getLikelihood(const pcl::search::KdTree<pcl::PointXYZI> *kdtree,const pcl::PointCloud<pcl::PointXYZI> *pointset,const pcl::PointCloud<pcl::PointXYZI> *mypoint) {
	cout << "likelihood function." << endl;
	for (int i = 0; i < MAX_PARTICLE_NUM; ++i) {
//		cout << endl;
//		cout << "particle: " << i << endl;
		pcl::PointCloud<pcl::PointXYZI> pfpoint; // observed points at the current position
		vector<int> pointRadiusSearch;
		vector<float> pointRadiusSquareDistance;
		pfpoint.clear();
		pointRadiusSearch.clear();
		pointRadiusSquareDistance.clear();
//		vector<int> (pointRadiusSearch).swap(pointRadiusSearch);
//		vector<float> (pointRadiusSquareDistance).swap(pointRadiusSquareDistance);
//		cout << "radius capacity:" << pointRadiusSearch.capacity() << endl;
//		cout << "dis capacity:" << pointRadiusSquareDistance.capacity() << endl;
		pcl::PointXYZI point0;
		point0.x = particles[i].x;
		point0.y = particles[i].y;
		point0.z = particles[i].z;
		const pcl::PointXYZI point = point0;
		float radius;
		float width = particles[i].width;
		float height = particles[i].height;
		float longth = particles[i].longth;
//		cout << width << " " << height << " " << longth << endl;
//		cout << "(" << point.x << "," << point.y << "," << point.z << ")" << endl;
		radius = 0.5 * sqrt(width*width + height*height + longth*longth);
//		cout << "radius: " << radius << endl;

		int result = kdtree->radiusSearch(point,radius,pointRadiusSearch,pointRadiusSquareDistance,0);
//		cout << result << endl;
		if(result > 0){
//			cout << "in if section." << endl;
//			cout << "num: " << pointRadiusSearch.size() << endl;
			for (int j = 0; j < pointRadiusSearch.size(); ++j) {
				//cout << "kdtree_for:" << j << endl;
				double dx = abs(pointset->points[pointRadiusSearch[j]].x - point.x);
				double dy = abs(pointset->points[pointRadiusSearch[j]].y - point.y);
				double dz = abs(pointset->points[pointRadiusSearch[j]].z - point.z);
				//cout <<"("<< dx << "," << dy << "," << dz <<")"<< endl;
				//cout << "(" << pointset->points[pointRadiusSearch[j]].x << "," << pointset->points[pointRadiusSearch[j]].y <<","<< pointset->points[pointRadiusSearch[j]].z << ")" << endl;
				if(dx > 0.5*width || dy > 0.5*height || dz > 0.5*longth){
					continue;
				}
				else{
//					cout << "push_back." << endl;
					pcl::PointXYZI points = pointset->points[pointRadiusSearch[j]];
					pfpoint.push_back(points);
				}
			}
			if(pfpoint.size() > 0){
//				cout << "!!" << endl;
				int maxSize = max(mypoint->size(),pfpoint.size());
//				cout << maxSize << " " << mypoint->size() << " " << pfpoint.size() << endl;
				int *object_intensity = new int[MAX_INTENSITY];
				int *pf_intensity = new int[MAX_INTENSITY];
				for (int ii = 0; ii < MAX_INTENSITY; ++ii) {
					object_intensity[ii] = 0;
					pf_intensity[ii] = 0;
				}

				for (int j = 0; j < mypoint->size(); ++j) {
					int intensity = round(mypoint->points[j].intensity);
					object_intensity[intensity] = object_intensity[intensity] + 1;
				}
				for (int k = 0; k < pfpoint.size(); ++k) {
					int intensity = round(pfpoint.points[k].intensity);
					pf_intensity[intensity] = pf_intensity[intensity] + 1;
				}

//				cout << "@@" << endl;
				//normalization
				float *fobject_intensity = new float[MAX_INTENSITY];
				float *fpf_intensity = new float[MAX_INTENSITY];
				for (int l = 0; l < MAX_INTENSITY; ++l) {
					fobject_intensity[l] = object_intensity[l]*1.0/mypoint->size();
					fpf_intensity[l] = pf_intensity[l]*1.0/pfpoint.size();
				}

				// calculate the similarity by point number and intensity
				float numWeight = 0.0;
				float intensityWeight = 0.0;
				float similarity = 0.0;
				for (int m = 0; m < MAX_INTENSITY; ++m) {
					intensityWeight = intensityWeight + sqrt(fobject_intensity[m]*fpf_intensity[m]);
				}
//				cout << "Bdistance: " << intensityWeight << endl;
				intensityWeight = 1 - intensityWeight;
				intensityWeight = exp(-1.0*intensityWeight);

//				cout << "intensityWeight: " << intensityWeight << endl;
				numWeight = pfpoint.size()*1.0/mypoint->points.size();
				similarity = numWeight * intensityWeight;

				/*
				 * need consider more methods to calculate likelihood, for example:
				 * compare particle's point distribution and object's point distribution.
				 */

				particles[i].likelihood = similarity;
				particles[i].observed_value = pfpoint;
//				cout << "likelihood: " << similarity << endl;
				delete []pf_intensity;
				delete []object_intensity;
				delete []fpf_intensity;
				delete []fobject_intensity;
				vector<int> ().swap(pointRadiusSearch);
				vector<float> ().swap(pointRadiusSquareDistance);

			}
		} else{
			particles[i].likelihood = 0.0;
		}

	}
	cout << "end likelihood." << endl;
}

void ParticleFilter::printAllParticle() {
	for (int i = 0; i < MAX_PARTICLE_NUM; ++i) {
		cout << "particle " << i << ":" << endl;
		cout << "current position:";
		cout << "(" << particles[i].x << "," << particles[i].y << "," << particles[i].z << ")" << endl;
		cout << "likelihood:" << particles[i].likelihood << endl;
		cout << "original position:";
		cout << "(" << particles[i].x0 << "," << particles[i].y0 << "," << particles[i].z0 << ")" << endl;
	}
}

void ParticleFilter::printThisParticle(int i) {
	cout << "particle " << i << ":" << endl;
	cout << "current position:";
	cout << "(" << particles[i].x << "," << particles[i].y << "," << particles[i].z << ")" << endl;
	cout << "likelihood:" << particles[i].likelihood << endl;
	cout << "original position:";
	cout << "(" << particles[i].x0 << "," << particles[i].y0 << "," << particles[i].z0 << ")" << endl;
	cout << "width:" << particles[i].width << " longth:" << particles[i].longth << " height:" << particles[i].height << endl;
}

void ParticleFilter::updateParticle(const particle *myparticle,const pcl::PointXYZ *center_point, const pcl::PointXYZ size) {

	for (int i = 0; i < MAX_PARTICLE_NUM; ++i) {
		particles[i].x = center_point->x;
		particles[i].y = center_point->y;
		particles[i].z = center_point->z;
		particles[i].px = myparticle->x;
		particles[i].py = myparticle->y;
		particles[i].pz = myparticle->z;
		particles[i].x0 = myparticle->x0;
		particles[i].y0 = myparticle->y0;
		particles[i].z0 = myparticle->z0;
		particles[i].width = size.x;
		particles[i].longth = size.y;
		particles[i].height = size.z;
	}

}

void ParticleFilter::normalizeWeights() {
	double sum = 0.0;
	for (int i = 0; i < MAX_PARTICLE_NUM; ++i) {
		sum += particles[i].likelihood;
	}
	for (int j = 0; j < MAX_PARTICLE_NUM; ++j) {
		particles[j].likelihood = particles[j].likelihood / sum;
	}
}

struct sortparticle{
	int particleid;
	double likelihood;
	sortparticle(){
		particleid = 0;
		likelihood = 0.0;
	}
	sortparticle& operator =(const sortparticle &spone){
		particleid = spone.particleid;
		likelihood = spone.likelihood;
	}
};

bool compareWeight(const sortparticle &a,const sortparticle &b){
	return a.likelihood >= b.likelihood;
}

void ParticleFilter::resample() {
	int number = 0;
	int count = 0;
	sortparticle *cparticles = new sortparticle[MAX_PARTICLE_NUM];
	for (int l = 0; l < MAX_PARTICLE_NUM; ++l) {
		cparticles[l].particleid = l;
		cparticles[l].likelihood = particles[l].likelihood;
//		cout << "particle " << l << " likelihood " << cparticles[l].likelihood << endl;
	}
	// sort in descending order
	for (int m = 0; m < MAX_PARTICLE_NUM; ++m) {
		sortparticle temp = cparticles[m];

		int id = m;
		bool flag = false;
		for (int n = m+1; n < MAX_PARTICLE_NUM; ++n) {
			if(cparticles[n].likelihood > temp.likelihood){
				id = n;
				temp = cparticles[n];
				flag = true;
			}
		}
		if(flag==true){
			cparticles[id] = cparticles[m];
			cparticles[m] = temp;
		}
	}
//	for (int k = 0; k < MAX_PARTICLE_NUM; ++k) {
//		cout <<"particle " <<  k << ":" << endl;
//		cout << cparticles[k].particleid << " " << cparticles[k].likelihood << endl;
//		cout << "----------------" << endl;
//	}

	// replicating particles according to likelihood(weight)
	sortparticle *tmp = new sortparticle[MAX_PARTICLE_NUM];
	for (int i = 0; i < MAX_PARTICLE_NUM; ++i) {
		number = round(cparticles[i].likelihood * MAX_PARTICLE_NUM);
		for (int j = 0; j < number; ++j) {
			tmp[count++] = cparticles[i];
			if(count == MAX_PARTICLE_NUM-1)
				break;
		}
		if(count == MAX_PARTICLE_NUM-1)
			break;
	}

	while(count < MAX_PARTICLE_NUM){
		tmp[count] = cparticles[0];
		count++;
	}

	particle *tmpf = new particle[MAX_PARTICLE_NUM];
	for (int k = 0; k < MAX_PARTICLE_NUM; ++k) {
		tmpf[k] = particles[k];
	}
	for (int i = 0; i < MAX_PARTICLE_NUM; ++i) {
		particles[i] = tmpf[tmp[i].particleid];
	}

	/*
	for (int j = 0; j < MAX_PARTICLE_NUM; ++j) {
		cout << "particle " << j << ": " << endl;
		cout << "(" << particles[j].x << "," << particles[j].y << "," << particles[j].z << ")" << endl;
		cout << "likelihood :" << particles[j].likelihood << endl;
		cout << "point size: " << particles[j].observed_value.size() << endl;
		for (int i = 0; i < particles[j].observed_value.size(); ++i) {
			cout << "(" << particles[j].observed_value[i].x << "," << particles[j].observed_value[i].y << "," << particles[j].observed_value[i].z << ")" << endl;
		}
		cout << "-------------------" << endl;
	}
	*/
	delete []cparticles;
	delete []tmp;
	delete []tmpf;
	cout << "end sample." << endl;
}

pcl::PointXYZ ParticleFilter::getPosition() {
	pcl::PointXYZ point;
	point.x = 0.0;
	point.y = 0.0;
	point.z = 0.0;
	for (int i = 0; i < MAX_PARTICLE_NUM; ++i) {
		point.x += particles[i].x;
		point.y += particles[i].y;
		point.z += particles[i].z;
	}
	point.x /= MAX_PARTICLE_NUM;
	point.y /= MAX_PARTICLE_NUM;
	point.z /= MAX_PARTICLE_NUM;
	return point;
}

/*
 * @cluster_xyz: the center position of the pointcloud cluster
 * @height: max(z) - min(z)
 * @width:  max(x) - min(x)
 * @longth: max(y) - min(y)
 * @pf:create 30 particle objects for cluster
 * calculate the moment invariant of the cluster by (height,width,longth)
 */
struct cluster_info{
	double center_x;
	double center_y;
	double center_z;
	double height;
	double width;
	double longth;
	ParticleFilter *pf;
	pcl::PointCloud<pcl::PointXYZI> points;
	cluster_info(){
		pf = new ParticleFilter;
		center_x = 0.0;
		center_y = 0.0;
		center_z = 0.0;
		height = 0.0;
		width = 0.0;
		longth = 0.0;
		points.clear();
	}
};

/*
 * @point_cluster_num: the point cluster number
 * @cluster: store all pointcloud clusters in the current frame, and differentiate pointcloud clusters by different intensities
 */
struct frame_info{

	int point_cluster_num;
	cluster_info *cluster;
	pcl::PointCloud<pcl::PointXYZI> allpoints;
	frame_info(){
		cluster = new cluster_info;
		allpoints.clear();
		point_cluster_num = 0;
	}
};

class Lidar_node{
public:
    Lidar_node();
    // function
    void processPointCloud(const sensor_msgs::PointCloud2 &scan);
    void TrackingModel(const pcl::PointCloud<pcl::PointXYZI> *pointset);
	float calculate_distance2(const pcl::PointXYZ a, const pcl::PointXYZ b);
private:
    ros::NodeHandle node_handle_;
    ros::Subscriber points_node_sub_;
    ros::Publisher test_points_pub_;
    ros::Publisher points_node_pub_;
	int frame_id;
    const int frame_num;
    const int searchNum;
    float x_min;
    float x_max;
    float y_min;
    float y_max;
    float z_min;
    float z_max;
	vector<frame_info> frame_points;
};

Lidar_node::Lidar_node():searchNum(100),frame_num(3){ // error : node_handle_("~")
    ROS_INFO("In constructed function.");
    x_min = 0;
    x_max = 6;
    y_min = -3.06;
    y_max = 4;
    z_min = -0.55;
    z_max = 1.45;
	frame_id = 0;
	frame_points.clear();
    points_node_sub_ = node_handle_.subscribe("velodyne_points", 1028, &Lidar_node::processPointCloud, this);
    points_node_pub_ = node_handle_.advertise<sensor_msgs::PointCloud2 >("point_cloud",10);
    test_points_pub_ = node_handle_.advertise<sensor_msgs::PointCloud2 >("test_point",10);
}


void Lidar_node::processPointCloud(const sensor_msgs::PointCloud2 &scan) {
    pcl::PCLPointCloud2 pcl_pc;
    pcl_conversions::toPCL(scan,pcl_pc);
    pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromPCLPointCloud2(pcl_pc,*temp_cloud); // all points' data are stored in temp_cloud

    // declare variable 'test' to store the ROI points
    pcl::PointCloud<pcl::PointXYZI> test;

    /*
      1.get the size of temp_cloud
      2.extract ROI
      3.clustering
    */
    size_t size = temp_cloud->size();
    for (size_t i = 0; i < size; i++) {
        float x = temp_cloud->points[i].x;
        float y = temp_cloud->points[i].y;
        float z = temp_cloud->points[i].z;
        if (x>x_min && x<x_max && y>y_min && y<y_max && z>z_min && z<z_max) {
            test.points.push_back(temp_cloud->points[i]);
        }
    }

    //cout << "Before tracking process, the points number is: " << test.points.size() << endl;

    TrackingModel(&test);

    // convert pcl pointcloud to ROS data form
    sensor_msgs::PointCloud2 point_cloud_msg;
    pcl::toROSMsg(test,point_cloud_msg);
    point_cloud_msg.header.frame_id = "/velodyne";
    points_node_pub_.publish(point_cloud_msg);
}

//calculate Euclidean distance between two points
float Lidar_node::calculate_distance2(pcl::PointXYZ a, pcl::PointXYZ b){
	float dx = a.x - b.x;
	float dy = a.y - b.y;
//    float dz = a.z - b.z;
	float dis = sqrt(dx*dx + dy*dy);
	return dis;
}

pcl::PointCloud<pcl::PointXYZI> getBox(const pcl::PointCloud<pcl::PointXYZI> *mypoint,pcl::PointXYZ origin, pcl::PointXYZ size)
{
	pcl::PointCloud<pcl::PointXYZI> box;
	box = *mypoint;
	origin.z += 0.2;
	size.z += 0.2;
	cout << "box size: " << box.size() << endl;
	int x = round(size.y / 0.05);
	int y = round(size.z / 0.05);
	// upper boundary
	for (int i = 0; i < x; ++i) {
		pcl::PointXYZI tmp;
		tmp.x = origin.x;
		tmp.y = origin.y-i*0.05;
		tmp.z = origin.z;
		box.push_back(tmp);
	}
	// lower boundary
	for (int i = 0; i < x; ++i) {
		pcl::PointXYZI tmp;
		tmp.x = origin.x;
		tmp.y = origin.y-i*0.05;
		tmp.z = origin.z-size.z;
		box.push_back(tmp);
	}
	// left boundary
	for (int j = 0; j < y; ++j) {
		pcl::PointXYZI tmp;
		tmp.x = origin.x;
		tmp.y = origin.y;
		tmp.z = origin.z-j*0.05;
		box.push_back(tmp);
	}
	// right boundary
	for (int j = 0; j < y; ++j) {
		pcl::PointXYZI tmp;
		tmp.x = origin.x;
		tmp.y = origin.y-size.y;
		tmp.z = origin.z-j*0.05;
		box.push_back(tmp);
	}
	return box;
}

pcl::PointXYZ calculateSize(const pcl::PointCloud<pcl::PointXYZI> *points,pcl::PointXYZ &origin)
{
	double maxWidth=-10000.0, minWidth=10000.0;
	double maxHeight=-10000.0, minHeight=10000.0;
	double maxLongth=-10000.0, minLongth=10000.0;
	for (int i = 0; i < points->size(); ++i) {
		maxWidth = Max(maxWidth,points->points[i].x);
		maxLongth = Max(maxLongth, points->points[i].y);
		maxHeight = Max(maxHeight,points->points[i].z);
		minWidth = Min(minWidth,points->points[i].x);
		minLongth = Min(minLongth, points->points[i].y);
		minHeight = Min(minHeight,points->points[i].z);
	}
	pcl::PointXYZ size;
	size.x = maxWidth - minWidth;
	size.y = maxLongth - minLongth;
	size.z = maxHeight - minHeight;

	origin.x = minWidth;
	origin.y = maxLongth;
	origin.z = maxHeight;

	return size;
}

/*
 * step 1: if pointcloud size > 5000, down-sample
 * step 2: use API(EuclideanClusterExtraction) to extract cluster, this test case has only one cluster
 * step 3: extract point from pcl::EuclideanClusterExtraction<pcl::PointXYZI > extractor
 * step 4: publish pointcloud message : test_point
 * step 5: input 1 to show each frame's data
 */
void Lidar_node::TrackingModel(const pcl::PointCloud<pcl::PointXYZI> *pointset)
{

	cout << "##############################" << endl;
	cout << "frame_id:" << frame_id << endl;
	if(frame_id >= 3){
		vector<frame_info >::iterator it = frame_points.begin();
		frame_points.erase(it);
		frame_points[0] = frame_points[1];
		frame_points[1] = frame_points[2];
	}
    pcl::PointCloud<pcl::PointXYZI>::Ptr pointer(new pcl::PointCloud<pcl::PointXYZI>);
    pointer = pointset->makeShared(); // transform to pointer form

    // down-sampling
    if(pointset->points.size() > 5000){
        pcl::VoxelGrid<pcl::PointXYZI> vg;
        vg.setInputCloud(pointer);
        vg.setLeafSize(0.015f,0.015f,0.015f);
        vg.filter(*pointer);
    }

    //cout<<"before filtering has:"<<pointset->size()<<"points"<<endl;
    //cout<<"after filtering has:"<<pointer->size()<<"points"<<endl;

    // KD tree to construct point cloud
    pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZI>);
    kdtree->setInputCloud(pointer);

    vector<pcl::PointIndices > cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI > extractor;
    extractor.setClusterTolerance(0.3); // search range, the smaller the number, the more the number of clusters
    extractor.setMinClusterSize(5);
    extractor.setMaxClusterSize(5000);
    extractor.setSearchMethod(kdtree);
    extractor.setInputCloud(pointer);
    extractor.extract(cluster_indices);

	pcl::search::KdTree<pcl::PointXYZI> vkdtree;
	vkdtree = *kdtree;

	frame_info pinfo;
	pinfo.allpoints = *pointer;
	pinfo.point_cluster_num = cluster_indices.size();
	pinfo.cluster = new cluster_info[cluster_indices.size()];
	for (int k = 0; k < cluster_indices.size(); ++k) {
		pinfo.cluster[k].points.clear();
	}

    cout << cluster_indices.size() << " clusters" << endl;
    pcl::PointCloud<pcl::PointXYZI> mycloud;

    int j = 1;
    float intensity = 255.0f / cluster_indices.size();
    int point_nums = 0;

    pcl::PointCloud<pcl::PointXYZI>::Ptr mcluster(new pcl::PointCloud<pcl::PointXYZI>); // use different intensities to differentiate point clusters
    mcluster->clear();
    for (vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++) {
        //cout << "in function " << endl;
        int cnt = 0;
		float centerx(0.0),centery(0.0),centerz(0.0);
        for(vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++){
            pcl::PointXYZI point;
            point.x = pointer->points[*pit].x;
            point.y = pointer->points[*pit].y;
            point.z = pointer->points[*pit].z;
//            point.intensity = intensity * j;
            point.intensity = pointer->points[*pit].intensity;
			pinfo.cluster[j-1].points.push_back(point);
            mcluster->points.push_back(point);
            cnt ++;
			centerx += point.x;
			centery += point.y;
			centerz += point.z;
        }
        cout << "###" << endl;
		centerx /= cnt;
		centery /= cnt;
		centerz /= cnt;
		pinfo.cluster[j-1].center_x = centerx;
		pinfo.cluster[j-1].center_y = centery;
		pinfo.cluster[j-1].center_z = centerz;
		pinfo.cluster[j-1].points.width = pinfo.cluster[j-1].points.size();
		pinfo.cluster[j-1].points.height = 1;
		pinfo.cluster[j-1].points.is_dense = true;
        mcluster->width = mcluster->points.size();
        mcluster->height = 1;
        mcluster->is_dense = true;
        point_nums += mcluster->points.size();

        cout << "No." << j << ":" << cnt <<" points"  << endl;
        j++;
    }
	/*
	  your tracking algorithm section
	*/

	pcl::PointCloud<pcl::PointXYZI> box;
	if(frame_id == 0){
		for (int i = 0; i < pinfo.point_cluster_num; ++i) {
			cout << "#############" << endl;
			cout << "^No^." << i << endl;
			pinfo.cluster[i].pf->objectid = i;
			pinfo.cluster[i].pf->initialParticle(&pinfo.cluster[i].points);
			cout << "cluster:" << "(" << pinfo.cluster[i].center_x << "," << pinfo.cluster[i].center_y << "," << pinfo.cluster[i].center_z << ")" << endl;
			pinfo.cluster[i].pf->printThisParticle(0);
			cout << "#############" << endl;
		}
	}
	else{
		cout << "frame_id: " << frame_id << endl;
		int id = frame_id >= 3 ? 1 : (frame_id%3)-1;
		cout << "id: " << id << endl;
		cout << "cluster num: " << frame_points[id].point_cluster_num << endl;
		for (int i = 0; i < frame_points[id].point_cluster_num; ++i) {
			particle previous_position_particle;
			previous_position_particle = frame_points[id].cluster[i].pf->particles[0];

			frame_points[id].cluster[i].pf->transition();
			cout << "transition." << endl;

			frame_points[id].cluster[i].pf->getLikelihood(&vkdtree,&pinfo.allpoints,&frame_points[id].cluster[i].points);
			cout << "calculate likelihood." << endl;

			frame_points[id].cluster[i].pf->normalizeWeights();
			cout << "normalize." << endl;

			frame_points[id].cluster[i].pf->resample();
			cout << "resample." << endl;

			pcl::PointXYZ point;
			point = frame_points[id].cluster[i].pf->getPosition();
			cout << "-------------------------------------------" << endl;
			cout << "predict position:(" << point.x << "," << point.y << "," << point.z << ")" << endl;

			// find the nearest cluster in predict position
			float min_distance = 10000.0;
			int cluster_id = 0;
			pcl::PointXYZ cluster_center_point;
			for (int k = 0; k < pinfo.point_cluster_num; ++k) {
				cout << "***********************" << endl;
				cout << "Cluster No." << k << endl;
				cout << "Cluster center:" << endl;
				cout << "(" << pinfo.cluster[k].center_x << "," << pinfo.cluster[k].center_y << "," << pinfo.cluster[k].center_z << ")" << endl;
				cluster_center_point.x = pinfo.cluster[k].center_x;
				cluster_center_point.y = pinfo.cluster[k].center_y;
				cluster_center_point.z = pinfo.cluster[k].center_z;
				float distance = calculate_distance2(point,cluster_center_point);
				cout << " distance: " << distance << endl;
				if(distance < min_distance){
					cluster_id = k;
					min_distance = distance;
				}
				cout << "***********************" << endl;
			}
			cout << "-------cluster_id: " << cluster_id << " minmum distance: " << min_distance << "-------" << endl;
			pcl::PointXYZ size;
			pcl::PointXYZ origin;
			size = calculateSize(&pinfo.cluster[cluster_id].points,origin);
			cout <<"size: " <<  size.x << " " << size.y  << " " << size.z << endl;
			cout <<"origin: " << origin.x << " " << origin.y << " " << origin.z << endl;
			pinfo.cluster[cluster_id].pf->updateParticle(&previous_position_particle,&cluster_center_point,size);
			box = getBox(&pinfo.cluster[cluster_id].points,origin,size);
			int next;
//			while(1){
//				cout << "input 1 to continue running the program." << endl;
//				cin >> next;
//				if(next==1)
//					break;
//			}
			cout << "-------------------------------------------" << endl;
		}
	}

    mycloud = *mcluster;
    cout << "end." << endl;
    sensor_msgs::PointCloud2 pub_msgs;
//    pcl::toROSMsg(mycloud,pub_msgs);
	pcl::toROSMsg(box,pub_msgs);
    pub_msgs.header.frame_id = "/velodyne";
    test_points_pub_.publish(pub_msgs);
	frame_id++;
	frame_points.push_back(pinfo);
//    while(1){
//        int key;
//        cin >> key;
//        if(key == 1){
//            cout << "continue." << endl;
//            break;
//        }
//    }
	cout << "##############################" << endl;
}

int main(int argc, char **argv)
{
    cout << 1 << endl;
    ros::init(argc,argv,"Lidar_node");
    Lidar_node node;
    ros::spin();
    return 0;
}
