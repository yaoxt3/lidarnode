/*
  author: yxt
  create it in 2018-1-19
*/

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
#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
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
using namespace cv;


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

pcl::PointXYZ getCenter(const pcl::PointCloud<pcl::PointXYZI> *points);
pcl::PointXYZ getOrigin(const pcl::PointXYZ center, const pcl::PointXYZ origin, const pcl::PointXYZ predict);
pcl::PointXYZ calculateSize(const pcl::PointCloud<pcl::PointXYZI> *points,pcl::PointXYZ &origin);
pcl::PointCloud<pcl::PointXYZI> getBox(const pcl::PointCloud<pcl::PointXYZI> *mypoint,pcl::PointXYZ origin, pcl::PointXYZ size);
pcl::PointCloud<pcl::PointXYZI> transformPointCloud(const particle *particles, int len);
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
	void updateParticle(const particle *point,const pcl::PointXYZ *center_point,const pcl::PointXYZ);
	void addObject(const pcl::PointCloud<pcl::PointXYZI> *);
	pcl::PointXYZ getPosition();
	void printAllParticle();
	void printThisParticle(int);
	int objectid;
	double std_x,std_y,std_z;
	double A0,A1,B;
	static const int MAX_PARTICLE_NUM = 30;
	static const int MAX_INTENSITY = 300;
	particle *particles;
	gsl_rng *rng;
};

ParticleFilter::ParticleFilter(){
	objectid = 0;
	std_x = 0.15;
	std_y = 0.2;
	std_z = 0.1;
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
	pcl::PointXYZ center;
	center = getCenter(points);
    double maxWidth=0.0, minWidth=10000.0;
    double maxHeight=0.0, minHeight=10000.0;
    double maxLongth=0.0, minLongth=10000.0;
    for (int i = 0; i < points->size(); ++i) {
        maxWidth = Max(maxWidth,points->points[i].x);
        maxLongth = Max(maxLongth, points->points[i].y);
        maxHeight = Max(maxHeight,points->points[i].z);
        minWidth = Min(minWidth,points->points[i].x);
        minLongth = Min(minLongth, points->points[i].y);
        minHeight = Min(minHeight,points->points[i].z);
    }
    // initilize particle's position
    for (int j = 0; j < MAX_PARTICLE_NUM; ++j) {
        particles[j].width = maxWidth - minWidth;
        particles[j].height = maxHeight - minHeight;
        particles[j].longth = maxLongth - minLongth;
        particles[j].x0 = center.x;
        particles[j].y0 = center.y;
        particles[j].z0 = center.z;
        particles[j].x = particles[j].x0;
        particles[j].y = particles[j].y0;
        particles[j].z = particles[j].z0;
        particles[j].px = particles[j].x;
        particles[j].py = particles[j].y;
        particles[j].pz = particles[j].z;
        particles[j].likelihood = 0.0;
    }
}

void ParticleFilter::addObject(const pcl::PointCloud<pcl::PointXYZI> *point) {
	initialParticle(point);
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
//	cout << "likelihood function." << endl;
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
//	cout << "end likelihood." << endl;
}

void ParticleFilter::printAllParticle() {
	cout << "%%%%%%%%%%%%%%%%%%%%%" << endl;
	cout << "original position:";
	cout << "(" << particles[0].x0 << "," << particles[0].y0 << "," << particles[0].z0 << ")" << endl;
	for (int i = 0; i < MAX_PARTICLE_NUM; ++i) {
		cout << "particle " << i << ":";
		cout << "(" << particles[i].x << "," << particles[i].y << "," << particles[i].z << ") ";
		cout << "likelihood:" << particles[i].likelihood << endl;
	}
	cout << "%%%%%%%%%%%%%%%%%%%%%" << endl;
}

void ParticleFilter::printThisParticle(int i) {
	cout << "particle " << i << ":" << endl;
	cout << "current position:";
	cout << "(" << particles[i].x << "," << particles[i].y << "," << particles[i].z << ")" << endl;
	cout << "likelihood:" << particles[i].likelihood << endl;
	cout << "original position:";
	cout << "(" << particles[i].x0 << "," << particles[i].y0 << "," << particles[i].z0 << ")" << endl;
}

void ParticleFilter::updateParticle(const particle *myparticle,const pcl::PointXYZ *center_point,const pcl::PointXYZ size) {
	cout << "^^^^^^^^^^^^^^^^^^" << endl;
	cout << "In update function." << endl;
	cout << myparticle->x << " " << myparticle->y << " " << myparticle->z << endl;
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
//	cout << "end sample." << endl;
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
	pcl::PointXYZ center;
	double height;
	double width;
	double longth;
	ParticleFilter *pf;
	pcl::PointCloud<pcl::PointXYZI> points;
	cluster_info(){
		pf = new ParticleFilter;
		height = 0.0;
		width = 0.0;
		longth = 0.0;
		center.x = center.y = center.z = 0.0;
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
    ros::Publisher points_node_pub_;
    ros::Publisher test_points_pub_;

    // set left-right threshold for y-axis in lidar coordinate
	int frame_id;
	const int frame_num;
	const int searchNum;
	int max_frame;
    float left_threshold;
    float right_threshold;
    float forward_threshold;
    float forward_max_threshold;
    vector<frame_info> frame_points; // record three frames information
};


Lidar_node::Lidar_node():searchNum(100),frame_num(3){ // error : node_handle_("~")
	ROS_INFO("In constructed function.");
//    left_threshold = 2.5;
//    right_threshold = 2;
//    forward_threshold = 0.25;
//    forward_max_threshold = 3;
	left_threshold = 4;
	right_threshold = 4; // 2.5 single ball, 4 multi-ball
	forward_threshold = 0.25;
	forward_max_threshold = 6;
    frame_id = 0;
    max_frame = 10;
    frame_points.clear();
    points_node_sub_ = node_handle_.subscribe("velodyne_points", 1028, &Lidar_node::processPointCloud, this);
    points_node_pub_ = node_handle_.advertise<sensor_msgs::PointCloud2 >("point_cloud",10);
	test_points_pub_ = node_handle_.advertise<sensor_msgs::PointCloud2 >("test_point",10);
}

//calculate Euclidean distance between two points
float Lidar_node::calculate_distance2(pcl::PointXYZ a, pcl::PointXYZ b){
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    float dis = sqrt(dx*dx + dy*dy + dz*dz);
    return dis;
}

pcl::PointCloud<pcl::PointXYZI> getBox(const pcl::PointCloud<pcl::PointXYZI> *mypoint,pcl::PointXYZ origin, pcl::PointXYZ size)
{
	pcl::PointCloud<pcl::PointXYZI> box;
	box.clear();
	box = *mypoint;
//	origin.z += 0.2;
//	size.z += 0.2;
	int x = round(size.y / 0.05);
	int y = round(size.z / 0.05);
	/*** front square ***/
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

	/*** back square ***/
	for (int i = 0; i < x; ++i) {
		pcl::PointXYZI tmp;
		tmp.x = origin.x+size.x;
		tmp.y = origin.y-i*0.05;
		tmp.z = origin.z;
		box.push_back(tmp);
	}
	// lower boundary
	for (int i = 0; i < x; ++i) {
		pcl::PointXYZI tmp;
		tmp.x = origin.x+size.x;
		tmp.y = origin.y-i*0.05;
		tmp.z = origin.z-size.z;
		box.push_back(tmp);
	}
	// left boundary
	for (int j = 0; j < y; ++j) {
		pcl::PointXYZI tmp;
		tmp.x = origin.x+size.x;
		tmp.y = origin.y;
		tmp.z = origin.z-j*0.05;
		box.push_back(tmp);
	}
	// right boundary
	for (int j = 0; j < y; ++j) {
		pcl::PointXYZI tmp;
		tmp.x = origin.x+size.x;
		tmp.y = origin.y-size.y;
		tmp.z = origin.z-j*0.05;
		box.push_back(tmp);
	}

	/*** left square ***/
	int m = round(size.x/0.05);
	// upper
	for (int i = 0; i < m; ++i) {
		pcl::PointXYZI tmp;
		tmp.x = origin.x + i*0.05;
		tmp.y = origin.y;
		tmp.z = origin.z;
		box.push_back(tmp);
	}
	// lower
	for (int k = 0; k < m; ++k) {
		pcl::PointXYZI tmp;
		tmp.x = origin.x + k*0.05;
		tmp.y = origin.y;
		tmp.z = origin.z - size.z;
		box.push_back(tmp);
	}

	/*** right square ***/
	// upper
	for (int l = 0; l < m; ++l) {
		pcl::PointXYZI tmp;
		tmp.x = origin.x + l*0.05;
		tmp.y = origin.y - size.y;
		tmp.z = origin.z;
		box.push_back(tmp);
	}
	// lower
	for (int n = 0; n < m; ++n) {
		pcl::PointXYZI tmp;
		tmp.x = origin.x + n*0.05;
		tmp.y = origin.y - size.y;
		tmp.z = origin.z - size.z;
		box.push_back(tmp);
	}
	cout << "box size: " << box.size() << endl;
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

pcl::PointCloud<pcl::PointXYZI> transformPointCloud(const particle *particles, int len)
{
	pcl::PointCloud<pcl::PointXYZI> points;
	for (int i = 0; i < len; ++i) {
		pcl::PointXYZI point;
		point.x = particles[i].x;
		point.y = particles[i].y;
		point.z = particles[i].z;
		point.intensity = 10;
		points.push_back(point);
	}
	return points;
}

pcl::PointXYZ getCenter(const pcl::PointCloud<pcl::PointXYZI> *points)
{
	pcl::PointXYZ center;
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
	center.x = (minWidth+maxWidth)/2.0;
	center.y = (minLongth+maxLongth)/2.0;
	center.z = (minHeight+maxHeight)/2.0;
	return center;
}

pcl::PointXYZ getOrigin(const pcl::PointXYZ center, const pcl::PointXYZ origin, const pcl::PointXYZ predict)
{
	pcl::PointXYZ point;
	point.x = predict.x + origin.x - center.x;
	point.y = predict.y + origin.y - center.y;
	point.z = predict.z + origin.z - center.z;
	cout << "origin:(" << origin.x << "," << origin.y << "," << origin.z << ")" << endl;
	cout << "new_origin:(" << point.x << "," << point.y << "," << point.z << ")" << endl;

	return point;
}

// Tracking Model for point cloud

bool nocluster = false;
void Lidar_node::TrackingModel(const pcl::PointCloud<pcl::PointXYZI> *pointset)
{

	cout << "##############################" << endl;
	cout << "frame_id:" << frame_id << endl;
	if(frame_id >= max_frame && nocluster == false){
		vector<frame_info >::iterator it = frame_points.begin();
		frame_points.erase(it);
		for (int i = 0; i < max_frame-1; ++i) {
			frame_points[i] = frame_points[i+1];
		}
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
    extractor.setClusterTolerance(0.2);
    extractor.setMinClusterSize(35);
    extractor.setMaxClusterSize(5000);
    extractor.setSearchMethod(kdtree);
    extractor.setInputCloud(pointer);
    extractor.extract(cluster_indices);

    pcl::search::KdTree<pcl::PointXYZI> vkdtree;
    vkdtree = *kdtree;


    cout << cluster_indices.size() << " clusters" << endl;

    pcl::PointCloud<pcl::PointXYZI> mycloud;

    frame_info pinfo;
    pinfo.allpoints = *pointer;
    pinfo.point_cluster_num = cluster_indices.size();
    if(pinfo.point_cluster_num == 0){
    	nocluster = true;
    } else
    	nocluster = false;
    pinfo.cluster = new cluster_info[cluster_indices.size()];
	for (int k = 0; k < cluster_indices.size(); ++k) {
		pinfo.cluster[k].points.clear();
	}
	cout << "@@@" << endl;
    int j = 1;
    float intensity = 255.0f / cluster_indices.size();
    int point_nums = 0;

	pcl::PointCloud<pcl::PointXYZI>::Ptr mcluster(new pcl::PointCloud<pcl::PointXYZI>); // use different intensities to differentiate point clusters
	mcluster->clear();
	if(nocluster == true)
		return;
	for (vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++) {
        //cout << "in function " << endl;
        int cnt = 0;
		cout << "cluster " << j-1 << ":" << endl;
		for(vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++){
            pcl::PointXYZI point;
            point.x = pointer->points[*pit].x;
            point.y = pointer->points[*pit].y;
            point.z = pointer->points[*pit].z;
            //point.intensity = intensity * j;
           	point.intensity = pointer->points[*pit].intensity;
            pinfo.cluster[j-1].points.push_back(point);
            mcluster->points.push_back(point);
//            cout << "(" << point.x << "," << point.y << "," << point.z << ")" << endl;
		    cnt ++;
		}
		cout << endl;
		cout << "###" << endl;
		pcl::PointXYZ center;
		center = getCenter(&pinfo.cluster[j-1].points);
		pinfo.cluster[j-1].center = center;
		pinfo.cluster[j-1].points.width = pinfo.cluster[j-1].points.size();
		pinfo.cluster[j-1].points.height = 1;
		pinfo.cluster[j-1].points.is_dense = true;
        mcluster->width = mcluster->points.size();
        mcluster->height = 1;
        mcluster->is_dense = true;
        point_nums += mcluster->points.size();

        cout << "No." << j << ":" << cnt <<" points"  << endl;
        cout << "Center position: (" << center.x << "," << center.y << "," << center.z << ")" << endl;
        j++;
	}
	cout << "***********************" << endl;

	/*##particle filter section##
	 * initialize particles in the first frame, otherwise process previous particles
	 * this takes three steps:
	 * 1.transition according to a certain motion model
	 * 2.observing the likelihood of the object being at the translated position (results in a weight)
	 * 3.re-sample according to that likelihood (given by the weight)
	 */
	pcl::PointCloud<pcl::PointXYZI> particles;
	pcl::PointCloud<pcl::PointXYZI> box;
	pcl::PointCloud<pcl::PointXYZI> allParticles;
	box.clear();
	if(frame_id == 0){
		for (int i = 0; i < pinfo.point_cluster_num; ++i) {
			cout << "#############" << endl;
			cout << "^No^." << i << endl;
			pinfo.cluster[i].pf->objectid = i;
			pinfo.cluster[i].pf->initialParticle(&pinfo.cluster[i].points);
			cout << "cluster:" << "(" << pinfo.cluster[i].center.x << "," << pinfo.cluster[i].center.y << "," << pinfo.cluster[i].center.z << ")" << endl;
//			pinfo.cluster[i].pf->printThisParticle(0);
			cout << "#############" << endl;
		}
	}
	else{
		cout << "frame_id: " << frame_id << endl;
		int id = frame_id >= max_frame ? max_frame-2 : (frame_id%max_frame)-1;
		cout << "id: " << id << endl;
		if(pinfo.point_cluster_num==0){
			nocluster = true;
			return;
		} else
			nocluster = false;

		cout << "precious cluster num: " << frame_points[id].point_cluster_num << endl;
		// judges whether this object belongs to the previous frame through its original position (x0,y0,z0)
		// if not, add it to the objectList
		pcl::PointCloud<pcl::PointXYZ> newObjectList;
		newObjectList.clear();
		pcl::PointCloud<pcl::PointXYZ> deleteObjectList;
		deleteObjectList.clear();
		if(pinfo.point_cluster_num > frame_points[id].point_cluster_num) {
			int min_flag[pinfo.point_cluster_num];
			for (int k = 0; k < pinfo.point_cluster_num; ++k)
				min_flag[k] = -1;
			for (int l = 0; l < frame_points[id].point_cluster_num; ++l) {
				double mindis = 10000.0;
				int idd;
				for (int i = 0; i < pinfo.point_cluster_num; ++i) {
					double dis = calculate_distance2(pinfo.cluster[i].center, frame_points[id].cluster[l].center);
					if (mindis > dis) {
						mindis = dis;
						idd = i;
					}
				}
				cout << "old object id:" << idd << ", distances: " << mindis << endl;
				min_flag[idd] = l;
			}
			for (int m = 0; m < pinfo.point_cluster_num; ++m) {
				if (min_flag[m] == -1) {
					pcl::PointXYZ point;
					pinfo.cluster[m].pf->addObject(&pinfo.cluster[m].points);
					point.x = pinfo.cluster[m].pf->particles[0].x0;
					point.y = pinfo.cluster[m].pf->particles[0].y0;
					point.z = pinfo.cluster[m].pf->particles[0].z0;
					newObjectList.push_back(point);
					cout << "add new object:(" << point.x << "," << point.y << "," << point.z << ")" << endl;
				}
			}
		}
		cout<< "newObjectList size:" << newObjectList.size() << endl;
		allParticles.clear();
		for (int i = 0; i < frame_points[id].point_cluster_num; ++i) {
			pcl::PointCloud<pcl::PointXYZI> iparticles;
			particle previous_position_particle;
			bool go_next = false;
			previous_position_particle = frame_points[id].cluster[i].pf->particles[0];

			frame_points[id].cluster[i].pf->transition();
			cout << "transition." << endl;

			iparticles = transformPointCloud(frame_points[id].cluster[i].pf->particles,frame_points[id].cluster[i].pf->MAX_PARTICLE_NUM);
//			frame_points[id].cluster[i].pf->printAllParticle();

			frame_points[id].cluster[i].pf->getLikelihood(&vkdtree,&pinfo.allpoints,&frame_points[id].cluster[i].points);
//			cout << "likelihood." << endl;

			frame_points[id].cluster[i].pf->normalizeWeights();
//			cout << "normalize." << endl;

			frame_points[id].cluster[i].pf->resample();
			cout << "resample." << endl;

//			frame_points[id].cluster[i].pf->printAllParticle();

			pcl::PointXYZ point;
			point = frame_points[id].cluster[i].pf->getPosition();
			cout << "-------------------------------------------" << endl;
			cout << "predict position:(" << point.x << "," << point.y << "," << point.z << ")" << endl;

			// find the nearest cluster in predict position
			float min_distance = 10000.0;
			int cluster_id = 0;
			pcl::PointXYZ cluster_center_point;
			for (int k = 0; k < pinfo.point_cluster_num; ++k) {
				// ignore new object
				cout << "check." << k << endl;
				double x1 = pinfo.cluster[k].pf->particles[0].x0;
				double y1 = pinfo.cluster[k].pf->particles[0].y0;
				double z1 = pinfo.cluster[k].pf->particles[0].z0;
				cout << "this center is (" << x1 << "," << y1 << "," << z1 << ")" << endl;
				for (int ll = 0; ll < newObjectList.size(); ++ll) {
					double x0 = newObjectList[ll].x;
					double y0 = newObjectList[ll].y;
					double z0 = newObjectList[ll].z;
					cout << "the objectList center is (" << x0 << "," << y0 << "," << z0 << ")" << endl;
					if(x0==x1 && y0==y1 && z0==z1){
						cout << "continue." << endl;
						go_next = true;
						break;
					}
				}
				if(go_next == true)
					continue;
				cout << "Cluster No." << k << endl;
				cout << "Cluster center:" << endl;
				cout << "(" << pinfo.cluster[k].center.x << "," << pinfo.cluster[k].center.y << "," << pinfo.cluster[k].center.z << ")" << endl;
				cluster_center_point.x = pinfo.cluster[k].center.x;
				cluster_center_point.y = pinfo.cluster[k].center.y;
				cluster_center_point.z = pinfo.cluster[k].center.z;
				float distance = calculate_distance2(point,cluster_center_point);
				cout << " distance: " << distance << endl;
				if(distance < min_distance){
					cluster_id = k;
					min_distance = distance;
				}
				cout << "***********************" << endl;
			}
			cout << "***********************" << endl;
			cout << "-------cluster_id: " << cluster_id << " minmum distance: " << min_distance << "-------" << endl;
			pcl::PointXYZ size;
			pcl::PointXYZ porigin;
			pcl::PointXYZ origin;
			pcl::PointXYZ center;
			double longth;
			center.x = pinfo.cluster[cluster_id].center.x;
			center.y = pinfo.cluster[cluster_id].center.y;
			center.z = pinfo.cluster[cluster_id].center.z;

			size = calculateSize(&pinfo.cluster[cluster_id].points,porigin);
			longth = sqrtf(size.x*size.x + size.y*size.y + size.z*size.z);
			// to be completed
			if(min_distance > longth){
				cout << "minmum distance:" << min_distance << " longth:" << longth << endl;
				cout << "minmum distance is bigger than longth, not match." << endl;
				continue;
			}
			origin = getOrigin(center,porigin,point);
			cout <<"size: " <<  size.x << " " << size.y  << " " << size.z << endl;
			//cout <<"origin: " << origin.x << " " << origin.y << " " << origin.z << endl;
			cout <<"previous position: (" << previous_position_particle.x << "," << previous_position_particle.y << "," << previous_position_particle.z << ")" << endl;
			cout <<"original position: (" << previous_position_particle.x0 << "," << previous_position_particle.y0 << "," << previous_position_particle.z0 << ")" << endl;
			pinfo.cluster[cluster_id].pf->updateParticle(&previous_position_particle,&center,size);
//			pinfo.cluster[cluster_id].pf->printAllParticle();
			box = getBox(&box,origin,size);
			allParticles.insert(allParticles.end(),iparticles.begin(),iparticles.end());

			cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << endl;
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

	mcluster->insert(mcluster->end(),box.begin(),box.end());
//	mcluster->insert(mcluster->end(),allParticles.begin(),allParticles.end());

	mycloud = *mcluster;
	cout << "end." << endl;
    sensor_msgs::PointCloud2 pub_msgs;
    pcl::toROSMsg(mycloud,pub_msgs);
//	pcl::toROSMsg(box,pub_msgs);
    pub_msgs.header.frame_id = "/velodyne";
    test_points_pub_.publish(pub_msgs);
	frame_id++;
	frame_points.push_back(pinfo);
//    while(1){
//        int key;
//        cin >> key;
//		cout << "input 1 to continue running the program." << endl;
//        if(key == 1){
//            break;
//        }
//    }
	cout << "##############################" << endl;
}


// extract pointclouds from different bags, and publish them by topic "point_cloud"
void Lidar_node::processPointCloud(const sensor_msgs::PointCloud2 &scan) {
    pcl::PCLPointCloud2 pcl_pc;
    pcl_conversions::toPCL(scan,pcl_pc);
    pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromPCLPointCloud2(pcl_pc,*temp_cloud); // all points' data are stored in temp_cloud

    // declare variable 'test' to store river surface points
    pcl::PointCloud<pcl::PointXYZI> test;

    /*
      1.get the size of temp_cloud
      2.extract river surface
      3.clustering
    */
    size_t size = temp_cloud->size();
    for (size_t i = 0; i < size; i++) {
      float x = temp_cloud->points[i].x;
      float y = temp_cloud->points[i].y;
      if (x>forward_threshold && x<forward_max_threshold && y>-1*left_threshold && y<right_threshold) {
        test.points.push_back(temp_cloud->points[i]);
      }
    }

    cout << "Before tracking process, the points number is: " << test.points.size() << endl;

    //cluster_function(&test); // pass data by pointer
    TrackingModel(&test);
    //while(1);
    // convert pcl pointcloud to ROS data form
//    sensor_msgs::PointCloud2 point_cloud_msg;
//    pcl::toROSMsg(test,point_cloud_msg);
//    point_cloud_msg.header.frame_id = "/velodyne";
//    points_node_pub_.publish(point_cloud_msg);
}

int main(int argc, char **argv) {
    cout<<111<<endl;
    srand(time(NULL));

    ros::init(argc,argv,"Lidar_node");
    Lidar_node node;
    ros::spin();

    return 0;
}
