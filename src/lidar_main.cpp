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
	void getLikelihood(const pcl::search::KdTree<pcl::PointXYZI> *kdtree,const pcl::PointCloud<pcl::PointXYZI> *pointset, const pcl::PointCloud<pcl::PointXYZI> *mypoint);
	pcl::PointXYZ getPosition();
	void printAllParticle();
	void printThisParticle(int);
	static bool compareWeight(const particle &a,const particle &b){
		return a.likelihood >= b.likelihood;
	}
	int objectid;
	double std_x,std_y,std_z;
	double A0,A1,B;
	static const int MAX_PARTICLE_NUM = 30;
	particle *particles;
	gsl_rng *rng;
};

ParticleFilter::ParticleFilter(){
	objectid = 0;
	std_x = 1.0;
	std_y = 1.5;
	std_z = 0.3;
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
	delete []particles;
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
    double maxWidth=0.0, minWidth=10000.0;
    double maxHeight=0.0, minHeight=10000.0;
    double maxLongth=0.0, minLongth=10000.0;
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

void ParticleFilter::getLikelihood(const pcl::search::KdTree<pcl::PointXYZI> *kdtree, const pcl::PointCloud<pcl::PointXYZI> *pointset, const pcl::PointCloud<pcl::PointXYZI> *mypoint) {
	cout << "likelihood function." << endl;
	int *pf_intensity,*object_intensity;
	vector<int>pointRadiusSearch;
	vector<float>pointRadiusSquareDistance;
	pcl::PointCloud<pcl::PointXYZI> pfpoint; // observed points at the current position
	pfpoint.clear();
	for (int i = 0; i < MAX_PARTICLE_NUM; ++i) {
		pcl::PointXYZI point;
		point.x = particles[i].x;
		point.y = particles[i].y;
		point.z = particles[i].z;
		float radius;
		float width = particles[i].width;
		float height = particles[i].height;
		float longth = particles[i].longth;
		cout << width << " " << height << " " << longth << endl;
		radius = 0.5 * sqrt(width*width + height*height + longth*longth);
		cout << "radius: " << radius << endl;
		if(kdtree->radiusSearch(point,radius,pointRadiusSearch,pointRadiusSquareDistance) > 0){
			cout << "in if section." << endl;
			cout << "num: " << pointRadiusSearch.size() << endl;
			for (int j = 0; j < pointRadiusSearch.size(); ++j) {
				double dx = abs(pointset->points[pointRadiusSearch[j]].x - width);
				double dy = abs(pointset->points[pointRadiusSearch[j]].y - height);
				double dz = abs(pointset->points[pointRadiusSearch[j]].z - longth);
				cout << dx << " " << dy << " " << dz << endl;
				if(dx > 0.5*width || dy > 0.5*height || dz > 0.5*longth){
					continue;
				}
				else{
					pfpoint.push_back(pointset->points[pointRadiusSearch[j]]);
				}
			}
			if(pfpoint.size() > 0){
				cout << "!!" << endl;
				int maxSize = max(mypoint->size(),pfpoint.size());
				cout << maxSize << " " << mypoint->size() << " " << pfpoint.size() << endl;
				object_intensity = new int[maxSize];
				pf_intensity = new int[maxSize];
				for (int i = 0; i < maxSize; ++i) {
					object_intensity[i] = 0;
					pf_intensity[i] = 0;
				}
				for (int k = 0; k < maxSize; ++k) {
					int intensity = round(mypoint->points[k].intensity);
					object_intensity[intensity] = object_intensity[intensity] + 1;

					int intensity2 = round(pfpoint.points[k].intensity);
					pf_intensity[intensity2] = pf_intensity[intensity2] + 1;
				}
				cout << "@@" << endl;
				//normalization
				float *fobject_intensity = new float[maxSize];
				float *fpf_intensity = new float[maxSize];
				for (int l = 0; l < maxSize; ++l) {
					fobject_intensity[l] = object_intensity[l]*1.0/mypoint->size();
					fpf_intensity[l] = pf_intensity[l]*1.0/pfpoint.size();
				}

				// calculate the similarity by point number and intensity
				float numWeight = 0.0;
				float intensityWeight = 0.0;
				float similarity = 0.0;
				for (int m = 0; m < maxSize; ++m) {
					intensityWeight = intensityWeight + sqrt(fobject_intensity[m]*fpf_intensity[m]);
				}
				intensityWeight = 1 - intensityWeight;
				intensityWeight = exp(-1.0*intensityWeight);

				numWeight = pfpoint.size()/mypoint->points.size();
				similarity = numWeight * intensityWeight;

				/*
				 * need consider more methods to calculate likelihood, for example:
				 * compare particle's point distribution and object's point distribution.
				 */

				particles[i].likelihood = similarity;
			}
		} else
			particles[i].likelihood = 0.0;
	}
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

void ParticleFilter::resample() {
	int number = 0;
	int count = 0;
	particle *tmp = new particle[MAX_PARTICLE_NUM];
	sort(particles,particles+MAX_PARTICLE_NUM,compareWeight);
	for (int i = 0; i < MAX_PARTICLE_NUM; ++i) {
		number = round(particles[i].likelihood * MAX_PARTICLE_NUM);
		for (int j = 0; j < number; ++j) {
			tmp[count++] = particles[i];
			if(count == MAX_PARTICLE_NUM)
				break;
		}
		if(count == MAX_PARTICLE_NUM)
			break;
	}

	while(count < MAX_PARTICLE_NUM){
		tmp[count] = particles[0];
		count++;
	}

	for (int k = 0; k < MAX_PARTICLE_NUM; ++k) {
		particles[k] = tmp[k];
	}

	delete tmp;
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
	float calculate_distance2(const pcl::PointXYZI a, const pcl::PointXYZI b);
private:
    ros::NodeHandle node_handle_;
    ros::Subscriber points_node_sub_;
    ros::Publisher points_node_pub_;
    ros::Publisher test_points_pub_;

    // set left-right threshold for y-axis in lidar coordinate
	int frame_id;
	const int frame_num;
	const int searchNum;
    float left_threshold;
    float right_threshold;
    float forward_threshold;
    float forward_max_threshold;
    vector<frame_info > frame_points; // record three frames information
};


Lidar_node::Lidar_node():searchNum(100),frame_num(3){ // error : node_handle_("~")
	ROS_INFO("In constructed function.");
    left_threshold = 2.5;
    right_threshold = 2;
    forward_threshold = 0.25;
    forward_max_threshold = 3;
    frame_id = 0;
    frame_points.clear();
    points_node_sub_ = node_handle_.subscribe("velodyne_points", 1028, &Lidar_node::processPointCloud, this);
    points_node_pub_ = node_handle_.advertise<sensor_msgs::PointCloud2 >("point_cloud",10);
	test_points_pub_ = node_handle_.advertise<sensor_msgs::PointCloud2 >("test_point",10);
}

//calculate Euclidean distance between two points
float Lidar_node::calculate_distance2(pcl::PointXYZI a, pcl::PointXYZI b){
    float dx = a.x - b.y;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    float dis = dx*dx + dy*dy + dz*dz;
    return dis;
}


// Tracking Model for point cloud
void Lidar_node::TrackingModel(const pcl::PointCloud<pcl::PointXYZI> *pointset)
{

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
    extractor.setClusterTolerance(0.05);
    extractor.setMinClusterSize(5);
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
	for (vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++) {
        //cout << "in function " << endl;
        int cnt = 0;
        float centerx(0.0),centery(0.0),centerz(0.0);
		for(vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++){
            pcl::PointXYZI point;
            point.x = pointer->points[*pit].x;
            point.y = pointer->points[*pit].y;
            point.z = pointer->points[*pit].z;
            point.intensity = intensity * j;
           	//point.intensity = pointer->points[*pit].intensity;
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


	/*##particle filter section##
	 * initialize particles in the first frame, otherwise process previous particles
	 * this takes three steps:
	 * 1.transition according to a certain motion model
	 * 2.observing the likelihood of the object being at the translated position (results in a weight)
	 * 3.re-sample according to that likelihood (given by the weight)
	 */
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
			frame_points[id].cluster[i].pf->transition();
			//frame_points[id].cluster[i].pf->printAllParticle();
			cout << "transition." << endl;

			frame_points[id].cluster[i].pf->getLikelihood(&vkdtree,&pinfo.allpoints,&frame_points[id].cluster[i].points);
			cout << "likelihood." << endl;

			frame_points[id].cluster[i].pf->normalizeWeights();
			cout << "normalize." << endl;

			frame_points[id].cluster[i].pf->resample();
			cout << "resample." << endl;

			pcl::PointXYZ point;
			point = frame_points[id].cluster[i].pf->getPosition();
			cout << "predict position:(" << point.x << "," << point.y << "," << point.z << ")" << endl;
			cout << "########################" << endl;
			for (int k = 0; k < pinfo.point_cluster_num; ++k) {
				cout << "Cluster center:" << endl;
				cout << "Cluster No." << k << endl;
				cout << "(" << pinfo.cluster[k].center_x << "," << pinfo.cluster[k].center_y << "," << pinfo.cluster[k].center_z << ")" << endl;
			}
			while(1);
		}
	}

	mycloud = *mcluster;
	cout << "end." << endl;
    sensor_msgs::PointCloud2 pub_msgs;
    pcl::toROSMsg(mycloud,pub_msgs);
    pub_msgs.header.frame_id = "/velodyne";
    test_points_pub_.publish(pub_msgs);
	frame_id++;
	frame_points.push_back(pinfo);
    while(1){
        int key;
        cin >> key;
        if(key == 1){
            cout << "continue." << endl;
            break;
        }
    }
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
    sensor_msgs::PointCloud2 point_cloud_msg;
    pcl::toROSMsg(test,point_cloud_msg);
    point_cloud_msg.header.frame_id = "/velodyne";
    points_node_pub_.publish(point_cloud_msg);
}

int main(int argc, char **argv) {
    cout<<111<<endl;
    srand(time(NULL));

    ros::init(argc,argv,"Lidar_node");
    Lidar_node node;
    ros::spin();

    return 0;
}
