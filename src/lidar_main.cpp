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
	particle transition(particle p);
	void normalizeWeights();
	void resample();
	void getLikelihood(const pcl::search::KdTree<pcl::PointXYZI> *kdtree,const pcl::PointCloud<pcl::PointXYZI> *pointset);
	bool compareWeight(const particle&,const particle&);
	int objectid;
	double std_x,std_y,std_z;
	double A0,A1,B;
	const int MAX_PARTICLE_NUM;
	particle *particles;
	gsl_rng *rng;
};

ParticleFilter::ParticleFilter():MAX_PARTICLE_NUM(30){
	objectid = 0;
	std_x = 1.0;
	std_y = 0.6;
	std_z = 1.3;
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
        this->particles[j].width = maxWidth - minWidth;
        this->particles[j].height = maxHeight - minHeight;
        this->particles[j].longth = maxLongth - minLongth;
        this->particles[j].x0 = mean_x / points->size();
        this->particles[j].y0 = mean_y / points->size();
        this->particles[j].z0 = mean_z / points->size();
        this->particles[j].x = this->particles[j].x0;
        this->particles[j].y = this->particles[j].y0;
        this->particles[j].z = this->particles[j].z0;
        this->particles[j].px = this->particles[j].x;
        this->particles[j].py = this->particles[j].y;
        this->particles[j].pz = this->particles[j].z;
    }

}

/*
 * x[t+1] - x[t] = x[t] - x[t-1] + N(0,1)
 * x[t+1] = 2x[t] - x[t-1] + N(0,1)
 */
particle ParticleFilter::transition(particle p) {
    particle next_p;
    double next_x = A0*(p.x-p.x0)+A1*(p.px-p.x0)+B*gsl_ran_gaussian(rng,std_x)+p.x0;
    double next_y = A0*(p.y-p.y0)+A1*(p.py-p.y0)+B*gsl_ran_gaussian(rng,std_y)+p.y0;
    double next_z = A0*(p.z-p.z0)+A1*(p.pz-p.z0)+B*gsl_ran_gaussian(rng,std_z)+p.z0;

    next_p.x = next_x;
    next_p.y = next_y;
    next_p.z = next_z;
    next_p.px = p.x;
    next_p.py = p.y;
    next_p.pz = p.z;
    next_p.x0 = p.x0;
    next_p.y0 = p.y0;
    next_p.z0 = p.z0;
    next_p.longth = p.longth;
    next_p.width = p.width;
    next_p.height = p.height;
    next_p.likelihood = 0.0;
    next_p.observed_value.clear();

    return next_p;
}

void ParticleFilter::getLikelihood(const pcl::search::KdTree<pcl::PointXYZI> *kdtree, const pcl::PointCloud<pcl::PointXYZI> *pointset) {
	int *pf_intensity,*object_intensity;
	vector<int>pointRadiusSearch;
	vector<float>pointRadiusSquareDistance;
	pcl::PointCloud<pcl::PointXYZI> pfpoint; // the point of the particle observed at the current positiosn
	pfpoint.clear();
	for (int i = 0; i < MAX_PARTICLE_NUM; ++i) {
		pcl::PointXYZI point;
		point.x = particles[i].x;
		point.y = particles[i].y;
		point.z = particles[i].z;
		float radius;
		radius = 0.5 * sqrt(point.x*point.x + point.y*point.y + point.z*point.z);
		if(kdtree->radiusSearch(point,radius,pointRadiusSearch,pointRadiusSquareDistance) > 0){
			for (int j = 0; j < pointRadiusSearch.size(); ++j) {
				double dx = abs(pointset->points[pointRadiusSearch[j]].x - particles[i].width);
				double dy = abs(pointset->points[pointRadiusSearch[j]].y - particles[i].height);
				double dz = abs(pointset->points[pointRadiusSearch[j]].z - particles[i].longth);
				if(dx > 0.5*particles[i].width || dy > 0.5*particles[i].height || dz > 0.5*particles[i].longth){
					continue;
				}
				else{
					pfpoint.push_back(pointset->points[pointRadiusSearch[j]]);
				}
			}

			int maxSize = max(pointset->size(),pfpoint.size());
			object_intensity = new int[maxSize];
			pf_intensity = new int[maxSize];
			for (int i = 0; i < maxSize; ++i) {
				object_intensity[i] = 0;
				pf_intensity[i] = 0;
			}
			for (int k = 0; k < maxSize; ++k) {
				int intensity = round(pointset->points[k].intensity);
				object_intensity[intensity] = object_intensity[intensity] + 1;

				int intensity2 = round(pfpoint.points[k].intensity);
				pf_intensity[intensity2] = pf_intensity[intensity2] + 1;
			}

			//normalization
			float *fobject_intensity = new float[maxSize];
			float *fpf_intensity = new float[maxSize];
			for (int l = 0; l < maxSize; ++l) {
				fobject_intensity[l] = object_intensity[l]*1.0/pointset->size();
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

			numWeight = pfpoint.size()/pointset->points.size();
			similarity = numWeight * intensityWeight;

			/*
			 * need consider more methods to calculate likelihood, for example:
			 * compare particle's point distribution and object's point distribution.
			 */

			particles[i].likelihood = similarity;
		}
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

bool ParticleFilter::compareWeight(const particle &a, const particle &b) {
	return a.likelihood >= b.likelihood;
}

void ParticleFilter::resample() {
	int number = 0;
	int count = 0;
	particle *tmp = new particle[MAX_PARTICLE_NUM];

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
	frame_info(){
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
    extractor.setClusterTolerance(0.08);
    extractor.setMinClusterSize(5);
    extractor.setMaxClusterSize(5000);
    extractor.setSearchMethod(kdtree);
    extractor.setInputCloud(pointer);
    extractor.extract(cluster_indices);

    pcl::search::KdTree<pcl::PointXYZI> vkdtree;
    vkdtree = *kdtree;

//    pcl::PointXYZI point1;
//    float radius = 1.0;
//    vector<int> a;
//    vector<float> b;
//    vkdtree.radiusSearch(point1,radius,a,b);

    cout << cluster_indices.size() << " clusters" << endl;
    pcl::PointCloud<pcl::PointXYZI> mycloud;

    frame_info pinfo;
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
		for(vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++){
            pcl::PointXYZI point;
            point.x = pointer->points[*pit].x;
            point.y = pointer->points[*pit].y;
            point.z = pointer->points[*pit].z;
            point.intensity = intensity * j;
            pinfo.cluster[j-1].points.push_back(point);
            mcluster->points.push_back(point);
		    cnt ++;
		}
		cout << "###" << endl;
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

	//particle filter section
	for (int i = 0; i < pinfo.point_cluster_num; ++i) {
		pinfo.cluster[i].pf->initialParticle(&pinfo.cluster[i].points);

//		double likelihood = pinfo.cluster[i].pf->getLikelihood();
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
