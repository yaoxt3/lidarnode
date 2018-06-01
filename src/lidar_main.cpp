/*
  author: yxt
  create it in 2018-1-19
*/

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <ctime>
#include <sstream>
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
	double height;
	double width;
	double longth;
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
 * @initialParticle: initialize particle's state
 * @transition: update the current particle's state by previous state
 * @normalizeWeights: normalize particle's weights
 * @resample: resampling the particles to keep the diversity of particles
 * @getLikelihood: calculate the similarity between the particle's observed value and the tracked object
 * @objectid: the tracked object id
 * @MAX_PARTICLE_NUM: maximum partilce number
 * @particles: particle set
 * @rng: gsl library variable to generate guassian-distribution number
 */
class ParticleFilter{
public:
	ParticleFilter();
	void initialParticle(pcl::PointCloud<pcl::PointXYZI> points);
	void transition();
	void normalizeWeights();
	void resample();
	double getLikelihood();

	int objectid;
	const int MAX_PARTICLE_NUM;
	particle *particles;
	gsl_rng *rng;
};

ParticleFilter::ParticleFilter():MAX_PARTICLE_NUM(30){
    particles=new particle[MAX_PARTICLE_NUM];
}
//initialize the particles
void ParticleFilter::initialParticle(pcl::PointCloud<pcl::PointXYZI> points) {
    //get the size of tracking object
    size_t size = points.size();
    pcl::PointXYZI center;
    center.x=center.y=center.z=0;
    float min_x=0,min_y=0,min_z=0;
    float max_x=0,max_y=0,max_z=0;
    for(size_t i=0;i<size;i++){
        center.x+=points.points[i].x;
        center.y+=points.points[i].y;
        center.z+=points.points[i].z;
        if(min_x>points.points[i].x) min_x=points.points[i].x;
        if(min_y>points.points[i].y) min_y=points.points[i].y;
        if(min_z>points.points[i].z) min_z=points.points[i].z;
        if(max_x>points.points[i].x) max_x=points.points[i].x;
        if(max_y>points.points[i].y) max_y=points.points[i].y;
        if(max_z>points.points[i].z) max_z=points.points[i].z;
    }
    center.x/=(float)size;
    center.y/=(float)size;
    center.z/=(float)size;
    float length=max_x-min_x;
    float width=max_y-min_y;
    float height=max_z-min_z;
    //initialize the particles
    const gsl_rng_type *T;
    gsl_rng_env_setup();
    T=gsl_rng_default;
    rng=gsl_rng_alloc(T);
    for(int i=0;i<MAX_PARTICLE_NUM;i++){
        particles[i].height=height;
        particles[i].width=width;
        particles[i].longth=length;
        particles[i].x=center.x+gsl_ran_gaussian(rng,0.4);
        particles[i].y=center.y+gsl_ran_gaussian(rng,0.4);
        particles[i].z=center.z+gsl_ran_gaussian(rng,0.4);
    }
}
//transit the particles from previos frame
void ParticleFilter::transition() {
    for(int i=0;i<MAX_PARTICLE_NUM;i++){

    }
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
	const int particle_num;
	double center_x;
	double center_y;
	double center_z;
	double height;
	double width;
	double longth;
	ParticleFilter *pf;
	pcl::PointCloud<pcl::PointXYZI> points;
	cluster_info():particle_num(30){
		center_x = 0.0;
		center_y = 0.0;
		center_z = 0.0;
		height = 0.0;
		width = 0.0;
		longth = 0.0;
		//pf = new ParticleFilter;
		points.clear();
	}
};

/*
 * @point_cluster_num: the point cluster number
 * @index: the vector index of different point clusters
 * @cluster: store all pointcloud clusters in the current frame, and differentiate pointcloud clusters by different intensities
 */
struct frame_info{
	int point_cluster_num;
//	int *index;
//	pcl::PointCloud<pcl::PointXYZI> cluster;
	cluster_info *cluster;
	frame_info(){
		point_cluster_num = 0;
	}
};

class Lidar_node{
public:
    Lidar_node();
	// function
	void cluster_function(const pcl::PointCloud<pcl::PointXYZI> *pointset);
	void processPointCloud(const sensor_msgs::PointCloud2 &scan);
	void TrackingModel(const pcl::PointCloud<pcl::PointXYZI> *pointset);
	float calculate_distance2(const pcl::PointXYZI a, const pcl::PointXYZI b);
	void find_center(const pcl::PointCloud<pcl::PointXYZI> *pointset, pcl::PointCloud<pcl::PointXYZI> *cluster_center);
private:
    ros::NodeHandle node_handle_;
    ros::Subscriber points_node_sub_;
    ros::Publisher points_node_pub_;
    ros::Publisher test_points_pub_;

    // set left-right threshold for y-axis in lidar coordinate
    int cluster_k; // cluster center's number
	int frame_id;
	const int frame_num;
	const int searchNum;
    float left_threshold;
    float right_threshold;
    float forward_threshold;
    float forward_max_threshold;
    vector<float > vmin_dist;
    vector<frame_info > frame_points; // record three frames information
};


Lidar_node::Lidar_node():searchNum(100),frame_num(3){ // error : node_handle_("~")
	ROS_INFO("In constructed function.");
    left_threshold = 2.5;
    right_threshold = 2;
    forward_threshold = 0.25;
    forward_max_threshold = 3;
    cluster_k = 6;
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

    cout << cluster_indices.size() << " clusters" << endl;
    pcl::PointCloud<pcl::PointXYZI> mycloud;

    frame_info pinfo;
    pinfo.point_cluster_num = cluster_indices.size();
    pinfo.cluster = new cluster_info[cluster_indices.size()];

	cout << "@@@" << endl;
    int j = 1;
    float intensity = 255.0f / cluster_indices.size();
    int point_nums = 0;

	pcl::PointCloud<pcl::PointXYZI>::Ptr mcluster(new pcl::PointCloud<pcl::PointXYZI>); // use different intensities to differentiate point clusters
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
	//	pinfo.cluster[i].pf->initialParticle();
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

/*
  -cluster function-
  algorithm: kmeans++ & ISODATA
*/
void Lidar_node::find_center(const pcl::PointCloud<pcl::PointXYZI> *pointset, pcl::PointCloud<pcl::PointXYZI> *cluster_center) {
    size_t size = pointset->size();
    int init_pos = rand()%size; // initialization position
    cluster_center->points.push_back(pointset->points[init_pos]);

  /*
  kmeans++:
  1.for each point, find the nearest cluster center, calculate the distance 'dist(i)'
  2.for i-point, the bigger dist(i) is, the bigger probabilty of i-point will be selected
  3.if i-point is selected, it becomes a new cluster center
  4.continue above steps, until k cluster centers are selected
  5.kmeans algorithm beginning...
  */
    int num = 1;
    while (num < cluster_k) {
      int pos = 0;

      float dist_sum = 0.0;
      float probabilty = 0.0;
      float prob_sum = 0.0;
      vmin_dist.clear();
      for (size_t i = 0; i < size; i++) {
        float min_dist = 999999999;
        for (size_t j = 0; j < cluster_center->size(); j++) { // find the max distance between i-th point and k cluster centers
          float dist = calculate_distance2(cluster_center->points[j],pointset->points[i]);
          if (min_dist > dist) {
            min_dist = dist;
          }
        }
        vmin_dist.push_back(min_dist);
        dist_sum += min_dist;
      }
      for (size_t i = 0; i < size; i++) {
        vmin_dist[i] /= dist_sum;
      }
      probabilty = (rand()%size)/(float)size;
      //cout << "prob: " << probabilty << endl;
      size_t vsize = vmin_dist.size();
      for (size_t i = 0; i < vsize; i++) {
        prob_sum += vmin_dist[i];
        if (prob_sum >= probabilty) {
          //cout << "prob_sum: " << prob_sum << endl;
          pos = i;  // get a new cluster center
          break;
        }
      }
      //cout << " pos: " << pos << endl;
      cluster_center->points.push_back(pointset->points[pos]);
      num++;
    }
    //cout << cluster_center->size() << endl;
    //cout << "-----------------------" << endl;
}

void Lidar_node::cluster_function(const pcl::PointCloud<pcl::PointXYZI> *pointset){
    pcl::PointCloud<pcl::PointXYZI> cluster_center; // cluster centers
    find_center(pointset,&cluster_center);
    //cout<<"cluster:"<<cluster_center.size()<<endl;

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
