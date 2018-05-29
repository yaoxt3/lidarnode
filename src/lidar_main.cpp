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

using namespace std;
using namespace cv;


class ParticleFilter{
	
};

struct particle_state{
	double x;
	double y;
	double z;
	particle_state(){
		x = 0.0;
		y = 0.0;
		z = 0.0;
	}
};


/*
 * @state: particle's position (x,y,z)
 * @observed_value: particle's observed value in (x,y,z)
 * @likelihood: the likelihood between the current particle and the initial object
 */
struct particle_filter{
	particle_state state;
	pcl::PointCloud<pcl::PointXYZI> observed_value;
	double likelihood;
	particle_filter(){
		observed_value.clear();
		likelihood = 0.0;
	}
};

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
	particle_filter *pf;
	pcl::PointCloud<pcl::PointXYZI> points;
	cluster_info():particle_num(30){
		center_x = 0.0;
		center_y = 0.0;
		center_z = 0.0;
		height = 0.0;
		width = 0.0;
		longth = 0.0;
		pf = new particle_filter[particle_num];
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
    // function
    void cluster_function(const pcl::PointCloud<pcl::PointXYZI> *pointset);
    void processPointCloud(const sensor_msgs::PointCloud2 &scan);
    void TrackingModel(const pcl::PointCloud<pcl::PointXYZI> *pointset);
    float calculate_distance2(const pcl::PointXYZI a, const pcl::PointXYZI b);
    void find_center(const pcl::PointCloud<pcl::PointXYZI> *pointset, pcl::PointCloud<pcl::PointXYZI> *cluster_center);
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
	//pinfo.cluster = *mcluster;
	//pinfo.cluster[j-1].points = *mcluster;
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
