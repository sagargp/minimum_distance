#include "pcl_ros/publisher.h"
#include <cmath>
#include <pcl/ModelCoefficients.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ros/conversions.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>

#define MIN(x, y) ( (x) <= (y) ) ? (x) : (y)
#define MAX(x, y) ( (x) >= (y) ) ? (x) : (y)

using namespace std;

// the cloud publisher
pcl_ros::Publisher<sensor_msgs::PointCloud2> pub;

// the closest pair of points after findDistance()
pcl::PointXYZRGB o1;
pcl::PointXYZRGB o2;

// a list of text labels for rviz
std::vector<visualization_msgs::Marker> labels;

// the input/output clouds for processing
pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputcloud;
pcl::PointCloud<pcl::PointXYZRGB> outputcloud;
sensor_msgs::PointCloud2 outputROScloud;

// global list of clustered point clouds
std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds;

// did the clustering finish?
bool ready = false;

void addLabel(pcl::PointXYZRGB p, int num)
{
	char label[5];
	sprintf(label, "%d", num); 

	visualization_msgs::Marker marker;
	marker.header.frame_id = "/map";
	marker.header.stamp = ros::Time();
	marker.ns = "block";
	marker.id = labels.size();
	marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
	marker.text = std::string(label); 
	marker.action = visualization_msgs::Marker::ADD;
	marker.pose.position.x = p.x;
	marker.pose.position.y = p.y;
	marker.pose.position.z = p.z;// + 0.05;
	marker.scale.x = 0.1;
	marker.scale.y = 0.1;
	marker.scale.z = 0.02;
	marker.color.a = 1.0;
	marker.color.r = 1.0;
	marker.color.g = 1.0;
	marker.color.b = 1.0;
	marker.lifetime = ros::Duration();
	labels.push_back(marker);
}

double findDistance(pcl::PointCloud<pcl::PointXYZRGB>::Ptr c1, pcl::PointCloud<pcl::PointXYZRGB>::Ptr c2)
{
	pcl::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::KdTreeFLANN<pcl::PointXYZRGB>);
	tree->setInputCloud(c1);

	int k = 1;
	float min = -1.0;

	std::vector<int> k_indices(1);
	std::vector<float> k_distances(1);

	for (int i = 0; i < c2->points.size(); i++)
	{
		tree->nearestKSearch(c2->points[i], k, k_indices, k_distances);

		if (k_distances[0] < min || min == -1.0)
		{
			min = k_distances[0];

			o1.x = c1->points[k_indices[0]].x;
			o1.y = c1->points[k_indices[0]].y;
			o1.z = c1->points[k_indices[0]].z;

			o2.x = c2->points[i].x;
			o2.y = c2->points[i].y;
			o2.z = c2->points[i].z;
		}
	}
	return sqrt(min);
}

void toHSV(int ri, int gi, int bi, float *h, float *s, float *v)
{
	float r, g, b;
	r = ri/255.0;
	g = gi/255.0;
	b = bi/255.0;

	float mn, mx, delta;
	mn = MIN(r, MIN(g, b));
	mx = MAX(r, MAX(g, b));

	*v = mx;
	delta = mx - mn;

	if (mx != 0)
		*s = delta/mx;
	else
	{
		*s = 0;
		*h = -1;
		return;
	}

	if (r == mx)
		*h = (g-b)/delta;
	else if (g == mx)
		*h = 2 + (b-r)/delta;
	else
		*h = 4 + (r-g)/delta;

	*h *= 60;
	if (*h < 0)
		*h += 360;
}

void colorSegment(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input, 
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr RCloud, 
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr GCloud, 
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr BCloud, 
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr YCloud, 
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr OCloud) 
{
	float r_range[][2] = {{343.0, 10.0},
		{0.10, 1.0},
		{0.10, 0.75}};

	float b_range[][2] = {{170.0, 243.0},
		{0.50, 1.0},
		{0.18, 0.70}};

	float g_range[][2] = {{61.0, 169.0},
		{0.0, 1.0},
		{0.25, 0.70}};

	float y_range[][2] = {{26.0, 60.0},
		{0.5, 1.0},
		{0.5, 1.0}};

	float o_range[][2] = {{11.0, 25.0},
		{0.5, 1.0},
		{0.6, 1.0}};
	/*
	   float r_range[][2] = { {350.0, 10.0},
	   {0.7377, 1.0},
	   {0.4431, 0.8353}};

	   float g_range[][2] = { {133.0, 157.0},
	   {0.3761, 1.0},
	   {0.2471, 0.9647}};

	   float b_range[][2] = { {220.0, 235.0},
	   {0.662, 1.0},
	   {0.2863, 0.6941}};

	   float y_range[][2] = { {21.0, 60.0},
	   {0.7917, 1.0},
	   {0.5686, 1.0}};

	   float o_range[][2] = { {13.0, 22.0},
	   {0.9228, 1.0},
	   {0.6706, 1.0}};

	   float l_range[][2] = { {68.0, 94.0},
	   {0.7124, 1.0},
	   {0.2667, 0.6784}};
	   */

	for (int i = 0; i < input->points.size(); i++)
	{
		float h, s, v;
		toHSV(input->points[i].r, input->points[i].g, input->points[i].b, &h, &s, &v);

		if ( (h > r_range[0][0]) || (h < r_range[0][1]) &&
				(s > r_range[1][0]) && (s < r_range[1][1]) &&
				(v > r_range[2][0]) && (v < r_range[2][1]))
		{
			RCloud->push_back(input->points[i]);
		}

		else if ( (h > g_range[0][0]) && (h < g_range[0][1]) &&
				(s > g_range[1][0]) && (s < g_range[1][1]) &&
				(v > g_range[2][0]) && (v < g_range[2][1]))
		{
			GCloud->push_back(input->points[i]);
		}

		else if ( (h > b_range[0][0]) && (h < b_range[0][1]) &&
				(s > b_range[1][0]) && (s < b_range[1][1]) &&
				(v > b_range[2][0]) && (v < b_range[2][1]))
		{
			BCloud->push_back(input->points[i]);
		}

		else if ( (h > y_range[0][0]) && (h < y_range[0][1]) &&
				(s > y_range[1][0]) && (s < y_range[1][1]) &&
				(v > y_range[2][0]) && (v < y_range[2][1]))
		{
			YCloud->push_back(input->points[i]);
		}

		else if ( (h > o_range[0][0]) && (h < o_range[0][1]) &&
				(s > o_range[1][0]) && (s < o_range[1][1]) &&
				(v > o_range[2][0]) && (v < o_range[2][1]))
		{
			OCloud->push_back(input->points[i]);
		}
	}
}

int clusterExtraction(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input, std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> *clouds)
{
	pcl::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::KdTreeFLANN<pcl::PointXYZRGB>);
	tree->setInputCloud(input);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> extraction;

	extraction.setClusterTolerance(0.010); // 1cm -- decreasing makes more clusters
	extraction.setMinClusterSize(120);
	extraction.setMaxClusterSize(4000);
	extraction.setSearchMethod(tree);
	extraction.setInputCloud(input);
	extraction.extract(cluster_indices);

	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);

		// keep track of the center of this cloud
		pcl::PointXYZRGB center;

		for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
			cloud_cluster->points.push_back (input->points[*pit]);

		// only find the center of the first half of the cloud. fine, i don't care
		for (int i = 0; i < cloud_cluster->size()/2; i++)
		{
			center.x += cloud_cluster->points[i].x;
			center.y += cloud_cluster->points[i].y;
			center.z += cloud_cluster->points[i].z;
		}
		center.x /= cloud_cluster->size()/2;
		center.y /= cloud_cluster->size()/2;
		center.z /= cloud_cluster->size()/2;

		cloud_cluster->width = cloud_cluster->points.size ();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		std::cout << "PointCloud has " << cloud_cluster->points.size () << " data points." << std::endl;
		clouds->push_back(cloud_cluster);
		addLabel(center, clouds->size()-1);
	}
	return cluster_indices.size();
}

void processPointCloud() 
{
	ROS_INFO("Processing point cloud...");

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr RGBcloud = inputcloud;
	std::cout << "Before filtering, cloud has: " << RGBcloud->points.size () << " data points." << std::endl;

	clouds.reserve(10);
	labels.reserve(10);

	// only for old data sets
	//	for (size_t i=0; i < RGBcloud->points.size();++i)
	//	{
	//		if (!((RGBcloud->points[i].x > -.35 && RGBcloud->points[i].x < .35) && (RGBcloud->points[i].y > -.35 && RGBcloud->points[i].y < .35)))
	//		{
	//			RGBcloud->points[i].x = 0;
	//			RGBcloud->points[i].y = 0;
	//			RGBcloud->points[i].z = 0;
	//		}
	//	}

	// Create the filtering object: downsample the dataset using a leaf size of 1cm
	pcl::VoxelGrid<pcl::PointXYZRGB> vg;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);

	vg.setInputCloud(RGBcloud);
	vg.setLeafSize (0.001f, 0.001f, 0.001f);
	vg.filter (*cloud_filtered);

	std::cout << "After filtering, cloud has: " << cloud_filtered->points.size()  << " data points." << std::endl;

	// Create the segmentation object for the planar model and set all the parameters
	pcl::SACSegmentation<pcl::PointXYZRGB> seg;
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZRGB> ());
	pcl::PCDWriter writer;

	seg.setOptimizeCoefficients (true);
	seg.setModelType (pcl::SACMODEL_PLANE);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setMaxIterations (100);
	seg.setDistanceThreshold (0.0085);//lesser the values, more points seep into the table

	// Segment the largest planar component from the remaining cloud until 30% of the points remain
	int i=0, nr_points = (int) cloud_filtered->points.size();
	while (cloud_filtered->points.size () > 0.30* nr_points)
	{
		seg.setInputCloud(cloud_filtered);
		seg.segment (*inliers, *coefficients); 

		if (inliers->indices.size () == 0)
		{
			std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
			exit(-1);	
		}

		// Extract the planar inliers from the input cloud
		pcl::ExtractIndices<pcl::PointXYZRGB> extract;
		extract.setInputCloud (cloud_filtered);
		extract.setIndices (inliers);
		extract.setNegative (false);

		// Write the planar inliers to disk
		extract.filter (*cloud_plane); //*
		std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;
		//std::cin.get();
		// Remove the planar inliers, extract the rest
		extract.setNegative (true);
		extract.filter (*cloud_filtered); //*
		std::cerr <<" The Coefficients are: " << coefficients->values[0]<< " "<< coefficients->values[1]<< " "<< coefficients->values[2]<< " " << coefficients->values[3]<< " "<< std::endl;
	}
	// color segmentation
	//
	// define this vector so we can operate over all colored point clouds in a loop
	std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> coloredClouds(5);

	// instantiate the colored clouds
	for (int i = 0; i < 5; i++)
		coloredClouds[i] = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

	// alias the colored clouds with names 
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr Rcloud = coloredClouds[0];
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr Gcloud = coloredClouds[1];
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr Bcloud = coloredClouds[2];
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr Ycloud = coloredClouds[3];
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr Ocloud = coloredClouds[4];

	// color segment the cloud using min/max ranges
	colorSegment(cloud_filtered, Rcloud, Gcloud, Bcloud, Ycloud, Ocloud);
	
	// empty out the lists before re-filling them with colorful points 
	clouds.clear();
	labels.clear();

	// now find clusters inside each of the colored point clouds 
	int num_clusters = 0;
	for (int i = 0; i < coloredClouds.size(); i++)
	{
		if (coloredClouds[i]->size() == 0)
			continue;

		cout << "Clustering colored cloud #" << i << " which has " << coloredClouds[i]->size() << " points" << endl;
		num_clusters += clusterExtraction(coloredClouds[i], &clouds);
	}
	
	std::cout << "Done clustering!" << endl;

	// now we need to take all the coloredClouds[i] and push them into one cloud object, then publish that to rviz
	for (int i = 0; i < clouds.size(); i++)
		for (int j = 0; j < clouds[i]->size(); j++)
			outputcloud.push_back(clouds[i]->points[j]);
	
	pcl::toROSMsg(outputcloud, outputROScloud); 
}

void callback(const sensor_msgs::PointCloud2ConstPtr &cloud)
{
	if ((cloud->width * cloud->height) == 0)
		return;

	ROS_INFO("Received %d data points in frame %s with the following fields: %s", (int)cloud->width * cloud->height, cloud->header.frame_id.c_str(), pcl::getFieldsList(*cloud).c_str()); 

	// convert to the native PCL format
	pcl::fromROSMsg(*cloud, *inputcloud);

	// now process as per the assignment
	processPointCloud();
	ready = true;
}

int  main (int argc, char** argv)
{
	ros::init(argc, argv, "min_dist_ros");

	ros::NodeHandle nh;
	ros::Subscriber sub;
	ros::Publisher vis_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 0);
	
	pub.advertise(nh, "min_dist_cloud", 1);
	
	inputcloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

	// listen for point cloud data on "input", and call callback()
	sub = nh.subscribe("input", 1, callback);
	ROS_INFO("Listening for incoming data on topic %s", nh.resolveName("input").c_str());
	
	int cl1, cl2;
	
	ros::Rate rate(10);
	while (nh.ok())
	{
		// "spin" the node... that is, wait for a cloud to come in, process it (via the callback), and set "ready" to true
		ros::spinOnce();

		// did we process a cloud?
		if (ready)
		{
			// keep waiting until someone subscribes to our cloud topic
			while (1)
			{
				if (pub.getNumSubscribers() > 0)
				{
					ROS_INFO("publishing...");
					outputROScloud.header.frame_id = "/map";
					pub.publish(outputROScloud);
					for (int i = 0; i < labels.size(); i++)
						vis_pub.publish(labels[i]);
					break;
				}
				ros::Duration (0.001).sleep();
			}

			// now that we've displayed the clouds and numbers, do the distance calculation
			while (1)
			{
				cout << "Enter the ID of the first cloud: ";
				cin >> cl1;
				
				cout << "Enter the ID of the second cloud: ";
				cin >> cl2;
				
				cout << "Finding distance... ";
				cout << findDistance(clouds[cl1], clouds[cl2]) << endl << endl;

				// now o1, o2 are filled with the closest points
				// so publish them on the vis_pub
				geometry_msgs::Point start;
				geometry_msgs::Point end;
				start.x = o1.x;
				start.y = o1.y;
				start.z = o1.z;
				end.x = o2.x;
				end.y = o2.y;
				end.z = o2.z;
				
				visualization_msgs::Marker arrow;
				arrow.header.frame_id = "/map";
				arrow.header.stamp = ros::Time();
				arrow.ns = "arrow";
				arrow.id = 1; 
				arrow.type = visualization_msgs::Marker::ARROW;
				arrow.action = visualization_msgs::Marker::ADD;
				arrow.points.push_back(start);
				arrow.points.push_back(end);
				arrow.scale.x = 0.01;
				arrow.scale.y = 0.05;
				arrow.color.a = 1.0;
				arrow.color.r = 1.0;
				arrow.color.g = 1.0;
				arrow.color.b = 1.0;
				arrow.lifetime = ros::Duration();
				vis_pub.publish(arrow);
				
				ros::Duration (0.001).sleep();
			}
		}
		rate.sleep();
	}
}
