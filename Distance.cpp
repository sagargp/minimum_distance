#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <cmath>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>

#define MIN(x, y) ( (x) <= (y) ) ? (x) : (y)
#define MAX(x, y) ( (x) >= (y) ) ? (x) : (y)

pcl::PointXYZRGB o1;
pcl::PointXYZRGB o2;
std::vector<pcl::PointXYZRGB> labels;

void  viewerOneOff ( pcl::visualization::PCLVisualizer& viewer)
{
	viewer.setBackgroundColor (0.0, 0.0, 0.0);
	viewer.removeShape("line", 0);
	viewer.addArrow(o1, o2, 1.0, 0.0, 0.0, "line", 0);
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

double findDistanceSlow(pcl::PointCloud<pcl::PointXYZRGB>::Ptr c1, pcl::PointCloud<pcl::PointXYZRGB>::Ptr c2)
{
	float min = -1.0;
	float dist = 0.0;
	float x, y, z;

	for (int c1_point = 0; c1_point < c1->points.size(); c1_point++)
	{
		for (int c2_point = 0; c2_point < c2->points.size(); c2_point++)
		{
			x = pow((c1->points[c1_point].x) - (c2->points[c2_point].x), 2);
			y = pow((c1->points[c1_point].y) - (c2->points[c2_point].y), 2);
			z = pow((c1->points[c1_point].z) - (c2->points[c2_point].z), 2);

			dist = sqrt(x + y + z);

			if (dist < min || min == -1.0) {
				min = dist;

				o1.x = c1->points[c1_point].x;
				o1.y = c1->points[c1_point].y;
				o1.z = c1->points[c1_point].z;

				o2.x = c2->points[c2_point].x;
				o2.y = c2->points[c2_point].y;
				o2.z = c2->points[c2_point].z;            
			}
		}
	}
	return min;
} 

void label (pcl::visualization::PCLVisualizer& viewer)
{
	for(size_t i = 0; i < labels.size(); i++)
	{
	  labels[i].z = labels[i].z - .05;
	  char f[5];
	  sprintf(f,"%d",i);
	  viewer.addText3D (f, labels[i], 0.02,1,1,1,f,0);
	}
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
		{0.30, 1.0},
		{0.40, 0.75}};

	float g_range[][2] = {{75.0, 190.0},
		{0.0, 1.0},
		{0.25, 0.70}};

	float b_range[][2] = {{225.0, 243.0},
		{0.50, 1.0},
		{0.18, 0.70}};

	float y_range[][2] = {{29.0, 60.0},
		{0.5, 1.0},
		{0.5, 1.0}};

	float o_range[][2] = {{13.0, 20.0},
		{0.90, 1.0},
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

int clusterExtraction(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input, std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> *clouds, std::vector<pcl::PointXYZRGB> *labels)
{
	pcl::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::KdTreeFLANN<pcl::PointXYZRGB>);
	tree->setInputCloud(input);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> extraction;

	extraction.setClusterTolerance(0.005); // 1cm -- decreasing makes more clusters
	extraction.setMinClusterSize(100);
	extraction.setMaxClusterSize(4000);
	extraction.setSearchMethod(tree);
	extraction.setInputCloud(input);
	extraction.extract(cluster_indices);

	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);

		for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
			cloud_cluster->points.push_back (input->points[*pit]);

		cloud_cluster->width = cloud_cluster->points.size ();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		std::cout << "PointCloud has " << cloud_cluster->points.size () << " data points." << std::endl;
		clouds->push_back(cloud_cluster);
		labels->push_back(cloud_cluster->points[0]);
	}
	return cluster_indices.size();
}

int  main (int argc, char** argv)
{
	// Read in the cloud data
	pcl::PCDReader reader;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr RGBcloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	reader.read (argv[1], *RGBcloud);

	std::cout << "RGBPointCloud before filtering has: " << RGBcloud->points.size () << " data points." << std::endl; //*

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudF (new pcl::PointCloud<pcl::PointXYZRGB>);
	reader.read (argv[1], *cloudF);

	std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds;
	clouds.reserve(10);
	labels.reserve(10);

	//this is only for the old data sets
	//for (size_t i=0; i < RGBcloud->points.size();++i)
	//{
	//	if (!((RGBcloud->points[i].x > -.35 && RGBcloud->points[i].x < .35) && (RGBcloud->points[i].y > -.35 && RGBcloud->points[i].y < .35)))
	//	{
	//		RGBcloud->points[i].x = 0;
	//		RGBcloud->points[i].y = 0;
	//		RGBcloud->points[i].z = 0;
	//	}
	//}

	// Create the filtering object: downsample the dataset using a leaf size of 1cm
	pcl::VoxelGrid<pcl::PointXYZRGB> vg;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_color_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);	
	vg.setInputCloud (RGBcloud);
	vg.setLeafSize (0.001f, 0.001f, 0.001f);
	vg.filter (*cloud_filtered);
	//cloud_filtered = cloud;
	std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl;

	//pcl::visualization::CloudViewer viewer("Cloud Viewer");
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
	while (cloud_filtered->points.size () > 0.99 * nr_points)
	{
		seg.setInputCloud(cloud_filtered);
		seg.segment (*inliers, *coefficients); //*

		if (inliers->indices.size () == 0)
		{
			std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
			break;
		}

		// Extract the planar inliers from the input cloud
		pcl::ExtractIndices<pcl::PointXYZRGB> extract;
		extract.setInputCloud (cloud_filtered);
		extract.setIndices (inliers);
		extract.setNegative (false);

		// Write the planar inliers to disk
		extract.filter (*cloud_plane); //*
		std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;
		//viewer.showCloud(cloud_plane, "cloud_name");
		//std::cin.get();
		// Remove the planar inliers, extract the rest
		extract.setNegative (true);
		extract.filter (*cloud_filtered); //*
		std::cerr <<" The Coefficients are: " << coefficients->values[0]<< " "<< coefficients->values[1]<< " "<< coefficients->values[2]<< " " << coefficients->values[3]<< " "<< std::endl;
	}

	// extract clusters from the filtered cloud
	clusterExtraction(cloud_filtered, &clouds, &labels);	

	// color segmentation code
	std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> coloredClouds(5);

	for (int i = 0; i < 5; i++)
		coloredClouds[i] = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
	
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr Rcloud = coloredClouds[0];
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr Gcloud = coloredClouds[1];
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr Bcloud = coloredClouds[2];
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr Ycloud = coloredClouds[3];
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr Ocloud = coloredClouds[4];

	// color segment each cluster found from the filtered cloud
	for (int i = 0; i < clouds.size(); i++)
		colorSegment(clouds[i], Rcloud, Gcloud, Bcloud, Ycloud, Ocloud);

	// empty out the lists for now
	clouds.clear();
	labels.clear();

	// now find clusters inside each of the colored blobs
	int num_clusters = 0;
	for (int i = 0; i < coloredClouds.size(); i++)
		num_clusters += clusterExtraction(coloredClouds[i], &clouds, &labels);

	// display what we've found	
	std::cerr<<"Waiting 3 "<<std::endl;
	pcl::visualization::CloudViewer viewer("Cloud Viewer");

	viewer.runOnVisualizationThreadOnce (label);

	char c_name[] = "Cloud No: ";
	char cloud_name [20];
	std::cerr << "Total Clusters: " << num_clusters << std::endl;

	for (size_t k = 0; k < num_clusters; k++)
	{	
		std::sprintf(cloud_name,"%s%d",c_name,k);
		viewer.showCloud(clouds[k], cloud_name);
	}

	std::cin.get();
	int xx, yy;
	while(1)
	{
		cout << "Enter the label of the first cloud: ";
		cin >> xx;

		cout << "Enter the label of the second cloud: ";
		cin >> yy;

		cout << "Finding distance...";
		cout << findDistance(clouds[xx], clouds[yy]) << endl;

		viewer.runOnVisualizationThreadOnce (viewerOneOff);
	}
	
	while (!viewer.wasStopped ());

	return 0;
}
