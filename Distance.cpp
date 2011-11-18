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
#include <math.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>

pcl::PointXYZRGB o1;
pcl::PointXYZRGB o2;
std::vector<pcl::PointXYZRGB> labels;

void viewerOneOff(pcl::visualization::PCLVisualizer& viewer)
{
	viewer.setBackgroundColor (0.0, 0.0, 0.0);
	viewer.removeShape("line", 0);
	viewer.addArrow(o1, o2, 1.0, 0.0, 0.0, "line", 0);
}

double findDistanceKDTree(pcl::PointCloud<pcl::PointXYZRGB>::Ptr c1, pcl::PointCloud<pcl::PointXYZRGB>::Ptr c2)
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
			min == k_distances[0];

			o1.x = c1->points[k_indices[0]].x;
			o1.y = c1->points[k_indices[0]].y;
			o1.z = c1->points[k_indices[0]].z;
			
			o2.x = c2->points[i].x;
			o2.y = c2->points[i].y;
			o2.z = c2->points[i].z;
		}
	}
	return min;
}

double findDistance(pcl::PointCloud<pcl::PointXYZRGB>::Ptr c1, pcl::PointCloud<pcl::PointXYZRGB>::Ptr c2)
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

void label(pcl::visualization::PCLVisualizer& viewer)
{
	for(size_t i = 0; i < labels.size(); i++)
	{
		labels[i].z = labels[i].z - .05;
		char f[5];
		sprintf(f,"%d",i);
		viewer.addText3D (f, labels[i], 0.02,0,0,0,f,0);
	}
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

	for (size_t i=0; i < RGBcloud->points.size();++i)
	{
		if (!((RGBcloud->points[i].x > -.35 && RGBcloud->points[i].x < .35) && (RGBcloud->points[i].y > -.35 && RGBcloud->points[i].y < .35)))///Bounding
		{
			RGBcloud->points[i].x = 0;
			RGBcloud->points[i].y = 0;
			RGBcloud->points[i].z = 0;
		}
	}

	// Create the filtering object: downsample the dataset using a leaf size of 1cm
	pcl::VoxelGrid<pcl::PointXYZRGB> vg;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
	vg.setInputCloud (RGBcloud);
	vg.setLeafSize (0.001f, 0.001f, 0.001f);
	vg.filter (*cloud_filtered);
	std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl;

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
	int i=0, nr_points = (int) cloud_filtered->points.size ();
	while (cloud_filtered->points.size () > 0.3 * nr_points)
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
		
		// Remove the planar inliers, extract the rest
		extract.setNegative (true);
		extract.filter (*cloud_filtered); //*
		std::cerr <<" The Coefficients are: " << coefficients->values[0]<< " "<< coefficients->values[1]<< " "<< coefficients->values[2]<< " " << coefficients->values[3]<< " "<< std::endl;
	}

	////////////////////////Table has been awesomely removed ///////////////////////
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr Rcloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr Gcloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr Bcloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr Ycloud (new pcl::PointCloud<pcl::PointXYZRGB>);

	std::cout << "PointCloud after filtering has: " << Gcloud->points.size ()  << " data points." << std::endl;
	
	for (size_t i=0; i < cloud_filtered->points.size();++i)///green extraction
	{
		if (!(cloud_filtered->points[i].g > cloud_filtered->points[i].r && cloud_filtered->points[i].g > cloud_filtered->points[i].b))///Bounding
		{
			cloud_filtered->points[i].z = cloud_filtered->points[i].z + 3;

		}
	}

	pcl::PassThrough<pcl::PointXYZRGB> pass;
	pass.setInputCloud (cloud_filtered);
	pass.setFilterFieldName ("z");
	pass.setFilterLimits (-1.0, 1.0);
	pass.filter (*Gcloud);

	for (size_t i=0; i < cloud_filtered->points.size();++i)//blue extraction
	{
		if (!(cloud_filtered->points[i].b > cloud_filtered->points[i].g && cloud_filtered->points[i].b > cloud_filtered->points[i].r))///Bounding
		{
			cloud_filtered->points[i].z = cloud_filtered->points[i].z - 3;
		}
	}

	pcl::PassThrough<pcl::PointXYZRGB> pass1;
	pass1.setInputCloud (cloud_filtered);
	pass1.setFilterFieldName ("z");
	pass1.setFilterLimits (2.0, 4.0);
	pass1.filter (*Bcloud);
	
	for (size_t i=0; i < Bcloud->points.size();++i)//blue extraction
	{
		Bcloud->points[i].z = Bcloud->points[i].z - 3;
	}

	for (size_t i=0; i < cloud_filtered->points.size();++i)// yellow extraction
	{
		if (!(int(cloud_filtered->points[i].r) > int(cloud_filtered->points[i].g) && int(cloud_filtered->points[i].g) - int(cloud_filtered->points[i].b) > 30))///Bounding
		{
			cloud_filtered->points[i].z = cloud_filtered->points[i].z + 6;
		}
	}

	pcl::PassThrough<pcl::PointXYZRGB> pass2;
	pass2.setInputCloud (cloud_filtered);
	pass2.setFilterFieldName ("z");
	pass2.setFilterLimits (-1.0,1.0);
	pass2.filter (*Ycloud);

	pcl::PassThrough<pcl::PointXYZRGB> pass3;
	pass3.setInputCloud (cloud_filtered);
	pass3.setFilterFieldName ("z");
	pass3.setFilterLimits (5.0,7.0);
	pass3.filter (*Rcloud);

	for (size_t i=0; i < Rcloud->points.size();++i)//Red correction 
	{
		Rcloud->points[i].z = Rcloud->points[i].z - 6;
	}

	std::cerr<<"Waiting 1 "<<std::endl;
	
	// Creating the KdTree object for the search method of the extraction
	pcl::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::KdTreeFLANN<pcl::PointXYZRGB>);
	tree->setInputCloud ( Rcloud);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
	ec.setClusterTolerance (0.01); // 1cm /// decreasing makes more clusters
	ec.setMinClusterSize (100);
	ec.setMaxClusterSize (4000);
	ec.setSearchMethod (tree);
	ec.setInputCloud(  Rcloud);
	ec.extract (cluster_indices);

	int j = 0;

	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);

		for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
			cloud_cluster->points.push_back ( Rcloud->points[*pit]); //*

		cloud_cluster->width = cloud_cluster->points.size ();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
		clouds.push_back(cloud_cluster);
		labels.push_back(cloud_cluster->points[0]);
		j++;
	}

	/////////////////playing around//////////////////////
	// Creating the KdTree object for the search method of the extraction
	pcl::KdTree<pcl::PointXYZRGB>::Ptr treeG (new pcl::KdTreeFLANN<pcl::PointXYZRGB>);
	tree->setInputCloud ( Gcloud);

	std::vector<pcl::PointIndices> cluster_indicesG;
	pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ecG;
	ecG.setClusterTolerance (0.01); // 1cm /// decreasing makes more clusters
	ecG.setMinClusterSize (100);
	ecG.setMaxClusterSize (4000);
	ecG.setSearchMethod (treeG);
	ecG.setInputCloud(  Gcloud);
	ecG.extract (cluster_indicesG);

	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indicesG.begin (); it != cluster_indicesG.end (); ++it)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);

		for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
			cloud_cluster->points.push_back ( Gcloud->points[*pit]); //*

		cloud_cluster->width = cloud_cluster->points.size ();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
		clouds.push_back(cloud_cluster);
		labels.push_back(cloud_cluster->points[0]);
		j++;
	}
	//////////////////////////////////////

	// Creating the KdTree object for the search method of the extraction
	pcl::KdTree<pcl::PointXYZRGB>::Ptr treeB (new pcl::KdTreeFLANN<pcl::PointXYZRGB>);
	tree->setInputCloud ( Bcloud);

	std::vector<pcl::PointIndices> cluster_indicesB;
	pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ecB;
	ecB.setClusterTolerance (0.01); // 1cm /// decreasing makes more clusters
	ecB.setMinClusterSize (100);
	ecB.setMaxClusterSize (4000);
	ecB.setSearchMethod (treeB);
	ecB.setInputCloud(  Bcloud);
	ecB.extract (cluster_indicesB);

	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indicesB.begin (); it != cluster_indicesB.end (); ++it)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);

		for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
			cloud_cluster->points.push_back ( Bcloud->points[*pit]); //*

		cloud_cluster->width = cloud_cluster->points.size ();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
		clouds.push_back(cloud_cluster);
		labels.push_back(cloud_cluster->points[0]);
		j++;
	}
	/////////////////////////////////////

	//Creating the KdTree object for the search method of the extraction
	pcl::KdTree<pcl::PointXYZRGB>::Ptr treeY (new pcl::KdTreeFLANN<pcl::PointXYZRGB>);
	tree->setInputCloud ( Ycloud);

	std::vector<pcl::PointIndices> cluster_indicesY;
	pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ecY;
	ecY.setClusterTolerance (0.01); // 1cm /// decreasing makes more clusters
	ecY.setMinClusterSize (100);
	ecY.setMaxClusterSize (4000);
	ecY.setSearchMethod (treeY);
	ecY.setInputCloud(  Ycloud);
	ecY.extract (cluster_indicesY);

	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indicesY.begin (); it != cluster_indicesY.end (); ++it)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);

		for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
			cloud_cluster->points.push_back ( Ycloud->points[*pit]); //*

		cloud_cluster->width = cloud_cluster->points.size ();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
		clouds.push_back(cloud_cluster);
		labels.push_back(cloud_cluster->points[0]);
		j++;
	}

	std::cerr<<"Waiting 3 "<<std::endl;
	pcl::visualization::CloudViewer viewer("Cloud Viewer");
	viewer.showCloud(cloudF, "Full cloud");

	viewer.runOnVisualizationThreadOnce (label);

	char c_name[] = "Cloud No: ";
	char cloud_name [20];
	std::cerr<<"Total Clusters: "<<j<<std::endl;

	for (size_t k = 0; k < j ; k++)
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
		//while (!viewer.wasStopped ())
		//{
		// }		
		viewer.runOnVisualizationThreadOnce (viewerOneOff);
	}

	while (!viewer.wasStopped ())
	{
	}
	return 0;
}
