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

pcl::PointXYZ o1;
pcl::PointXYZ o2;
std::vector<pcl::PointXYZ> labels;

void  viewerOneOff ( pcl::visualization::PCLVisualizer& viewer)
{
	viewer.setBackgroundColor (0.0, 0.0, 0.0);
	viewer.removeShape("line", 0);
	viewer.addArrow(o1, o2, 1.0, 0.0, 0.0, "line", 0);
}

double findDistance(pcl::PointCloud<pcl::PointXYZ>::Ptr c1, pcl::PointCloud<pcl::PointXYZ>::Ptr c2)
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

int  main (int argc, char** argv)
{
  // Read in the cloud data
  pcl::PCDReader reader;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  reader.read (argv[1], *cloud);

  std::cout << "PointCloud before filtering has: " << cloud->points.size () << " data points." << std::endl; //*

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudF (new pcl::PointCloud<pcl::PointXYZRGB>);
  reader.read (argv[1], *cloudF);

  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds;
  clouds.reserve(10);
  labels.reserve(10);

  for (size_t i=0; i < cloud->points.size();++i)
  {
    if (!(cloud->points[i].x > -.3 && cloud->points[i].x < .3))///Bounding
    {
      cloud->points[i].x = 0;
      cloud->points[i].y = 0;
      cloud->points[i].z = 0;
    }
  }

	

  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_color_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);	
  vg.setInputCloud (cloud);
  vg.setLeafSize (0.001f, 0.001f, 0.001f);
  vg.filter (*cloud_filtered);
	//cloud_filtered = cloud;
  std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl;


	pcl::visualization::CloudViewer viewer("Cloud Viewer");



  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::PCDWriter writer;
  
  
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (100);
  seg.setDistanceThreshold (0.008);

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
    pcl::ExtractIndices<pcl::PointXYZ> extract;
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
  
  // Creating the KdTree object for the search method of the extraction
  pcl::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::KdTreeFLANN<pcl::PointXYZ>);
  tree->setInputCloud (cloud_filtered);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (0.02); // 2cm
  ec.setMinClusterSize (100);
  ec.setMaxClusterSize (40000);
  ec.setSearchMethod (tree);
  ec.setInputCloud( cloud_filtered);
  ec.extract (cluster_indices);

  int j = 0;
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
    
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
      cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*
    
    cloud_cluster->width = cloud_cluster->points.size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
    clouds.push_back(cloud_cluster);
    labels.push_back(cloud_cluster->points[0]);
    j++;
  }


 // pcl::visualization::CloudViewer viewer("Cloud Viewer");//Initialized above

	//viewer.showCloud(cloudF, "Full cloud");
	viewer.showCloud(cloudF, "Full cloud");
  viewer.runOnVisualizationThreadOnce (label);
	



viewer.showCloud(clouds[0], "Full cloud21");
viewer.showCloud(clouds[1], "Full cloud32");
viewer.showCloud(clouds[2], "Full cloud23");
viewer.showCloud(clouds[3], "Full cloud34");
viewer.showCloud(clouds[4], "Full cloud25");
viewer.showCloud(clouds[5], "Full cloud36");
viewer.showCloud(clouds[6], "Full cloud27");
viewer.showCloud(clouds[7], "Full cloud38");


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

  while (!viewer.wasStopped ())
  {
  }
  return 0;
}
