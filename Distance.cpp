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
void 
viewerOneOff ( pcl::visualization::PCLVisualizer& viewer)
{
    viewer.setBackgroundColor (0.5, 0.8, 1.0);
    viewer.addLine(o1, o2, "line", 0);
    std::cout << "i only run once" << std::endl;
}


int 
main (int argc, char** argv)
{
  // Read in the cloud data
  pcl::PCDReader reader;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  reader.read ("group2_1.pcd", *cloud);
  std::cout << "PointCloud before filtering has: " << cloud->points.size () << " data points." << std::endl; //*
/*pcl::PointCloud<pcl::PointXYZ>::Ptr cloudS[6];     ///////////////////////////////////////////////////
*cloudS = new pcl::PointCloud<pcl::PointXYZ>;	*/	///////////////////////////////////////////////////
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudF (new pcl::PointCloud<pcl::PointXYZRGB>);
  reader.read ("group2_1.pcd", *cloudF);
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud0 (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud3 (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud4 (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud5 (new pcl::PointCloud<pcl::PointXYZ>);



 

for (size_t i=0; i < cloud->points.size();++i)

{
if (!(cloud->points[i].x > -.3 && cloud->points[i].x < .3))
{

cloud->points[i].x = 0;
cloud->points[i].y = 0;
cloud->points[i].z = 0;

}


}





  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  vg.setInputCloud (cloud);
  vg.setLeafSize (0.001f, 0.001f, 0.001f);
  vg.filter (*cloud_filtered);
  std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl; //*





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
  seg.setDistanceThreshold (0.02);

  int i=0, nr_points = (int) cloud_filtered->points.size ();
  while (cloud_filtered->points.size () > 0.3 * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
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
 std::cerr <<" The Coefficients are: " << coefficients->values[0]<< " "<< coefficients->values[1]<< " "<< coefficients->values[2]<< " " 
      << coefficients->values[3]<< " "<< std::endl;

  }

  // Creating the KdTree object for the search method of the extraction
  pcl::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::KdTreeFLANN<pcl::PointXYZ>);
  tree->setInputCloud (cloud_filtered);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (0.02); // 2cm
  ec.setMinClusterSize (10);
  ec.setMaxClusterSize (25000);
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
    std::stringstream ss;
    ss << "cloud_cluster_" << j << ".pcd";
    writer.write<pcl::PointXYZ> (ss.str (), *cloud_cluster, false); //*
//pcl::io::loadPCDFile<pcl::PointXYZ> (ss.str(), *cloudS[j]);
    j++;
  }
 
pcl::io::loadPCDFile<pcl::PointXYZ> ("cloud_cluster_0.pcd", *cloud0);   ///////////////////////////////////////////////////
pcl::io::loadPCDFile<pcl::PointXYZ> ("cloud_cluster_1.pcd", *cloud1);   ///////////////////////////////////////////////////
pcl::io::loadPCDFile<pcl::PointXYZ> ("cloud_cluster_2.pcd", *cloud2);   ///////////////////////////////////////////////////
pcl::io::loadPCDFile<pcl::PointXYZ> ("cloud_cluster_3.pcd", *cloud3);   ///////////////////////////////////////////////////
pcl::io::loadPCDFile<pcl::PointXYZ> ("cloud_cluster_4.pcd", *cloud4);   ///////////////////////////////////////////////////
pcl::io::loadPCDFile<pcl::PointXYZ> ("cloud_cluster_5.pcd", *cloud5);   ///////////////////////////////////////////////////

int counter[6];

counter[0] = cloud0->points.size();
counter[1] = cloud1->points.size();
counter[2] = cloud2->points.size();
counter[3] = cloud3->points.size();
counter[4] = cloud4->points.size();
counter[5] = cloud5->points.size();
float x,y,z,distance;
float low_distance=10;
//x1 = (cloud1->points[100].y); x2 = (cloud2->points[100].y);

//pcl::PointXYZ o1;
//pcl::PointXYZ o2;
for (size_t N=1; N<6; ++N)
for (size_t j=0; j < counter[0];++j)
{

for (size_t i=0; i < counter[4];++i)

{

x = ((cloud1->points[i].x)-(cloud2->points[j].x))*((cloud1->points[i].x)-(cloud2->points[j].x));
y = ((cloud1->points[i].y)-(cloud2->points[j].y))*((cloud1->points[i].y)-(cloud2->points[j].y));
z = ((cloud1->points[i].z)-(cloud2->points[j].z))*((cloud1->points[i].z)-(cloud2->points[j].z));
distance = sqrt(x+y+z);
//std::cerr<<"Distance "<< distance<<std::endl;

if (distance< low_distance)
{
low_distance = distance;
    o1.x = cloud1->points[i].x;
    o1.y = cloud1->points[i].y;
    o1.z = cloud1->points[i].z;
    o2.x = cloud2->points[j].x;
    o2.y = cloud2->points[j].y;
    o2.z = cloud2->points[j].z;
}
}
}

std::cerr<<"Distance "<< low_distance <<std::endl;

pcl::visualization::CloudViewer viewer("Cloud Viewer");
   
viewer.runOnVisualizationThreadOnce (viewerOneOff);


//viewer.runOnVisualizationThreadOnce (viewerOneOff);

int xx;
while(1)
{
//viewer.showCloud(0, "cloud1");
cin >> xx;

if(xx==0)
{
viewer.showCloud(cloud0, "cloud0");
}

if(xx==1)
{ 
  viewer.showCloud(cloud1,"cloud1");
 } 
   if(xx==2)
{
viewer.showCloud(cloud2, "cloud2");
}

if(xx==3)
{ 
  viewer.showCloud(cloud3,"cloud3");
 } 
  
if(xx==4)
{
viewer.showCloud(cloud4, "cloud4");
}

if(xx==5)
{ 
  viewer.showCloud(cloud5,"cloud5");
 } 
    

if(xx==9)
{ 
  viewer.showCloud(cloudF,"cloudF");
 } 


}

    while (!viewer.wasStopped ())
    {
  }


  return (0);
}

