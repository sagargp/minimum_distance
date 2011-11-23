#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions.hpp>
#include <cmath>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>

#define MIN(x, y) ( (x) <= (y) ) ? (x) : (y)
#define MAX(x, y) ( (x) >= (y) ) ? (x) : (y)

#define NRT_D_2_PI 6.28318531

pcl::PointXYZRGB o1;
pcl::PointXYZRGB o2;
std::vector<pcl::PointXYZRGB> labels;

double vonMisesPDF(double x, double mu, double k)
{
	return 1.0 - 1.0/(NRT_D_2_PI) + exp(k * cos(x-mu)) / exp(NRT_D_2_PI * boost::math::cyl_bessel_i(0,k));
}

void  drawArrow(pcl::visualization::PCLVisualizer& viewer)
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

void label (pcl::visualization::PCLVisualizer& viewer)
{
	for(size_t i = 0; i < labels.size(); i++)
	{
		labels[i].z = labels[i].z; // - .05;
		char f[5];
		sprintf(f,"%d",i);
		viewer.addText3D (f, labels[i], 0.01, 1, 1, 1, f, 0);
	}
}

void toHSV(int ri, int gi, int bi, float *h, float *s, float *v)
{
	float r = ri/255.0;
	float g = gi/255.0;
	float b = bi/255.0;

	float M = MAX( MAX(r, g), b);
	float m = MIN( MIN(r, g), b);

	float C = M - m;

	if (C == 0)
		*h = 0;
	else if (M == r)
		*h = ((g-b)/C);
	else if (M == g)
		*h = (b-r)/C + 2;
	else if (M == g)
		*h = (r-g)/C + 4;

	*h *= 60.0; 
	*v = M;

	if (C == 0)
		*s = 0;
	else
		*s = C/(*v);

	if (*h < 0)
		*h += 360.0;
}

void normalSegment(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input, 
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr RCloud, 
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr GCloud, 
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr BCloud, 
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr YCloud, 
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr OCloud)
{
	// pseudocode
	//t = table of mean/variance of each color
	//for Point p in input:
	//	norm_g = normal_distribution(t[redmean], t[redstd]);
	//	prob_g = norm_r.cdf(p+0.5) - norm_r.cdf(p-0.5);
	//	
	//	norm_b = normal_distribution(t[redmean], t[redstd]);
	//	prob_b = norm_r.cdf(p+0.5) - norm_r.cdf(p-0.5);
	//	
	//	norm_y = normal_distribution(t[redmean], t[redstd]);
	//	prob_y = norm_r.cdf(p+0.5) - norm_r.cdf(p-0.5);
	//	
	//	norm_o = normal_distribution(t[redmean], t[redstd]);
	//	prob_o = norm_r.cdf(p+0.5) - norm_r.cdf(p-0.5);

	//	color = highest of prob_x

	//color		h					s					(mean, std)
	//red		
	//green		116.964, 36.432		0.8780, 0.1616
	//blue		229.644, 5.292		0.8527, 0.1730
	//yellow	037.188, 6.948		0.9835, 0.0347
	//orange	012.361, 7.1757		0.9822, 0.0813	

	double gh[] = {116.964, 36.432};
	double bh[] = {229.644, 5.2920};
	double yh[] = {037.188, 6.9480};
	double oh[] = {012.361, 7.1757};

	double gs[] = {0.8780, 0.1616};
	double bs[] = {0.8527, 0.1730};
	double ys[] = {0.9835, 0.0347};
	double os[] = {0.9822, 0.0813};

	for (int i = 0; i < input->points.size(); i++)
	{
		float h, s, v;
		toHSV(input->points[i].r, input->points[i].g, input->points[i].b, &h, &s, &v);

		//printf("%d: (%d, %d, %d) -> (%f, %f, %f)\n", i, input->points[i].r, input->points[i].g, input->points[i].b, h, s, v);
		
		if (h != h) // check if h is NaN
		{
			cout << "skipping ..." << endl;
			continue;
		}

		/* calculate the probability that this point is green */
		boost::math::normal_distribution<double> norm_gh(gh[0], gh[1]);
		double prob_gh = boost::math::cdf(norm_gh, h);

		boost::math::normal_distribution<double> norm_gs(gs[0], gs[1]);
		double prob_gs = boost::math::cdf(norm_gs, s);

		double prob_g = prob_gh * prob_gs;

		/* calculate the probability that this point is blue */
		boost::math::normal_distribution<double> norm_bh(bh[0], bh[1]);
		double prob_bh = boost::math::cdf(norm_bh, h);

		boost::math::normal_distribution<double> norm_bs(bs[0], bs[1]);
		double prob_bs = boost::math::cdf(norm_bs, s);

		double prob_b = prob_bh * prob_bs;

		/* calculate the probability that this point is yellow */
		boost::math::normal_distribution<double> norm_yh(yh[0], yh[1]);
		double prob_yh = boost::math::cdf(norm_yh, h);

		boost::math::normal_distribution<double> norm_ys(ys[0], ys[1]);
		double prob_ys = boost::math::cdf(norm_ys, s);

		double prob_y = prob_yh * prob_ys;

		/* calculate the probability that this point is yellow */
		boost::math::normal_distribution<double> norm_oh(oh[0], oh[1]);
		double prob_oh = boost::math::cdf(norm_oh, h);

		boost::math::normal_distribution<double> norm_os(os[0], os[1]);
		double prob_os = boost::math::cdf(norm_os, s);

		double prob_o = prob_oh * prob_os;

		// now assign to buckets
		if (prob_g > prob_b && prob_g > prob_y && prob_g > prob_o)
			GCloud->push_back(input->points[i]);

		else if (prob_b > prob_g && prob_b > prob_y && prob_b > prob_o)
			BCloud->push_back(input->points[i]);

		else if (prob_y > prob_g && prob_y > prob_b && prob_y > prob_o)
			YCloud->push_back(input->points[i]);

		else if (prob_o > prob_g && prob_o > prob_b && prob_o > prob_y)
			OCloud->push_back(input->points[i]);
	}
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

int clusterExtraction(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input, std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> *clouds, std::vector<pcl::PointXYZRGB> *labels)
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
	reader.read(argv[1], *cloudF);

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

	vg.setInputCloud (RGBcloud);
	vg.setLeafSize (0.001f, 0.001f, 0.001f);
	vg.filter (*cloud_filtered);

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
	while (cloud_filtered->points.size () > 0.30* nr_points)
	{
		seg.setInputCloud(cloud_filtered);
		seg.segment (*inliers, *coefficients); //*

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
		//viewer.showCloud(cloud_plane, "cloud_name");
		//std::cin.get();
		// Remove the planar inliers, extract the rest
		extract.setNegative (true);
		extract.filter (*cloud_filtered); //*
		std::cerr <<" The Coefficients are: " << coefficients->values[0]<< " "<< coefficients->values[1]<< " "<< coefficients->values[2]<< " " << coefficients->values[3]<< " "<< std::endl;
	}

	// color segmentation
	std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> coloredClouds(5);
	for (int i = 0; i < 5; i++)
		coloredClouds[i] = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

	// just give the colored clouds handy names
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr Rcloud = coloredClouds[0];
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr Gcloud = coloredClouds[1];
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr Bcloud = coloredClouds[2];
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr Ycloud = coloredClouds[3];
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr Ocloud = coloredClouds[4];

	// color segment each cluster found from the filtered cloud
	//colorSegment(cloud_filtered, Rcloud, Gcloud, Bcloud, Ycloud, Ocloud);
	normalSegment(cloud_filtered, Rcloud, Gcloud, Bcloud, Ycloud, Ocloud);

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
		num_clusters += clusterExtraction(coloredClouds[i], &clouds, &labels);
	}
	// display what we've found	
	std::cerr<<"Waiting 3 "<<std::endl;
	pcl::visualization::CloudViewer viewer("Cloud Viewer");

	viewer.runOnVisualizationThreadOnce(label);

	for (int i = 0; i < coloredClouds.size(); i++)
	{
		viewer.showCloud(coloredClouds[i]);
		cin.get();
	}

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

		viewer.runOnVisualizationThreadOnce(drawArrow);

		cout << "Enter the number of the cloud you want to see alone: ";
		cin >> xx;
	}

	while (!viewer.wasStopped ());

	return 0;
}
