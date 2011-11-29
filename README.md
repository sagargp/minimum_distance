Ankur Agarwal
Farrukh Ali
Sagar Pandya

### USC CSCI 547 Sensing and Planning in Robotics project, F2011

Given a point cloud of Duplo blocks, find the "minimum distance" between any given pair. (The minimum distance is the shortest distance between the blocks, not the distance bewteen their center points). The input point cloud comes from an XBox Kinect sensor mounted on top of a PR2. We assume that no blocks of the same color are touching.

Overview of algorithm:
* Read point cloud
* Apply voxel grid filter to downsample the cloud
* Apply planar segmentation to remove the table
* Color segment the points in the HSV color space
* Cluster the colored clouds
* Calculate a distance after getting user input

Color segmentation was done by setting simple min/max range values for each expected color. This worked very well for our case. We attempted to make it more robust by gathering mean and variance HSV values for each color, and then computing the probability that a point was a given color, but this produced worse results. 

Distance calculation between a pair of clouds was done by creating a KD tree for one cloud, and then iterating over every point in the other cloud to find the closest one. This improved the runtime from the naive O(n^2) to roughly O(nlogn).

Finally our code had to be modified slightly to become a ROS node for the PR2. The perception_pcl stack was used to publish a PCD file as a point cloud sensor message. Our node subscribed to that data, conveted it to a PCL point cloud object, and then operated on it as described above. Visualization markers were used to identify cloud IDs in RVIZ.

To run the ROS node, first you have to build it:

> $ rosmake min_dist_ros

Then you'll need four terminals:

Terminal 1:
> $ roscore

Terminal 2:
> $ rosrun pcl_ros pcd_to_pointcloud /path/to/a/pcd/file 0.1

Terminal 3:
> $ rosrun rviz rviz

Terminal 4:
> $ rosrun min_dist_ros min_dist_ros input:=cloud_pcd
