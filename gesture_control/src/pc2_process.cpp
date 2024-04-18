#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/filters/passthrough.h> 
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/progressive_morphological_filter.h>

#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter_indices.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include <Eigen/Dense>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;



ros::Publisher pub_pc_filtered;
ros::Publisher markerPub;
ros::Publisher midpointPub;


void cloud_cb(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input){
  pcl::PCLPointCloud2 pcl_pc2;
  pcl_conversions::toPCL(*input,pcl_pc2);
  PointCloud::Ptr temp_cloud(new PointCloud);
  pcl::fromPCLPointCloud2(pcl_pc2,*temp_cloud);
  
  PointCloud::Ptr cloud_filtered(new PointCloud);
  // pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointIndicesPtr ground(new pcl::PointIndices);

  /// TODO: old code
  
  
  // //////////////////////////////////

  /// Remove Outliers
  pcl::StatisticalOutlierRemoval<PointT> sor;
  sor.setInputCloud(temp_cloud);
  sor.setMeanK(50);
  sor.setStddevMulThresh(1.0);
  sor.filter(*cloud_filtered);
  //////////////////////////////////

  // for (const auto& point: *cloud_filtered)
  //   std::cout << "    " << point.x
  //             << " "    << point.y
  //             << " "    << point.z << std::endl;

  /// Create filtering object
  pcl::ProgressiveMorphologicalFilter<PointT> pmf;
  pmf.setInputCloud(cloud_filtered);
  pmf.setMaxWindowSize(50);
  pmf.setSlope(5.0f);
  // pmf.setInitialDistance(-10.0f);
  pmf.setMaxDistance(10.0);
  pmf.extract(ground->indices);

  // Create filtering object
  pcl::ExtractIndices<PointT> extract;
  extract.setInputCloud(cloud_filtered);
  extract.setIndices(ground);
  // extract.filter(*ground_cloud_filtered);

  // Extract non-ground returns
  extract.setNegative(true);
  extract.filter(*cloud_filtered);
  //////////////////////////////////

  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (100);
  seg.setDistanceThreshold (0.02);

  int nr_points = (int) cloud_filtered->size ();
  while (cloud_filtered->size () > 0.2 * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (cloud_filtered);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }

    // Extract the planar inliers from the input cloud
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers);
    // Remove the planar inliers, extract the rest
    extract.setNegative (true);
    extract.filter (*cloud_filtered);
  }

  // std::cout << cloud_filtered->size() << std::endl;

  // for (const auto& point: *cloud_filtered)
  // std::cout << "    " << point.x
  //           << " "    << point.y
  //           << " "    << point.z << std::endl;

  /// Filter out ground plane
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud(cloud_filtered);
  pass.setFilterFieldName("x");

  pass.setFilterLimits(-9.0, 9.0);
  pass.filter(*cloud_filtered);

  pass.setInputCloud(cloud_filtered);
  pass.setFilterFieldName("y");

  pass.setFilterLimits(-9.0, 9.0);
  pass.filter(*cloud_filtered);

  // for (const auto& point: *cloud_filtered)
  //   std::cout << "    " << point.x
  //             << " "    << point.y
  //             << " "    << point.z << std::endl;

  // pcl::StatisticalOutlierRemoval<PointT> sor;
//   sor.setInputCloud(cloud_filtered);
//   sor.setMeanK(50);
//   sor.setStddevMulThresh(0.5);
//   sor.filter(*cloud_filtered);

  if(cloud_filtered->size() == 0)
    return;

  std::cout << "made it here" << std::endl;

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud_filtered);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (1.0); // 2cm
  ec.setMinClusterSize (10);
  ec.setMaxClusterSize (100);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_filtered);
  ec.extract (cluster_indices);

  visualization_msgs::MarkerArray centroidMarker;
  std::vector<Eigen::Vector4f> centroids;

  geometry_msgs::PoseArray pa;
  pa.header.frame_id = "base_footprint";

  int j = 0;
  for (const auto& cluster : cluster_indices)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& idx : cluster.indices) {
      cloud_cluster->push_back((*cloud_filtered)[idx]);
    } //*
    cloud_cluster->width = cloud_cluster->size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size () << " data points." << std::endl;

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud_cluster, centroid);
    centroids.push_back(centroid);

    geometry_msgs::Pose p;
    p.position.x = centroid[0];
    p.position.y = centroid[1];
    pa.poses.push_back(p);
    std::cout << "centroid: " << centroid[0] << " " << centroid[1] << " " << centroid[2] << " " << centroid[3] << "\n" << std::endl;
    
    visualization_msgs::Marker m;
    m.id = j;
    m.type = visualization_msgs::Marker::CUBE;
    m.header.frame_id = "base_footprint";
    m.scale.x = 0.3;
    m.scale.y = 0.3;
    m.scale.z = 0.3;
    m.action = visualization_msgs::Marker::ADD;
    m.color.a = 1.0;
    m.color.r = 1;
    m.color.g = 0;
    m.color.b = 0;
    m.pose.position.x = centroid[0];
    m.pose.position.y = centroid[1];
    m.pose.position.z = centroid[2];

    centroidMarker.markers.push_back(m);
    
    j++;
  }

  if(centroids.size() != 2){
    std::cout << "incorrect centroid size: " << centroids.size() << std::endl;
    return;
  }

  Eigen::Vector4f midpoint = (centroids[0] + centroids[1]) / 2;

  visualization_msgs::Marker m;
  m.id = j+1;
  m.type = visualization_msgs::Marker::CUBE;
  m.header.frame_id = "base_footprint";
  m.scale.x = 0.3;
  m.scale.y = 0.3;
  m.scale.z = 0.3;
  m.action = visualization_msgs::Marker::ADD;
  m.color.a = 1.0;
  m.color.r = 0;
  m.color.g = 1;
  m.color.b = 0;
  m.pose.position.x = midpoint[0];
  m.pose.position.y = midpoint[1];
  m.pose.position.z = midpoint[2];

  centroidMarker.markers.push_back(m);

  std::cout << j << " clusters found" << std::endl;
  markerPub.publish(centroidMarker);

  geometry_msgs::Pose p;
  p.position.x = midpoint[0];
  p.position.y = midpoint[1];
  pa.poses.push_back(p);

  std::cout << "publishing midpoint from clusters 0 and 1" << std::endl;
  midpointPub.publish(pa);
  

  sensor_msgs::PointCloud2 msg;
  pcl::toROSMsg(*cloud_filtered, msg);
  pub_pc_filtered.publish(msg);
}


int main(int argc, char **argv) {
  ros::init(argc, argv, "pc2_process");

  ros::NodeHandle nh;

  ros::Subscriber pc2Sub = nh.subscribe("/lidar1/velodyne_points", 1, cloud_cb);

  pub_pc_filtered = nh.advertise<sensor_msgs::PointCloud2>("pc_filtered", 1);
  markerPub = nh.advertise<visualization_msgs::MarkerArray>("centroid", 1);
  midpointPub = nh.advertise<geometry_msgs::PoseArray>("midpoint", 1);

  ros::spin();
}
