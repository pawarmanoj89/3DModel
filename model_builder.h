#ifndef MODEL_BUILDER_H
#define MODEL_BUILDER_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#define MAX_FRAME 32

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithPointNormalT;

typedef pcl::PointCloud<PointT>::Ptr PointCloudPtr;
typedef pcl::PointCloud<PointT>::ConstPtr PointCloudConstPtr;

typedef pcl::Normal Normal;
typedef pcl::PointCloud<Normal> PointCloudOfNormals;
typedef pcl::PointCloud<Normal>::Ptr PointCloudOfNormalsPtr;
typedef pcl::FPFHSignature33 FpfhSignature;
typedef pcl::PointCloud<FpfhSignature> PointCloudOfFpfh;
typedef pcl::PointCloud<FpfhSignature>::Ptr PointCloudOfFpfhPtr;

struct PCD
{
  PointCloud::Ptr cloud;
  std::string f_name;

  PCD() : cloud (new PointCloud) {};
};

class ModelBuilder
{
  public:
    ModelBuilder();
    ~ModelBuilder();
    
	protected:
  
};

#endif
