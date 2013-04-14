#ifndef INITIAL_GUESS_H
#define INITIAL_GUESS_H

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

class InitialGuess
{
  public:
    InitialGuess ();
    ~InitialGuess ();
    void computeInitialGuess(std::vector<PCD, Eigen::aligned_allocator<PCD> > &filteredData);
	void getNormals(PointCloudConstPtr cloud,PointCloudOfNormalsPtr &normals);
	void getSIFTKeypoints(PointCloudConstPtr cloud,pcl::PointCloud<pcl::PointXYZI>::Ptr &keypoints);
	void computeFPFHFeatures(PointCloudPtr cloud,PointCloudOfNormalsPtr normals,pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints,PointCloudOfFpfhPtr &descriptor);
	void findCorrespondance(PointCloudOfFpfhPtr descriptor1, PointCloudOfFpfhPtr descriptor2, pcl::CorrespondencesPtr &corres );
	void getInitialTransformation(pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints1, pcl::PointCloud<pcl::PointXYZI>::Ptr keypoint2, pcl::CorrespondencesPtr corres, Eigen::Matrix4f &initialTransformation);
	void checkMatrices(Eigen::Matrix4f Matrix1[],Eigen::Matrix4f Matrix2[],int size);
  protected:
  
};

#endif
