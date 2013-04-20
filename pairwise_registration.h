#ifndef PAIRWISE_REGISTRATION_H
#define PAIRWISE_REGISTRATION_H
#include "model_builder.h"

class PairwiseRegistration
{
  public:
    PairwiseRegistration();
    ~PairwiseRegistration();
    void registerCloud(std::vector<PCD, Eigen::aligned_allocator<PCD> > data, Eigen::Matrix4f *initialTransformationGuess, Eigen::Matrix4f*& finalTransformation);
	void getNormalPoints(PointCloudConstPtr src, PointCloudWithPointNormalT::Ptr &points_with_normals_src);
	void pairAlign(const PointCloudWithPointNormalT::Ptr cloud_src, const PointCloudWithPointNormalT::Ptr cloud_tgt, Eigen::Matrix4f guess, Eigen::Matrix4f &final_transform);
	//void pairAlign(const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt, Eigen::Matrix4f guess, Eigen::Matrix4f &final_transform);
	void showCloudsRight(const PointCloudWithPointNormalT::Ptr cloud_target, const PointCloudWithPointNormalT::Ptr cloud_source);
  protected:
  
};

#endif
