#ifndef INITIAL_GUESS_H
#define INITIAL_GUESS_H

#include "model_builder.h"

class InitialGuess
{
  public:
    InitialGuess ();
    ~InitialGuess ();
    void computeInitialGuess(std::vector<PCD, Eigen::aligned_allocator<PCD> > &filteredData,Eigen::Matrix4f*& initialTransformationGuess);
	void getNormals(PointCloudConstPtr cloud,PointCloudOfNormalsPtr &normals);
	void getSIFTKeypoints(PointCloudConstPtr cloud,pcl::PointCloud<pcl::PointXYZI>::Ptr &keypoints);
	void computeFPFHFeatures(PointCloudPtr cloud,PointCloudOfNormalsPtr normals,pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints,PointCloudOfFpfhPtr &descriptor);
	void findCorrespondance(PointCloudOfFpfhPtr descriptor1, PointCloudOfFpfhPtr descriptor2, pcl::CorrespondencesPtr &corres );
	void getInitialTransformation(pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints1, pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints2, pcl::CorrespondencesPtr corres,pcl::CorrespondencesPtr inlier, Eigen::Matrix4f &initialTransformation);
	void checkMatrices(Eigen::Matrix4f Matrix1[],Eigen::Matrix4f Matrix2[],int size);
	void displayCorrespondances(pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints1, pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints2, pcl::CorrespondencesPtr corres,pcl::CorrespondencesPtr inlier);
  protected:
  
};

#endif
