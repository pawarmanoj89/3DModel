
#include <boost/make_shared.hpp>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>


#include <pcl/registration/transforms.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/correspondence.h>

#include <pcl/keypoints/sift_keypoint.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/passthrough.h>
//#include "pcl/kdtree/kdtree_flann.h" 
#include "pcl/filters/passthrough.h" 
#include "pcl/filters/voxel_grid.h" 
#include "pcl/features/fpfh.h" 

#include <pcl/features/fpfh_omp.h>
#include <time.h>
#include <windows.h>
// cpp_compiler_options_openmp.cpp
#include <omp.h>
#include "initial_guess.h"

const double FILTER_LIMIT = 1000.0; 
const int MAX_SACIA_ITERATIONS = 500; 

//units are meters: 

const double NORMALS_RADIUS = 0.04; 
const double FEATURES_RADIUS = 0.04; 
const double SAC_MAX_CORRESPONDENCE_DIST = 0.001; 

const float MIN_SCALE = 0.0005; 
const int NR_OCTAVES = 4; 
const int NR_SCALES_PER_OCTAVE = 5; 
const float MIN_CONTRAST = 1; 
const float FPFH_REDIUS_SEARCH= 0.3;

using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;

//convenient typedefs
// moved to the openni_capture.h

// This is a tutorial so we can afford having global variables 
	//our visualizer

	int vp2_1, vp2_2, vp2_3, vp2_4;
	float dwStart,serialTime,ParallelTime;
	 Eigen::Matrix4f GlobalTransformSerial[20],GlobalTransformSerial2[20];
	 Eigen::Matrix4f GlobalTransformParallel[20],GlobalTransformParallel2[20];
//convenient structure to handle our pointclouds
//struct PCD
//{
//  PointCloud::Ptr cloud;
//  std::string f_name;
//
//  PCD() : cloud (new PointCloud) {};
//};
	  
//struct TRANSFORM_GUESS
//{
//	Eigen::Matrix4f guess;
//	
//	TRANSFORM_GUESS() : guess (new Eigen::Matrix4f::Identity() ) {};
//};
	 
struct PCD_CORS
{
	pcl::CorrespondencesPtr corres;
	
	PCD_CORS() : corres (new pcl::Correspondences) {};
};


struct PCD_FPFH
{
	PointCloudOfFpfhPtr descriptor;
	
	PCD_FPFH() : descriptor (new PointCloudOfFpfh) {};
};
struct PCD_SIFT
{
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
	
	PCD_SIFT() : cloud (new pcl::PointCloud<pcl::PointXYZI>) {};
};

struct PCD_NORMAL
{
	PointCloudOfNormalsPtr coludNormals;
	
	PCD_NORMAL() : coludNormals (new PointCloudOfNormals) {};
};
struct PCD_OUTPUT
{
  PointCloud::Ptr cloud;
  std::string f_name;

  PCD_OUTPUT() : cloud (new PointCloud) {};
};

std::vector<PCD_OUTPUT , Eigen::aligned_allocator<PCD_OUTPUT> > outclouds,outcloudsparallel,outclouds2,initoutclouds,initoutclouds2;


//struct PCDComparator
//{
//  bool operator () (const PCD& p1, const PCD& p2)
//  {
//    return (p1.f_name < p2.f_name);
//  }
//};


// Define a new point representation for < x, y, z, curvature >
class MyPointRepresentation : public pcl::PointRepresentation <PointNormalT>
{
  using pcl::PointRepresentation<PointNormalT>::nr_dimensions_;
public:
  MyPointRepresentation ()
  {
    // Define the number of dimensions
    nr_dimensions_ = 4;
  }

  // Override the copyToFloatArray method to define our feature vector
  virtual void copyToFloatArray (const PointNormalT &p, float * out) const
  {
    // < x, y, z, curvature >
    out[0] = p.x;
    out[1] = p.y;
    out[2] = p.z;
    out[3] = p.curvature;
  }
};

InitialGuess::InitialGuess(){}

InitialGuess::~InitialGuess(){}

void 
	InitialGuess::getNormals(PointCloudConstPtr cloudIn,PointCloudOfNormalsPtr &normalsPtr){
	
		//pcl::PointCloud<pcl::Normal>::Ptr normalsPtr (new pcl::PointCloud<pcl::Normal>); 
		pcl::NormalEstimationOMP<PointT, pcl::Normal> norm_est;
		pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
		norm_est.setSearchMethod(tree);
		
        norm_est.setInputCloud( cloudIn ); 
        norm_est.setRadiusSearch( NORMALS_RADIUS ); 
        norm_est.compute( *normalsPtr ); 
		//cerr<<"\nNormal size "<<normalsPtr->points.size();
        //return normalsPtr; 

}
void
	InitialGuess::getSIFTKeypoints(PointCloudConstPtr cloudIn,pcl::PointCloud<pcl::PointXYZI>::Ptr &keypoints){

	pcl::SIFTKeypoint<PointT ,pcl::PointXYZI> siftdetect;
	
	pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> () );
	siftdetect.setInputCloud(cloudIn);
	siftdetect.setSearchSurface(cloudIn);
	siftdetect.setSearchMethod(tree);
	siftdetect.setMinimumContrast(MIN_CONTRAST);
	siftdetect.setScales(MIN_SCALE,NR_OCTAVES,NR_SCALES_PER_OCTAVE);
	siftdetect.compute(*keypoints);
		
}

void
	InitialGuess::computeFPFHFeatures(PointCloudPtr cloudIn,PointCloudOfNormalsPtr normals,pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints,PointCloudOfFpfhPtr &descriptor){

	pcl::FPFHEstimationOMP<PointT, Normal,FpfhSignature > fpfh_est;
	
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::copyPointCloud (*keypoints,*keypoints_xyzrgb);
	
	pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> () );
	fpfh_est.setSearchMethod(tree);
	fpfh_est.setRadiusSearch (FPFH_REDIUS_SEARCH);
	fpfh_est.setSearchSurface (cloudIn);
	fpfh_est.setInputNormals (normals);
	fpfh_est.setInputCloud (keypoints_xyzrgb);
	
	fpfh_est.compute(*descriptor);
	//cerr<<" Descrip size "<<descriptor->points.size();
}

void 
	InitialGuess::findCorrespondance(PointCloudOfFpfhPtr descriptor1, PointCloudOfFpfhPtr descriptor2, pcl::CorrespondencesPtr &corres ){

	 pcl::registration::CorrespondenceEstimation< pcl::FPFHSignature33, pcl::FPFHSignature33  > cer;
	 
	 //pcl::registration::CorrespondenceEstimation <pcl::PointXYZRGB,pcl::PointXYZRGB > cer2;
	 //pcl::Correspondences corres,recpCorres;
	 
	 cer.setInputCloud(descriptor1);
	 cer.setInputTarget(descriptor2);
	 
	 cer.determineReciprocalCorrespondences(*corres);
	 //cerr<<"\nFPFH Corres size "<<corres.size();
	 
}

void 
	InitialGuess::getInitialTransformation(pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints1, pcl::PointCloud<pcl::PointXYZI>::Ptr keypoint2, pcl::CorrespondencesPtr corres, Eigen::Matrix4f &initialTransformation){

		//boost::shared_ptr<pcl::Correspondences > corres2Ptr (new pcl::Correspondences(corres2));
		boost::shared_ptr<pcl::Correspondences > newCorres2 (new pcl::Correspondences()) ;
		//pcl::CorrespondencesPtr  newCorres=new pcl::CorrCorrespondencesPtr();
		
		 pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZI> crsc;
		 crsc.setInputCloud(keypoints1);
		 crsc.setTargetCloud(keypoint2);
		 crsc.setInputCorrespondences(corres);
		 
		 crsc.setMaxIterations(200);
		 crsc.getCorrespondences(*newCorres2);
		 
		 initialTransformation=crsc.getBestTransformation();	

}

void 
	InitialGuess::computeInitialGuess(std::vector<PCD, Eigen::aligned_allocator<PCD> > &filteredData)
{
	std::vector<PCD_NORMAL, Eigen::aligned_allocator<PCD_NORMAL> > normalData(filteredData.size() );
	std::vector<PCD_SIFT, Eigen::aligned_allocator<PCD_SIFT> > siftData(filteredData.size() );
	std::vector<PCD_FPFH, Eigen::aligned_allocator<PCD_FPFH> > fpfhData(filteredData.size() );
	std::vector<PCD_CORS, Eigen::aligned_allocator<PCD_CORS> > corsData(filteredData.size() -1 );
	Eigen::Matrix4f transformData[MAX_FRAME],transformData2[MAX_FRAME],transformData3[MAX_FRAME];
	DWORD ser,par,mix,tic;
	tic=GetTickCount64();
	cerr<<"\ntic1 "<<tic<<"\n";
//#pragma omp parallel for
	 for(int i=0;i<filteredData.size();i++)
	 {
		 getNormals(filteredData[i].cloud,normalData[i].coludNormals);
		 getSIFTKeypoints(filteredData[i].cloud, siftData[i].cloud);
		 computeFPFHFeatures(filteredData[i].cloud,normalData[i].coludNormals,siftData[i].cloud,fpfhData[i].descriptor);
		 cerr<<"*";
	 }
	
//#pragma omp parallel for
	 for(int i=0;i<corsData.size();i++)
	 {
		 findCorrespondance(fpfhData[i].descriptor,fpfhData[i+1].descriptor,corsData[i].corres);
		 getInitialTransformation(siftData[i].cloud,siftData[i+1].cloud,corsData[i].corres,transformData[i]);
		 //cerr<<"\n"<<transformData[i];
		 cerr<<"!";
	 }
	 ser=GetTickCount64()-tic;
	 cerr<<"\nSerial Finished###\nTime Taken "<<ser;
	 tic=GetTickCount64();
	 cerr<<"\ntic2 "<<tic<<"\n";
#pragma omp parallel for
	 for(int i=0;i<filteredData.size();i++)
	 {
		 getNormals(filteredData[i].cloud,normalData[i].coludNormals);
		 getSIFTKeypoints(filteredData[i].cloud, siftData[i].cloud);
		 computeFPFHFeatures(filteredData[i].cloud,normalData[i].coludNormals,siftData[i].cloud,fpfhData[i].descriptor);
		 cerr<<"*";
	 }
	
#pragma omp parallel for
	 for(int i=0;i<corsData.size();i++)
	 {
		 findCorrespondance(fpfhData[i].descriptor,fpfhData[i+1].descriptor,corsData[i].corres);
		 getInitialTransformation(siftData[i].cloud,siftData[i+1].cloud,corsData[i].corres,transformData2[i]);
		 //cerr<<"\n"<<transformData[i];
		 cerr<<"!";
	 }
	 par=GetTickCount64()-tic;
	 cerr<<"\nParallel Finished###\nTime Taken "<<par;

	 tic=GetTickCount64();
	 cerr<<"\ntic1 "<<tic<<"\n";
	  for(int i=0;i<filteredData.size();i++)
	 {
		 getNormals(filteredData[i].cloud,normalData[i].coludNormals);
	}
	  
	 #pragma omp parallel for
	 for(int i=0;i<filteredData.size();i++)
	 {
		getSIFTKeypoints(filteredData[i].cloud, siftData[i].cloud);
	 }

	   for(int i=0;i<filteredData.size();i++)
	 {
		 computeFPFHFeatures(filteredData[i].cloud,normalData[i].coludNormals,siftData[i].cloud,fpfhData[i].descriptor);
		 cerr<<"*";
	 }

#pragma omp parallel for
	 for(int i=0;i<corsData.size();i++)
	 {
		 findCorrespondance(fpfhData[i].descriptor,fpfhData[i+1].descriptor,corsData[i].corres);
		 getInitialTransformation(siftData[i].cloud,siftData[i+1].cloud,corsData[i].corres,transformData3[i]);
		 //cerr<<"\n"<<transformData[i];
		 cerr<<"!";
	 }
	 mix=GetTickCount64()-tic;
	 cerr<<"\nSerial Parallel Finished###\nTime Taken "<<mix;
	 cerr<<"\nComparison\n"<<ser<<"\n"<<par<<"\n"<<mix;
	 cerr<<"\nChecking Matr..";
	 checkMatrices(transformData,transformData2,filteredData.size()-1);
	 checkMatrices(transformData,transformData3,filteredData.size()-1);

}




 void InitialGuess::checkMatrices(Eigen::Matrix4f Matrix1[],Eigen::Matrix4f Matrix2[],int size)
 {
	 for(int i=0;i<size;i++)
	 {
		 	if (!(Matrix1[i].operator==( Matrix2[i])) )
				cerr<<"\nMissmatch Matrices"<<i;
			cerr<<"\nchecked"<<i;				
	 }
	cerr<<"\nDone";
 }







