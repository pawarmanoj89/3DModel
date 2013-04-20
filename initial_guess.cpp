
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

pcl::visualization::PCLVisualizer *p;
int vp_1, vp_2, vp_3, vp_4;

using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;

//convenient typedefs
// moved to the openni_capture.h

// This is a tutorial so we can afford having global variables 
	//our visualizer

	
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
	 cer.determineCorrespondences(*corres);
	 //cer.determineReciprocalCorrespondences(*corres);
	 cerr<<"\nFPFH Corres size "<<corres->size();
	 
}

void 
	InitialGuess::getInitialTransformation(pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints1, pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints2, pcl::CorrespondencesPtr corres,pcl::CorrespondencesPtr inlier, Eigen::Matrix4f &initialTransformation){
		
		//boost::shared_ptr<pcl::Correspondences > corres2Ptr (new pcl::Correspondences(corres2));
		//boost::shared_ptr<pcl::Correspondences > newCorres2 (new pcl::Correspondences()) ;
		//pcl::CorrespondencesPtr  newCorres=new pcl::CorrCorrespondencesPtr();
		
		 pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZI> crsc;
		 crsc.setInputCloud(keypoints1);
		 crsc.setTargetCloud(keypoints2);
		 crsc.setInputCorrespondences(corres);
		 
		 crsc.setMaxIterations(200);
		 crsc.getCorrespondences(*inlier);
		 Eigen::Matrix4f trans=crsc.getBestTransformation();	
		 initialTransformation=trans;//.inverse();
		/*cerr<<"\n";
		cerr<<initialTransformation;*/

}

void 
	InitialGuess::computeInitialGuess(std::vector<PCD, Eigen::aligned_allocator<PCD> > &filteredData,Eigen::Matrix4f*& initialTransformationGuess)
{
	std::vector<PCD_NORMAL, Eigen::aligned_allocator<PCD_NORMAL> > normalData(filteredData.size() );
	std::vector<PCD_SIFT, Eigen::aligned_allocator<PCD_SIFT> > siftData(filteredData.size() );
	std::vector<PCD_FPFH, Eigen::aligned_allocator<PCD_FPFH> > fpfhData(filteredData.size() );
	std::vector<PCD_CORS, Eigen::aligned_allocator<PCD_CORS> > corsData(filteredData.size()  );
	std::vector<PCD_CORS, Eigen::aligned_allocator<PCD_CORS> > inlierData(filteredData.size()  );
	//Eigen::Matrix4f *transformData=new Eigen::Matrix4f[MAX_FRAME];  //,transformData2[MAX_FRAME],transformData3[MAX_FRAME];
	
	DWORD ser,par,mix,tic;
	tic=GetTickCount64();
	cerr<<"\ntic1 "<<tic<<"\n";


	#pragma omp parallel for
	 for(int i=0;i<filteredData.size();i++)
	 {
		 getNormals(filteredData[i].cloud,normalData[i].coludNormals);
		 getSIFTKeypoints(filteredData[i].cloud, siftData[i].cloud);
		 computeFPFHFeatures(filteredData[i].cloud,normalData[i].coludNormals,siftData[i].cloud,fpfhData[i].descriptor);
		 cerr<<"*";
	 }
	
	 initialTransformationGuess[0]=Eigen::Matrix4f::Identity();
#pragma omp parallel for
	 for(int i=1;i<corsData.size();i++)
	 {
		 findCorrespondance(fpfhData[i-1].descriptor,fpfhData[i].descriptor,corsData[i].corres);
		 getInitialTransformation(siftData[i-1].cloud,siftData[i].cloud,corsData[i].corres,inlierData[i].corres,initialTransformationGuess[i]);
		 //cerr<<"\n"<<transformData[i];
		 cerr<<"!";
	 }
	 par=GetTickCount64()-tic;
	 cerr<<"\nParallel Finished###\nTime Taken "<<par;

	 	  for(int i=0;i<filteredData.size()-1;i++)
		  {
			  cerr<<initialTransformationGuess[i];
			  cerr<<"\n\n";
		  }
	 p = new pcl::visualization::PCLVisualizer("3DViewer");
		  p->createViewPort (0.0, 0, 0.33, 1.0, vp_1);
		  p->createViewPort (0.33, 0, 0.66, 1.0, vp_2);
		  p->createViewPort (0.66, 0, 1.0, 1.0, vp_3);

		/*  for(int i=0;i<filteredData.size()-1;i++)
		  {		
			  std::stringstream ss1 ("cloud_vp_1_");
			  p->addPointCloud(filteredData[i].cloud,ss1.str(),vp_1);
			  ss1<<i;
			  p->addPointCloud(filteredData[i+1].cloud,ss1.str(),vp_1);
			  displayCorrespondances(siftData[i].cloud,siftData[i+1].cloud,corsData[i].corres,inlierData[i].corres);
			  p->removeAllPointClouds();
			  p->removeAllShapes();
		  }*/
}

void InitialGuess::displayCorrespondances(pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints1, pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints2, pcl::CorrespondencesPtr corres,pcl::CorrespondencesPtr inlier)
{		
	
	for(int i=1;i<corres->size();i++)
	{
		pcl::PointXYZI srcpt=keypoints1->points.at(corres->at(i).index_query);              //corres[i].index_query);
		 pcl::PointXYZI tgtpt=keypoints2->points.at(corres->at(i).index_match);
		 std::stringstream ss2 ("line_vp_2_");
		 ss2<<i;
		 p->addLine<pcl::PointXYZI>(srcpt,tgtpt,ss2.str(),vp_2);
	}

	for(int i=1;i<inlier->size();i++)
	{
		 pcl::PointXYZI srcpt=keypoints1->points.at(inlier->at(i).index_query);
		 pcl::PointXYZI tgtpt=keypoints2->points.at(inlier->at(i).index_match);
		 std::stringstream ss3 ("line_vp_3_");
		 ss3<<i;
		 p->addLine<pcl::PointXYZI>(srcpt,tgtpt,ss3.str(),vp_3);
	}
	p->spin();
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







