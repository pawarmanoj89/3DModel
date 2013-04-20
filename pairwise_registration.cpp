#include <boost/make_shared.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/ia_ransac.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <time.h>
#include<conio.h>

#include "initial_guess.h";

// cpp_compiler_options_openmp.cpp
#include <omp.h>

#include "pairwise_registration.h"


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


const double SAC_MAX_CORRESPONDENCE_DIST = 0.001; 

struct PCD_POINT_NORMAL
{
	PointCloudWithPointNormalT::Ptr pointNormals;
	
	PCD_POINT_NORMAL() : pointNormals (new PointCloudWithPointNormalT) {};
};

pcl::visualization::PCLVisualizer *p2;
int vp2_1, vp2_2, vp2_3, vp2_4;

PairwiseRegistration::PairwiseRegistration(){}

PairwiseRegistration::~PairwiseRegistration(){}



////////////////////////////////////////////////////////////////////////////////
/** \brief Display source and target on the second viewport of the visualizer
 *
 */
void PairwiseRegistration::showCloudsRight(const PointCloudWithPointNormalT::Ptr cloud_target, const PointCloudWithPointNormalT::Ptr cloud_source)
{
  p2->removePointCloud ("source");
  p2->removePointCloud ("target");


  pcl::visualization::PointCloudColorHandlerGenericField<PointNormalT> tgt_color_handler (cloud_target, "curvature");
  if (!tgt_color_handler.isCapable ())
      PCL_WARN ("Cannot create curvature color handler!");

  pcl::visualization::PointCloudColorHandlerGenericField<PointNormalT> src_color_handler (cloud_source, "curvature");
  if (!src_color_handler.isCapable ())
      PCL_WARN ("Cannot create curvature color handler!");


  p2->addPointCloud (cloud_target, tgt_color_handler, "target", vp2_2);
  p2->addPointCloud (cloud_source, src_color_handler, "source", vp2_2);

  p2->spinOnce();
}





 ////////////////////////////////////////////////////////////////////////////////
/** \brief ICPOMP call this function, it just return transformation not modifing cloud
  * \param cloud_src source cloud
  * \param cloud_tgt target cloud 
  * \param final_transform matrix
  */
 void
	 PairwiseRegistration::getNormalPoints(PointCloudConstPtr src, PointCloudWithPointNormalT::Ptr &points_with_normals_src)
 {
	 
      pcl::NormalEstimation<PointT, PointNormalT> norm_est;
	  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
	  norm_est.setSearchMethod (tree);
	  norm_est.setKSearch (30);
  
	  norm_est.setInputCloud (src);
	  norm_est.compute (*points_with_normals_src);
	  pcl::copyPointCloud (*src, *points_with_normals_src);
	 
 }
 
 
	void
		PairwiseRegistration::pairAlign(const PointCloudWithPointNormalT::Ptr cloud_src, const PointCloudWithPointNormalT::Ptr cloud_tgt,Eigen::Matrix4f guess, Eigen::Matrix4f &final_transform)
		//(const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt, Eigen::Matrix4f guess, Eigen::Matrix4f &final_transform)
	{
		
	PointCloudWithPointNormalT::Ptr points_with_normals_src (new PointCloudWithPointNormalT);
    PointCloudWithPointNormalT::Ptr points_with_normals_tgt (new PointCloudWithPointNormalT);
  
	points_with_normals_src=cloud_src;
    points_with_normals_tgt=cloud_tgt;

  // Instantiate our custom point representation (defined above) ...
  MyPointRepresentation point_representation;
  // ... and weight the 'curvature' dimension so that it is balanced against x, y, and z
  float alpha[4] = {1.0, 1.0, 1.0, 1.0};
  point_representation.setRescaleValues (alpha);
  //pcl::IterativeClosestPoint<PointNormalT, PointNormalT> reg;
  //
  // Align
  pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;
  reg.setTransformationEpsilon (1e-8);
  // Set the maximum distance between two correspondences (src<->tgt) to 10cm
  
  reg.setMaxCorrespondenceDistance (0.5);  
  // Set the point representation
  reg.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation));
  reg.setRANSACOutlierRejectionThreshold( 0.5 ); 
  reg.setInputCloud (points_with_normals_src);
  reg.setInputTarget (points_with_normals_tgt);



  //
  // Run the same optimization in a loop and visualize the results
  Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), prev, targetToSource;
  PointCloudWithPointNormalT::Ptr reg_result = points_with_normals_src;
  reg.setMaximumIterations (2);
  for (int i = 0; i < 30; ++i)
  {
    PCL_INFO ("Iteration Nr. %d.\n", i);

    // save cloud for visualization purpose
    points_with_normals_src = reg_result;

    // Estimate
    reg.setInputCloud (points_with_normals_src);
	cerr<<"\nguess not using";
    reg.align (*reg_result);

		//accumulate transformation between each Iteration
    Ti = reg.getFinalTransformation () * Ti;

		//if the difference between this transformation and the previous one
		//is smaller than the threshold, refine the process by reducing
		//the maximal correspondence distance
    if (fabs ((reg.getLastIncrementalTransformation () - prev).sum ()) < reg.getTransformationEpsilon ())
      reg.setMaxCorrespondenceDistance (reg.getMaxCorrespondenceDistance () - 0.001);
    
    prev = reg.getLastIncrementalTransformation ();

    // visualize current state
    //showCloudsRight(points_with_normals_tgt, points_with_normals_src);
  } 
  final_transform = Ti.inverse();
  cerr<<cloud_src<<"  "<<cloud_tgt<<" Coverage "<<reg.hasConverged()<< " score: " << reg.getFitnessScore() << std::endl;
 // cerr<<"\n"<<final_transform;
}


 void 
	  PairwiseRegistration::registerCloud(std::vector<PCD, Eigen::aligned_allocator<PCD> > filteredData, Eigen::Matrix4f *initialTransformationGuess, Eigen::Matrix4f*& finalTransformation)
  {
	  std::vector<PCD_POINT_NORMAL, Eigen::aligned_allocator<PCD_POINT_NORMAL> > pointNormalData(filteredData.size() );
	   p2 = new pcl::visualization::PCLVisualizer("3DViewer");
		  p2->createViewPort (0.0, 0, 0.33, 1.0, vp2_1);
		  p2->createViewPort (0.33, 0, 0.66, 1.0, vp2_2);
		  p2->createViewPort (0.66, 0, 1.0, 1.0, vp2_3);
		  

  #pragma omp parallel for 
	  for (int i = 0; i < filteredData.size (); ++i)
		  {
			  getNormalPoints(filteredData[i].cloud,pointNormalData[i].pointNormals);
		  }

	  finalTransformation[0]=Eigen::Matrix4f::Identity();
  #pragma omp parallel for  
	  for (int i = 1; i < filteredData.size (); ++i)
		  {
			pairAlign (pointNormalData[i-1].pointNormals, pointNormalData[i].pointNormals,  initialTransformationGuess[i], finalTransformation[i]);
			  //pairAlign (filteredData[i].cloud, filteredData[i+1].cloud,  initialTransformationGuess[i], finalTransformation[i]);
    	  }
			
			std::cout<<"Alignment Done";

	  #pragma omp parallel 
	{
		#pragma omp for ordered
			for (int i = 1; i < filteredData.size ()-1; ++i)
			{
				 #pragma omp ordered
				{
					
				finalTransformation[i]=finalTransformation[i] * finalTransformation[i-1] ;
							
				}
			}
	}

	
  
  }

