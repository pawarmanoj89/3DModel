
#include <boost/make_shared.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>

#include <pcl/io/pcd_io.h>


#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/correspondence.h>

#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/histogram_visualizer.h>

#include <time.h>
#include<conio.h>


#include <pcl/filters/passthrough.h>
//#include "pcl/kdtree/kdtree_flann.h" 
#include "pcl/filters/passthrough.h" 
#include "pcl/filters/voxel_grid.h" 
#include "pcl/features/fpfh.h" 

#include <pcl/features/fpfh_omp.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/mls_omp.h>

// cpp_compiler_options_openmp.cpp
#include <omp.h>

const double FILTER_LIMIT = 1000.0; 
const int MAX_SACIA_ITERATIONS = 500; 

//units are meters: 
const float VOXEL_GRID_SIZE = 0.03; 
const double NORMALS_RADIUS = 0.04; 
const double FEATURES_RADIUS = 0.04; 
const double SAC_MAX_CORRESPONDENCE_DIST = 0.001; 

const float MIN_SCALE = 0.0005; 
const int NR_OCTAVES = 4; 
const int NR_SCALES_PER_OCTAVE = 5; 
const float MIN_CONTRAST = 1; 

#include "openni_capture.h"
using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;



//convenient typedefs
// moved to the openni_capture.h

// This is a tutorial so we can afford having global variables 
	//our visualizer
	pcl::visualization::PCLVisualizer *p;
	//its left and right viewports
	int vp_1, vp_2, vp_3, vp_4;
	int vp2_1, vp2_2, vp2_3, vp2_4;
	float dwStart,serialTime,ParallelTime;
	 Eigen::Matrix4f GlobalTransformSerial[20],GlobalTransformSerial2[20];
	 Eigen::Matrix4f GlobalTransformParallel[20],GlobalTransformParallel2[20];
//convenient structure to handle our pointclouds
struct PCD
{
  PointCloud::Ptr cloud;
  std::string f_name;

  PCD() : cloud (new PointCloud) {};
};
struct SIFT_PCD
{
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
	
	SIFT_PCD() : cloud (new pcl::PointCloud<pcl::PointXYZI>) {};
}siftCloud[32];

struct PCD_OUTPUT
{
  PointCloud::Ptr cloud;
  std::string f_name;

  PCD_OUTPUT() : cloud (new PointCloud) {};
};

std::vector<PCD_OUTPUT , Eigen::aligned_allocator<PCD_OUTPUT> > outclouds,outcloudsparallel,outclouds2,initoutclouds,initoutclouds2;
//std::vector<SIFT_PCD , Eigen::aligned_allocator<SIFT_PCD> > siftData;

struct PCDComparator
{
  bool operator () (const PCD& p1, const PCD& p2)
  {
    return (p1.f_name < p2.f_name);
  }
};


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



////////////////////////////////////////////////////////////////////////////////
/** \brief Display source and target on the first viewport of the visualizer
 *
 */
void showCloudsLeft(const PointCloud::Ptr cloud_target, const PointCloud::Ptr cloud_source)
{
  p->removePointCloud ("vp1_target");
  p->removePointCloud ("vp1_source");

  /* PointCloudColorHandlerCustom<PointT> tgt_h (cloud_target, 0, 255, 0);
  PointCloudColorHandlerCustom<PointT> src_h (cloud_source, 255, 0, 0);
  p->addPointCloud (cloud_target, tgt_h, "vp1_target", vp_1);
  p->addPointCloud (cloud_source, src_h, "vp1_source", vp_1);*/
  
  p->addPointCloud(cloud_target,"vp1_target",vp_1);
  p->addPointCloud (cloud_source,"vp1_source",vp_1);
  PCL_INFO ("Press q to begin the registration.\n");
  p-> spin();
}


////////////////////////////////////////////////////////////////////////////////
/** \brief Display source and target on the second viewport of the visualizer
 *
 */
void showCloudsRight(const PointCloudWithNormals::Ptr cloud_target, const PointCloudWithNormals::Ptr cloud_source)
{
  p->removePointCloud ("source");
  p->removePointCloud ("target");


  PointCloudColorHandlerGenericField<PointNormalT> tgt_color_handler (cloud_target, "curvature");
  if (!tgt_color_handler.isCapable ())
      PCL_WARN ("Cannot create curvature color handler!");

  PointCloudColorHandlerGenericField<PointNormalT> src_color_handler (cloud_source, "curvature");
  if (!src_color_handler.isCapable ())
      PCL_WARN ("Cannot create curvature color handler!");


  p->addPointCloud (cloud_target, tgt_color_handler, "target", vp_2);
  p->addPointCloud (cloud_source, src_color_handler, "source", vp_2);

  //p->addPointCloud (cloud_source, src_color_handler, "source2", vp_3);
 
  p->spinOnce();
}

////////////////////////////////////////////////////////////////////////////////
/** \Clear all viewport of the visualizer
 *
 */
void cleanAllClouds()
{
	p->removeAllPointClouds(vp_1);
	p->removeAllPointClouds(vp_2);
	p->removeAllPointClouds(vp_3);
	p->spinOnce();
}


////////////////////////////////////////////////////////////////////////////////
/** \brief Load a set of PCD files that we want to register together
  * \param argc the number of arguments (pass from main ())
  * \param argv the actual command line arguments (pass from main ())
  * \param models the resultant vector of point cloud datasets
  */
void loadData (int argc, char **argv, std::vector<PCD, Eigen::aligned_allocator<PCD> > &models)
{
  std::string extension (".pcd");
  // Suppose the first argument is the actual test model
  for (int i = 1; i < argc; i++)
  {
    std::string fname = std::string (argv[i]);
    // Needs to be at least 5: .plot
    if (fname.size () <= extension.size ())
      continue;

    std::transform (fname.begin (), fname.end (), fname.begin (), (int(*)(int))tolower);

    //check that the argument is a pcd file
    if (fname.compare (fname.size () - extension.size (), extension.size (), extension) == 0)
    {
      // Load the cloud and saves it into the global list of models
      PCD m;
      m.f_name = argv[i];
      pcl::io::loadPCDFile (argv[i], *m.cloud);
      //remove NAN points from the cloud
      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(*m.cloud,*m.cloud, indices);
	  
      models.push_back (m);
    }
	cerr<<i<<" no cloud loaded";
  }

}


void captureData (int argc, char **argv, std::vector<PCD, Eigen::aligned_allocator<PCD> > &models)
{

	int nr_frames;

  
  std::cout<<"Enter No of Snaps";
  std::cin>> nr_frames;

  if(nr_frames>0)
  {
	  	  
		 

		  OpenNICapture camera;
		  camera.setTriggerMode (true);
		  for (int i = 0; i < nr_frames; ++i)
		  {
			  PCD m;	
			  m.f_name=""+i;
			  m.cloud=camera.snap();
			  //remove NAN points from the cloud
			  std::vector<int> indices;
			  pcl::removeNaNFromPointCloud(*m.cloud,*m.cloud, indices);
				
			  models.push_back (m);
			  std::cout<<endl << i << "captured frame";
		  }

  }
}

 void smothCloudsOMP(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud )
 {
	 cerr<<cloud<<"Processing..";
	  pcl::MovingLeastSquaresOMP<pcl::PointXYZRGB, pcl::PointNormal> mls;
	  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
	  pcl::PointCloud<pcl::PointNormal> mls_points;
	  mls.setComputeNormals (true);
	  
	  // Set parameters
	  mls.setInputCloud (cloud);
	  mls.setPolynomialFit (true);
	  mls.setSearchMethod (tree);
	  mls.setSearchRadius (0.3);
	  cerr<<"Para Set";
	  // Reconstruct
	  mls.process (mls_points);
		cerr<<"Done";
 }
 
 pcl::PointCloud<pcl::Normal>::Ptr getNormals_New( const pcl::PointCloud<pcl::PointXYZRGB>::Ptr incloud ) { 

        pcl::PointCloud<pcl::Normal>::Ptr normalsPtr (new pcl::PointCloud<pcl::Normal>); 
		pcl::NormalEstimation<PointT, pcl::Normal> norm_est;
		pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
		norm_est.setSearchMethod(tree);
		
        norm_est.setInputCloud( incloud ); 
        norm_est.setRadiusSearch( NORMALS_RADIUS ); 
        norm_est.compute( *normalsPtr ); 
		cerr<<"\nNormal size "<<normalsPtr->points.size();
        return normalsPtr; 
}

 void compute_FPFH_features(pcl::PointCloud<pcl::PointXYZRGB>::Ptr points, pcl::PointCloud<pcl::Normal>::Ptr normals, pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints, float feature_radius,
pcl::PointCloud<pcl::FPFHSignature33>::Ptr &descriptors_out)
{
	 cerr<<"\nIn FPFH";
	pcl::FPFHEstimationOMP<pcl::PointXYZRGB, pcl::Normal,pcl::FPFHSignature33> fpfh_est;
	//pcl::FPFHEstimation<pcl::PointXYZRGB, pcl::Normal,pcl::FPFHSignature33> fpfh_est2;

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::copyPointCloud (*keypoints,*keypoints_xyzrgb);
	
	/*p->setBackgroundColor(0,1,0,vp_3);
	p->addPointCloud(keypoints_xyzrgb,"siftRgb",vp_3);
	p->spinOnce();*/

	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> () );
	fpfh_est.setSearchMethod(tree);
	fpfh_est.setRadiusSearch (feature_radius);
	fpfh_est.setSearchSurface (points);
	fpfh_est.setInputNormals (normals);
	fpfh_est.setInputCloud (keypoints_xyzrgb);
	//fpfh_est.compute (*descriptors_out);
	
	fpfh_est.compute(*descriptors_out);
	cerr<<" Descrip size "<<descriptors_out->points.size();
	/* for(int i=0;i<descriptors_out->points.size();i++)
		 cerr<<"\n"<<descriptors_out->at(i);
	 */
}


 pcl::PointCloud<pcl::PointXYZI>::Ptr getSIFTKeypoints(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloudIn)
{
	pcl::SIFTKeypoint<pcl::PointXYZRGB,pcl::PointXYZI> siftdetect;
	pcl::SIFTKeypoint<pcl::PointXYZRGB,pcl::PointXYZI>::PointCloudOut output; 
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> () );
	siftdetect.setInputCloud(cloudIn);
	siftdetect.setSearchSurface(cloudIn);
	siftdetect.setSearchMethod(tree);
	siftdetect.setMinimumContrast(MIN_CONTRAST);
	siftdetect.setScales(MIN_SCALE,NR_OCTAVES,NR_SCALES_PER_OCTAVE);
	siftdetect.compute(output);
		
	pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::copyPointCloud(output,*keypoints);
	cerr<<"\nsift size.."<<keypoints->points.size();
	
	return keypoints;
}


////////////////////////////////////////////////////////////////////////////////
/** \brief Align a pair of PointCloud datasets and return the result, modify cloud
  * \param cloud_src the source PointCloud
  * \param cloud_tgt the target PointCloud
  * \param output the resultant aligned source PointCloud
  * \param final_transform the resultant transform between source and target
  */
void pairAlign (const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt, PointCloud::Ptr output, Eigen::Matrix4f &final_transform, bool downsample = true)
{
  //
  // Downsample for consistency and speed
  // \note enable this for large datasets
  PointCloud::Ptr src (new PointCloud);
  PointCloud::Ptr tgt (new PointCloud);
  PointCloud::Ptr stepoutput(new PointCloud);
  pcl::VoxelGrid<PointT> grid;
  if (downsample)
  {
    grid.setLeafSize (0.05, 0.05, 0.05);
    grid.setInputCloud (cloud_src);
    grid.filter (*src);

    grid.setInputCloud (cloud_tgt);
    grid.filter (*tgt);
  }
  else
  {
    src = cloud_src;
    tgt = cloud_tgt;
  }


  // Compute surface normals and curvature
  PointCloudWithNormals::Ptr points_with_normals_src (new PointCloudWithNormals);
  PointCloudWithNormals::Ptr points_with_normals_tgt (new PointCloudWithNormals);

  pcl::NormalEstimationOMP<PointT, PointNormalT> norm_est;
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
  
  norm_est.setSearchMethod (tree);
  norm_est.setKSearch (30);
  
  //if(downsample)
  //norm_est.setSearchSurface(cloud_src);
  
  norm_est.setInputCloud (src);
   time_t starttime = time(NULL);
  norm_est.compute (*points_with_normals_src);
  cerr<<"Time taken"<<time(NULL)-starttime;

  pcl::copyPointCloud (*src, *points_with_normals_src);

  cerr<<"Size of point cloud with normal src "<<points_with_normals_src->points.size()<<endl;

  //if(downsample)
  //norm_est.setSearchSurface(cloud_tgt);
  norm_est.setInputCloud (tgt);
  starttime = time(NULL);
  norm_est.compute (*points_with_normals_tgt);
  cerr<<"Time taken"<<time(NULL)-starttime;
  pcl::copyPointCloud (*tgt, *points_with_normals_tgt);
  cerr<<"Size of point cloud with normal trg "<<points_with_normals_tgt->points.size()<<endl;
  //
  // Instantiate our custom point representation (defined above) ...
  MyPointRepresentation point_representation;
  // ... and weight the 'curvature' dimension so that it is balanced against x, y, and z
  float alpha[4] = {1.0, 1.0, 1.0, 1.0};
  point_representation.setRescaleValues (alpha);

  //
  // Align
  pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;
  reg.setTransformationEpsilon (1e-6);
  // Set the maximum distance between two correspondences (src<->tgt) to 10cm
  // Note: adjust this based on the size of your datasets
  reg.setMaxCorrespondenceDistance (0.1);  
  // Set the point representation
  reg.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation));
  reg.setRANSACOutlierRejectionThreshold( 0.1 ); 
  reg.setInputCloud (points_with_normals_src);
  reg.setInputTarget (points_with_normals_tgt);

  Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), prev, targetToSource;
  PointCloudWithNormals::Ptr reg_result = points_with_normals_src;
  reg.setMaximumIterations(10);
  for (int i = 0; i < 1; ++i)
  {
    PCL_INFO ("Iteration Nr. %d.\n", i);

    // save cloud for visualization purpose
    points_with_normals_src = reg_result;

    // Estimate
    reg.setInputCloud (points_with_normals_src);
    reg.align (*reg_result);
	
		//accumulate transformation between each Iteration
    Ti = reg.getFinalTransformation () * Ti;

		//if the difference between this transformation and the previous one
		//is smaller than the threshold, refine the process by reducing
		//the maximal correspondence distance
    if (fabs ((reg.getLastIncrementalTransformation () - prev).sum ()) < reg.getTransformationEpsilon ())
      reg.setMaxCorrespondenceDistance (reg.getMaxCorrespondenceDistance () - 0.001);
    
    prev = reg.getLastIncrementalTransformation ();

      
  if(reg.getFitnessScore()<0.0015)
	  break;
  }

	//
  // Get the transformation from target to source
  targetToSource = Ti.inverse();

  //
  // Transform target back in source frame
  pcl::transformPointCloud (*cloud_tgt, *output, targetToSource);

  /*p->removePointCloud ("source");
  p->removePointCloud ("target");
  p->removePointCloud ("last");
  PointCloudColorHandlerCustom<PointT> cloud_tgt_h (output, 0, 255, 0);
  PointCloudColorHandlerCustom<PointT> cloud_src_h (cloud_src, 255, 0, 0);
  p->addPointCloud (output, cloud_tgt_h, "target", vp_2);
  p->addPointCloud (cloud_src, cloud_src_h, "source", vp_2);*/



    //add the source to the transformed target to visualize output of this step
  //*stepoutput= *cloud_src;
  //*stepoutput += *output;
  //
  //p->addPointCloud(stepoutput,"last",vp_3);

	//PCL_INFO ("Press q to continue the registration.\n");
 /*  if(reg.getFitnessScore()<0.0015)
	  break;
  }*/
  //p->spin ();
  //cleanAllClouds();
  //p->removePointCloud ("source"); 
  //p->removePointCloud ("target");

  cerr<<cloud_src<<"  "<<cloud_tgt<<" Coverage "<<reg.hasConverged()<< " score: " << reg.getFitnessScore() << std::endl;
  cerr<<targetToSource<<endl;
  
  final_transform = targetToSource;
 }

 ////////////////////////////////////////////////////////////////////////////////
/** \brief ICPOMP call this function, it just return transformation not modifing cloud
  * \param cloud_src source cloud
  * \param cloud_tgt target cloud 
  * \param final_transform matrix
  */
 void pairAlignForOMP (const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt, Eigen::Matrix4f &final_transform, bool downsample = true)
{
  //
  // Downsample for consistency and speed
  // \note enable this for large datasets
  PointCloud::Ptr src (new PointCloud);
  PointCloud::Ptr tgt (new PointCloud);
  PointCloud::Ptr stepoutput(new PointCloud);
  pcl::VoxelGrid<PointT> grid;
  if (downsample)
  {
    grid.setLeafSize (0.05, 0.05, 0.05);
    grid.setInputCloud (cloud_src);
    grid.filter (*src);

    grid.setInputCloud (cloud_tgt);
    grid.filter (*tgt);
  }
  else
  {
    src = cloud_src;
    tgt = cloud_tgt;
  }


  // Compute surface normals and curvature
  PointCloudWithNormals::Ptr points_with_normals_src (new PointCloudWithNormals);
  PointCloudWithNormals::Ptr points_with_normals_tgt (new PointCloudWithNormals);

  pcl::NormalEstimationOMP<PointT, PointNormalT> norm_est;
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
  
  norm_est.setSearchMethod (tree);
  norm_est.setKSearch (30);
  //norm_est.setRadiusSearch();

  //if(downsample)
  //norm_est.setSearchSurface(cloud_src);
  
  norm_est.setInputCloud (src);
   time_t starttime = time(NULL);
  norm_est.compute (*points_with_normals_src);
  cerr<<"Time taken"<<time(NULL)-starttime;

  pcl::copyPointCloud (*src, *points_with_normals_src);

  cerr<<"Size of point cloud with normal src "<<points_with_normals_src->points.size()<<endl;

  //if(downsample)
  //norm_est.setSearchSurface(cloud_tgt);
  norm_est.setInputCloud (tgt);
  starttime = time(NULL);
  norm_est.compute (*points_with_normals_tgt);
  cerr<<"Time taken"<<time(NULL)-starttime;
  pcl::copyPointCloud (*tgt, *points_with_normals_tgt);
  cerr<<"Size of point cloud with normal trg "<<points_with_normals_tgt->points.size()<<endl;
  //
  // Instantiate our custom point representation (defined above) ...
  MyPointRepresentation point_representation;
  // ... and weight the 'curvature' dimension so that it is balanced against x, y, and z
  float alpha[4] = {1.0, 1.0, 1.0, 1.0};
  point_representation.setRescaleValues (alpha);

  //
  // Align
  pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;
  reg.setTransformationEpsilon (1e-6);
  // Set the maximum distance between two correspondences (src<->tgt) to 10cm
  // Note: adjust this based on the size of your datasets
  reg.setMaxCorrespondenceDistance (0.1);  
  // Set the point representation
  reg.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation));
  reg.setRANSACOutlierRejectionThreshold( 0.1 ); 
  reg.setInputCloud (points_with_normals_src);
  reg.setInputTarget (points_with_normals_tgt);
  
  // Run the same optimization in a loop and visualize the results
  Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), prev, targetToSource;
  PointCloudWithNormals::Ptr reg_result = points_with_normals_src;
  reg.setMaximumIterations(10);
  for (int i = 0; i < 1; ++i)
  {
    PCL_INFO ("Iteration Nr. %d.\n", i);

    // save cloud for visualization purpose
    points_with_normals_src = reg_result;

    // Estimate
    reg.setInputCloud (points_with_normals_src);
    reg.align (*reg_result);
	
		//accumulate transformation between each Iteration
    Ti = reg.getFinalTransformation () * Ti;

		//if the difference between this transformation and the previous one
		//is smaller than the threshold, refine the process by reducing
		//the maximal correspondence distance
    if (fabs ((reg.getLastIncrementalTransformation () - prev).sum ()) < reg.getTransformationEpsilon ())
      reg.setMaxCorrespondenceDistance (reg.getMaxCorrespondenceDistance () - 0.001);
    
    prev = reg.getLastIncrementalTransformation ();

 
   if(reg.getFitnessScore()<0.0015)
	  break;
  }
  //p->spin ();
  //cleanAllClouds();
  //p->removePointCloud ("source"); 
  //p->removePointCloud ("target");
  targetToSource = Ti.inverse();
  cerr<<cloud_src<<"  "<<cloud_tgt<<" Coverage "<<reg.hasConverged()<< " score: " << reg.getFitnessScore() << std::endl;
  cerr<<targetToSource<<endl;
  
  final_transform = targetToSource;
 }

 void pairAlignUsingSift(pcl::PointCloud<pcl::PointXYZI>::Ptr &features1,pcl::PointCloud<pcl::PointXYZI>::Ptr &features2,Eigen::Matrix4f &final_transform)
{
 //pcl::IterativeClosestPointNonLinear<pcl::FPFHSignature33,pcl::FPFHSignature33 > reg;
  pcl::IterativeClosestPoint<pcl::PointXYZI,pcl::PointXYZI > reg;
  reg.setTransformationEpsilon (1e-6);
  reg.setMaxCorrespondenceDistance (1);  
  // Set the point representation
  reg.setRANSACOutlierRejectionThreshold( 0.1 ); 
  reg.setInputCloud (features1);
  reg.setInputTarget (features2);
  
  // Run the same optimization in a loop and visualize the results
  Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), targetToSource;
  pcl::PointCloud<pcl::PointXYZI>::Ptr reg_result(new pcl::PointCloud<pcl::PointXYZI>) ;
  reg.setMaximumIterations(10);
        // Estimate   
    reg.align (*reg_result);
	//accumulate transformation between each Iteration
    Ti = reg.getFinalTransformation () ;
	cerr<<Ti;
	targetToSource=Ti.inverse();
	final_transform=targetToSource;
	

}
 
 pcl::Correspondences findCorrespondance(pcl::PointCloud<pcl::FPFHSignature33>::Ptr &features1, pcl::PointCloud<pcl::FPFHSignature33>::Ptr &features2)
 {
	 pcl::registration::CorrespondenceEstimation< pcl::FPFHSignature33, pcl::FPFHSignature33  > cer;
	 pcl::registration::CorrespondenceEstimation <pcl::PointXYZRGB,pcl::PointXYZRGB > cer2;
	 
	 pcl::Correspondences corres,recpCorres;
	 cer.setInputCloud(features1);
	 cer.setInputTarget(features2);
	 
	 cer.determineReciprocalCorrespondences(corres);
	 cerr<<"\nFPFH Corres size "<<corres.size();
	/* cer.determineReciprocalCorrespondences(recpCorres);
	 cerr<<"\nRecip Correspondance size "<<recpCorres.size()<<"\n";
	 
	 for(int i=0;i<recpCorres.size();i++)
		 cerr<<"\n"<<recpCorres.at(i);*/
	 
	 return corres;
 }
 pcl::Correspondences findCorrespondanceFromSift(pcl::PointCloud<pcl::PointXYZI>::Ptr &keypoints1, pcl::PointCloud<pcl::PointXYZI>::Ptr &keypoints2)
 {
	 pcl::registration::CorrespondenceEstimation< pcl::PointXYZI, pcl::PointXYZI  > cer;

	 
	 pcl::Correspondences corres,recpCorres;
	 cer.setInputCloud(keypoints1);
	 cer.setInputTarget(keypoints2);
	 
	 cer.determineReciprocalCorrespondences(corres);
	 cerr<<"\nSIFT Corres size "<<corres.size();
	/* cer.determineReciprocalCorrespondences(recpCorres);
	 cerr<<"\nRecip Correspondance size "<<recpCorres.size()<<"\n";
	 
	 for(int i=0;i<recpCorres.size();i++)
		 cerr<<"\n"<<recpCorres.at(i);*/
	 
	 return corres;
 }

 pcl::CorrespondencesPtr findCorrespondanceNew(pcl::PointCloud<pcl::FPFHSignature33>::Ptr &features1, pcl::PointCloud<pcl::FPFHSignature33>::Ptr &features2)
 {
	 pcl::registration::CorrespondenceEstimation< pcl::FPFHSignature33, pcl::FPFHSignature33  > cer;
	 pcl::CorrespondencesPtr corres,recpCorres;
	 cer.setInputCloud(features1);
	 cer.setInputTarget(features2);
	 cer.determineCorrespondences(*corres);
	
	/* cer.determineReciprocalCorrespondences(recpCorres);
	 cerr<<"\nRecip Correspondance size "<<recpCorres.size()<<"\n";
	 
	 for(int i=0;i<recpCorres.size();i++)
		 cerr<<"\n"<<recpCorres.at(i);*/
	 
	 return corres;
 }

 void viewCorrespondance(pcl::Correspondences corres,int vp)
 {
  
 }
void temp(std::vector<PCD, Eigen::aligned_allocator<PCD> > &data)
 {
	 pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZRGB>);     
     pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZRGB>);     
	 
	 cerr<<"\ncloud size "<<data[0].cloud->points.size();
	 cerr<<"\ncloud size "<<data[1].cloud->points.size();

	 pcl::VoxelGrid<pcl::PointXYZRGB> vox_grid; 
       vox_grid.setLeafSize( VOXEL_GRID_SIZE, VOXEL_GRID_SIZE, VOXEL_GRID_SIZE ); 
	   vox_grid.setInputCloud(data[0].cloud);
	   vox_grid.filter(*cloud1);
	   vox_grid.setInputCloud(data[1].cloud);
	   vox_grid.filter(*cloud2);

	   cerr<<"\nCloud Filter Size "<<cloud1->points.size();
	   cerr<<"\nCloud Filter Size "<<cloud2->points.size();
	 pcl::PointCloud<pcl::PointXYZI>::Ptr siftKeyPoints1 = getSIFTKeypoints(cloud1);
	 pcl::PointCloud<pcl::PointXYZI>::Ptr siftKeyPoints2 = getSIFTKeypoints(cloud2);
		
	  PointCloudColorHandlerCustom<pcl::PointXYZI> tgt_h (siftKeyPoints1, 0, 255, 0);
	  PointCloudColorHandlerCustom<pcl::PointXYZI> src_h (siftKeyPoints2, 255, 0, 0);
	  p->addPointCloud (siftKeyPoints1, tgt_h, "vp1_target", vp_2);
	  p->addPointCloud (siftKeyPoints2, src_h, "vp1_source", vp_2);
	  p->addPointCloud (siftKeyPoints1, tgt_h, "vp3_target", vp_3);
	  p->addPointCloud (siftKeyPoints2, src_h, "vp3_source", vp_3);
	 // p->spin();

	 pcl::PointCloud<pcl::Normal>::Ptr normal1 = getNormals_New(cloud1);
	 pcl::PointCloud<pcl::Normal>::Ptr normal2 = getNormals_New(cloud2);
	 pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptor1(new pcl::PointCloud<pcl::FPFHSignature33>);
	 pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptor2(new pcl::PointCloud<pcl::FPFHSignature33>);
	 compute_FPFH_features(cloud1,normal1,siftKeyPoints1,0.3,descriptor1);
	 compute_FPFH_features(cloud2,normal2,siftKeyPoints2,0.3,descriptor2);
	 cerr<<"Finished";
	
	 /*pcl::visualization::PCLHistogramVisualizer hist; 
	 hist.addFeatureHistogram(*descriptor1,50,"Hist1",640,200);
	 hist.addFeatureHistogram(*descriptor2,50,"Hist2",640,200);
	 hist.spin();*/
	
	 pcl::Correspondences corres =findCorrespondance(descriptor1,descriptor2);
	 pcl::Correspondences corres2= findCorrespondanceFromSift(siftKeyPoints1,siftKeyPoints2);
	  //cerr<<"\nCorrespondance size "<<corres.size()<<"\n";
	/*viewCorrespondance(corres,vp_3); 
	viewCorrespondance(corres2,vp_2);*/
	
	 /*for(int i=0;i<corres.size();i++)
		 cerr<<"\n"<<corres.at(i);*/
	 
	 
	 p->addPointCloud(data[0].cloud,"cl12",vp_2);
	 p->addPointCloud(data[1].cloud,"cl22",vp_2);
	 p->addPointCloud(data[0].cloud,"cl13",vp_3);
	 p->addPointCloud(data[1].cloud,"cl23",vp_3);
	 
	 cerr<<"\nVisual Corre1 \n";
	 for(int  i=0;i<corres.size();i++)
	 {
		 pcl::PointXYZI srcpt=siftKeyPoints1->points.at(corres[i].index_query);
		 pcl::PointXYZI tgtpt=siftKeyPoints2->points.at(corres[i].index_match);
		 
		 //cerr<<"\n"<<srcpt.x<<" "<<srcpt.y<<" "<<srcpt.z;
		 //cerr<<"\n"<<tgtpt.x<<" "<<tgtpt.y<<" "<<tgtpt.z;
		
		 std::stringstream ss1 ("line_vp_1_");
		 std::stringstream ss2 ("line_vp_2_");
		 std::stringstream ss3 ("line_vp_3_");
		 ss1<<i;
		 ss2<<i;
		 ss3<<i;
		 p->addLine<pcl::PointXYZI>(srcpt,tgtpt,ss2.str(),vp_2);
		 //p->addLine<pcl::PointXYZRGB>(srcpt,tgtpt,1,0,0,ss1.str(),vp_1);
		 //p->addLine<pcl::PointXYZRGB>(srcpt,tgtpt,1,0,0,ss2.str(),vp_2);

	 }

	cerr<<"\nVisual Corre1 \n";	
	 for(int  i=0;i<corres2.size();i++)
	 {
		 pcl::PointXYZI srcpt=siftKeyPoints1->points.at(corres2[i].index_query);
		 pcl::PointXYZI tgtpt=siftKeyPoints2->points.at(corres2[i].index_match);
		 /*cerr<<"\n"<<srcpt.x<<" "<<srcpt.y<<" "<<srcpt.z;
		 cerr<<"\n"<<tgtpt.x<<" "<<tgtpt.y<<" "<<tgtpt.z;*/
		 /*pcl::PointXYZRGB srcpt=cloud1->points.at(corres[i].index_match);
		 pcl::PointXYZRGB tgtpt=cloud2->points.at(corres[i].index_query);*/
		 std::stringstream ss1 ("line_vp_1_");
		 std::stringstream ss2 ("line_vp_2_");
		 std::stringstream ss3 ("line_vp_3_");
		 ss1<<i;
		 ss2<<i;
		 ss3<<i;
		 //p->addLine<pcl::PointXYZRGB>(srcpt,tgtpt,1,0,0,ss2.str(),vp_2);
		 p->addLine<pcl::PointXYZI>(srcpt,tgtpt,ss3.str(),vp_3);
		 //p->addLine<pcl::PointXYZRGB>(srcpt,tgtpt,1,0,0,ss1.str(),vp_1);
		 //p->addLine<pcl::PointXYZRGB>(srcpt,tgtpt,1,0,0,ss2.str(),vp_2);

	 }
	 
	 p->spin();
	 p->removeAllPointClouds(vp_2);
	 p->removeAllPointClouds(vp_3);
	 p->spin();

		/* pcl::CorrespondencesPtr corresPtr =findCorrespondanceNew(descriptor1,descriptor2);
	  cerr<<"\nCorrespondance size "<<corresPtr->size()<<"\n";
*/
boost::shared_ptr<pcl::Correspondences > corresPtr (new pcl::Correspondences(corres));
		boost::shared_ptr<pcl::Correspondences > newCorres (new pcl::Correspondences()) ;
		//pcl::CorrespondencesPtr  newCorres=new pcl::CorrCorrespondencesPtr();
		 cerr<<"\nPtr correct";
		 pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZI> crsc;
		 crsc.setInputCloud(siftKeyPoints1);
		 crsc.setTargetCloud(siftKeyPoints2);
		 crsc.setInputCorrespondences(corresPtr);
		 
		 crsc.setMaxIterations(200);
		 cerr<<"\nIP correct";
		 crsc.getCorrespondences(*newCorres);
		 Eigen::Matrix4f initialTransformation=crsc.getBestTransformation();
		 cerr<<"\n"<<initialTransformation;
		 cerr<<"\nNew Corres Size "<<newCorres->size();

		  cerr<<"\nVisual new Corre \n";
		  p->removeAllShapes(vp_2);
		  //p->removeAllShapes(vp_2);
	 for(int  i=0;i<newCorres->size();i++)
	 {
		 pcl::PointXYZI srcpt=siftKeyPoints1->points.at( newCorres->at(i).index_query);
		 pcl::PointXYZI tgtpt=siftKeyPoints2->points.at(newCorres->at(i).index_match);
		 
		/* cerr<<"\n"<<srcpt.x<<" "<<srcpt.y<<" "<<srcpt.z;
		 cerr<<"\n"<<tgtpt.x<<" "<<tgtpt.y<<" "<<tgtpt.z;*/
		
		 std::stringstream ss1 ("line_vp_1_");
		 std::stringstream ss2 ("line_vp_2_");
		 std::stringstream ss3 ("newline_vp_3_");
		 ss1<<i;
		 ss2<<i;
		 ss3<<i;
		 
		 p->addLine<pcl::PointXYZI>(srcpt,tgtpt,ss3.str(),vp_2);
		 //p->addLine<pcl::PointXYZRGB>(srcpt,tgtpt,1,0,0,ss1.str(),vp_1);
		 //p->addLine<pcl::PointXYZRGB>(srcpt,tgtpt,1,0,0,ss2.str(),vp_2);

	 }


	 boost::shared_ptr<pcl::Correspondences > corres2Ptr (new pcl::Correspondences(corres2));
		boost::shared_ptr<pcl::Correspondences > newCorres2 (new pcl::Correspondences()) ;
		//pcl::CorrespondencesPtr  newCorres=new pcl::CorrCorrespondencesPtr();
		 cerr<<"\nPtr correct";
		 pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZI> crsc2;
		 crsc2.setInputCloud(siftKeyPoints1);
		 crsc2.setTargetCloud(siftKeyPoints2);
		 crsc2.setInputCorrespondences(corres2Ptr);
		 
		 crsc2.setMaxIterations(200);
		 cerr<<"\nIP correct";
		 crsc2.getCorrespondences(*newCorres2);
		 
		 Eigen::Matrix4f initialTransformation2=crsc2.getBestTransformation();	
		 cerr<<"\n"<<initialTransformation2;
		 cerr<<"\nNew Corres Size "<<newCorres2->size();
		 p->removeAllShapes(vp_3);
		  for(int  i=0;i<newCorres2->size();i++)
	 {
		 pcl::PointXYZI srcpt=siftKeyPoints1->points.at( newCorres2->at(i).index_query);
		 pcl::PointXYZI tgtpt=siftKeyPoints2->points.at(newCorres2->at(i).index_match);
		 
		 /*cerr<<"\n"<<srcpt.x<<" "<<srcpt.y<<" "<<srcpt.z;
		 cerr<<"\n"<<tgtpt.x<<" "<<tgtpt.y<<" "<<tgtpt.z;*/
				 
		 std::stringstream ss2 ("line2_vp_2_");
				 
		 ss2<<i;
				 
		 p->addLine<pcl::PointXYZI>(srcpt,tgtpt,ss2.str(),vp_3);
		 
	 }
	 p->spin();

	 /*p->removeAllPointClouds();
	 p->spin();*/
	 //p->removeAllPointClouds(vp_3);
	 //p->removeAllShapes(vp_2);
	 //p->setBackgroundColor(0,0,1);
	 //p->addCorrespondences<pcl::PointXYZI>(siftKeyPoints1,siftKeyPoints2,corres2,"CoRR",vp_2);
	 
	 pcl::visualization::PCLVisualizer *p2=new pcl::visualization::PCLVisualizer();
	 p2->setBackgroundColor(0,0,0);
	 p2->createViewPort (0.0, 0, 0.33, 1.0, vp2_1);
	 p2->createViewPort (0.33, 0, 0.66, 1.0, vp2_2);
	 p2->createViewPort (0.66, 0, 1.0, 1.0, vp2_3);
	 p2->addPointCloud(data[0].cloud,"cl11",vp2_1);
	 p2->addPointCloud(data[1].cloud,"cl12",vp2_1);
	 pcl::PointCloud<pcl::PointXYZRGB>::Ptr result1 (new pcl::PointCloud<pcl::PointXYZRGB>);
	 pcl::PointCloud<pcl::PointXYZRGB>::Ptr result2 (new pcl::PointCloud<pcl::PointXYZRGB>);

	 Eigen::Matrix4f initialTransformationInverse=initialTransformation.inverse();
	 pcl::transformPointCloud(*data[1].cloud,*result1,initialTransformationInverse);
	 p2->addPointCloud(data[0].cloud,"cl21",vp2_2);
	 p2->addPointCloud(result1,"cl22",vp2_2);

	/* pcl::transformPointCloud(*data[0].cloud,*result1,initialTransformation);
	 p2->addPointCloud(data[0].cloud,"cl21",vp2_2);
	 p2->addPointCloud(result1,"cl22",vp2_2);*/
	 
	 pcl::transformPointCloud(*data[0].cloud,*result2,initialTransformation2);
	 p2->addPointCloud(data[1].cloud,"cl31",vp2_3);
	 p2->addPointCloud(result2,"cl32",vp2_3);

	 /*PCD_OUTPUT po;
	 po.cloud=data[0].cloud;
	 initoutclouds.push_back(po);
	 initoutclouds2.push_back(po);
	 pcl::transformPointCloud(*data[1].cloud,*po.cloud,initialTransformation);
	 initoutclouds.push_back( po);
	 pcl::transformPointCloud(*data[1].cloud,*po.cloud,initialTransformation2);
	 initoutclouds2.push_back( po);
	  for(int  i=0;i<initoutclouds2.size();i++)
	 {
		 std::stringstream ss1 ("CloudOut1vp_1_");
		 std::stringstream ss2 ("CloudOut2vp_1_");
		 std::stringstream ss3 ("CloudOut3vp_1_");
		 ss1<<i;
		 ss2<<i;
		 ss3<<i;
		 p2->addPointCloud(data[i].cloud,ss1.str(),vp2_1);
		 p2->addPointCloud(initoutclouds[i].cloud,ss2.str(),vp2_2);
		 p2->addPointCloud(initoutclouds2[i].cloud,ss3.str(),vp2_3);
	  }*/
	 p2->spin();
}

 void cloudTransformation(Eigen::Matrix4f TransMatrix[],std::vector<PCD, Eigen::aligned_allocator<PCD> > &data)
 {
	 PCD_OUTPUT poc;
	 poc.cloud=data[0].cloud;
	 outcloudsparallel.push_back(poc);
	 PointCloud::Ptr result (new PointCloud ); 
	
	 for(int i=1;i<data.size();i++)
	 {
	
				cerr<<"\n_____ "<<i;
				pcl::transformPointCloud (*data[i].cloud, *result, TransMatrix[i]);
				cerr<<"\n^^^^ "<<i;
				// Pushtransformed cloud into outclouds
				PCD_OUTPUT poc;
				poc.cloud=result;
				outcloudsparallel.push_back(poc);
	 }
 
	 for (int i=0;i<outcloudsparallel.size();i++)
  {		
	  std::string ss=i+"zxcvbnmasdfghjklqwertyuiop";
	  	cerr<<ss;	
	  p->addPointCloud(outcloudsparallel[i].cloud,ss,vp_2);
  }
 }

 ////////////////////////////////////////////////////////////////////////////////
/** \brief Serial code to call pairAlign , cloud transformation in 2 steps
  * \param data captured frames
  */
 void callICP(std::vector<PCD, Eigen::aligned_allocator<PCD> > &data)
 {
  dwStart = GetTickCount();
  PointCloud::Ptr result (new PointCloud), source, target;
  Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity (), pairTransform;
  
  //add first frame to output cloud
  PCD_OUTPUT oc;
  oc.cloud=data[0].cloud;
  outclouds.push_back(oc);
  PointCloud::Ptr final(new PointCloud);
  *final=*data[0].cloud;

  for (size_t i = 1; i < data.size (); ++i)
	  {
			source = data[i-1].cloud;
			target = data[i].cloud;
			
			PointCloud::Ptr temp (new PointCloud);
			PCL_INFO ("Aligning %d (%d) with %d (%d).\n", i-1, source->points.size (), i, target->points.size ());
			pairAlign (source, target, temp, pairTransform, true);
			//GlobalTransformSerial[i*2]=pairTransform;
			
			PCL_INFO("Alignment Complete");
			std::cout<<"Alignment Done";
			//transform current pair into the global transform
			pcl::transformPointCloud (*temp, *result, GlobalTransform);
			
			cerr<<"Alignment Between"<<i<<" & "<<i-1<<"\n"; 
			cerr<<pairTransform*GlobalTransform;
			
			// Pushtransformed cloud into outclouds
				PCD_OUTPUT oc;
				oc.cloud=result;
				outclouds.push_back(oc);

				*final += *result;
			//update the global transform
			GlobalTransform = pairTransform * GlobalTransform;
			GlobalTransformSerial[i]=GlobalTransform;
			
	  }
  cerr<<"Registration Finished.";

  serialTime=GetTickCount()-dwStart;
  cerr<<"Serial Time Taken "<<serialTime;

  
  for (int i=0;i<outclouds.size();i++)
  {		
	  std::string ss=i+"_thCloudBigStringForCloudIdentifier_ ";
	  cerr<<ss;
	  p->addPointCloud(outclouds[i].cloud,ss,vp_2);
  }

   //p->addPointCloud(final,"Serial",vp_3);
 

  //p->spin();
 }

////////////////////////////////////////////////////////////////////////////////
/** \brief Parallely calling pairAlignForOMP function for getting transformation, cloud transformation serially
  * \param data captured frames
  */
 void callICP_OMP_Sift(std::vector<PCD, Eigen::aligned_allocator<PCD> > &data)
 {
 dwStart = GetTickCount();
 pcl::PointCloud<pcl::PointXYZI>::Ptr  source, target;
  Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity ();
  Eigen::Matrix4f pairTransform[50];
  Eigen::Matrix4f pairTransform2[50];
  pairTransform2[0]= Eigen::Matrix4f::Identity ();
  pairTransform[0]= Eigen::Matrix4f::Identity ();

  PointCloud::Ptr final(new PointCloud);
  *final=*data[0].cloud;
  
  PointCloud::Ptr final2(new PointCloud);
  *final2=*data[0].cloud;
   
  #pragma omp parallel for shared(pairTransform) 
	  for (int i = 1; i < data.size (); ++i)
		  {
				pcl::PointCloud<pcl::PointXYZI>::Ptr siftKeyPoints = getSIFTKeypoints(data[i].cloud);
				siftCloud[i].cloud=siftKeyPoints;
				cerr<<"\n"<<i<<" Done";			
		  }

  #pragma omp parallel for shared(pairTransform) private(source,target)
	  for (int i = 1; i < data.size (); ++i)
		  {
			
			source = siftCloud[i-1].cloud;
			target = siftCloud[i].cloud;
						
			PCL_INFO ("Aligning %d (%d) with %d (%d).\n", i-1, source->points.size (), i, target->points.size ());
			pairAlignUsingSift(source, target, pairTransform[i]);
    		}
			
			std::cout<<"Alignment Done";

	  #pragma omp parallel 
	{
		#pragma omp for ordered
			for (int i = 1; i < data.size (); ++i)
			{
				 #pragma omp ordered
				{
					
				pairTransform2[i]=pairTransform[i];
				pairTransform[i]=pairTransform[i]*pairTransform[i-1];
				
				pairTransform2[i]=pairTransform2[i-1]*pairTransform2[i];
				GlobalTransformParallel[i]=pairTransform[i];
				GlobalTransformParallel2[i]=pairTransform2[i];
				std::cout<<"\n"<<i<<" thread "<<omp_get_thread_num();
				cerr<<"\n alignment between "<<i<<" & "<<i-1<<"\n"<<pairTransform[i]<<"\n"<<pairTransform2[i];
				}
			}
	}

	PCD_OUTPUT poc;
	 poc.cloud=data[0].cloud;
	 outcloudsparallel.push_back(poc);
	 PointCloud::Ptr result (new PointCloud ); 
	

	 for(int i=1;i<data.size();i++)
	 {
	
				cerr<<"\n_____ "<<i;
				pcl::transformPointCloud (*data[i].cloud, *result, GlobalTransformParallel[i]);
				cerr<<"\n^^^^ "<<i;
				// Pushtransformed cloud into outclouds
				PCD_OUTPUT poc;
				poc.cloud=result;
				outcloudsparallel.push_back(poc);
	 }
 
	 for (int i=0;i<outcloudsparallel.size();i++)
  {		
	  std::string ss=i+"zxcvbnmasdfghjklqwertyuiopzxcvbnmasdfghjklqwertyuiop";
	  	cerr<<ss;	
	  p->addPointCloud(outcloudsparallel[i].cloud,ss,vp_2);
  }

			
  cerr<<"Registration Finished.";
   ParallelTime=GetTickCount()-dwStart;
  cerr<<"Parallel Time Taken "<<ParallelTime;
 

 }

////////////////////////////////////////////////////////////////////////////////
/** \brief Parallely calling pairAlignForOMP function for getting transformation, cloud transformation serially
  * \param data captured frames
  */
 void callICP_OMP(std::vector<PCD, Eigen::aligned_allocator<PCD> > &data)
 {
 dwStart = GetTickCount();
  PointCloud::Ptr  source, target;
  Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity ();
  Eigen::Matrix4f pairTransform[50];
  Eigen::Matrix4f pairTransform2[50];
  pairTransform2[0]= Eigen::Matrix4f::Identity ();
  pairTransform[0]= Eigen::Matrix4f::Identity ();

  PointCloud::Ptr final(new PointCloud);
  *final=*data[0].cloud;
  
  PointCloud::Ptr final2(new PointCloud);
  *final2=*data[0].cloud;
  
  int th_id;
  #pragma omp parallel for shared(pairTransform) private(source,target)
	  for (int i = 1; i < data.size (); ++i)
		  {
			 th_id=omp_get_thread_num();
			source = data[i-1].cloud;
			target = data[i].cloud;
						
			PCL_INFO ("Aligning %d (%d) with %d (%d).\n", i-1, source->points.size (), i, target->points.size ());
			pairAlignForOMP (source, target,  pairTransform[i], true);
    		}
			
			std::cout<<"Alignment Done";

	  #pragma omp parallel 
	{
		#pragma omp for ordered
			for (int i = 1; i < data.size (); ++i)
			{
				 #pragma omp ordered
				{
					
				pairTransform2[i]=pairTransform[i];
				pairTransform[i]=pairTransform[i]*pairTransform[i-1];
				
				pairTransform2[i]=pairTransform2[i-1]*pairTransform2[i];
				GlobalTransformParallel[i]=pairTransform[i];
				GlobalTransformParallel2[i]=pairTransform2[i];
				std::cout<<"\n"<<i<<" thread "<<omp_get_thread_num();
				cerr<<"\n alignment between "<<i<<" & "<<i-1<<"\n"<<pairTransform[i]<<"\n"<<pairTransform2[i];
				}
			}
	}

	PCD_OUTPUT poc;
	 poc.cloud=data[0].cloud;
	 outcloudsparallel.push_back(poc);
	 PointCloud::Ptr result (new PointCloud ); 
	

	 for(int i=1;i<data.size();i++)
	 {
	
				cerr<<"\n_____ "<<i;
				pcl::transformPointCloud (*data[i].cloud, *result, GlobalTransformParallel[i]);
				cerr<<"\n^^^^ "<<i;
				// Pushtransformed cloud into outclouds
				PCD_OUTPUT poc;
				poc.cloud=result;
				outcloudsparallel.push_back(poc);
	 }
 
	 for (int i=0;i<outcloudsparallel.size();i++)
  {		
	  std::string ss=i+"zxcvbnmasdfghjklqwertyuiopzxcvbnmasdfghjklqwertyuiop";
	  	cerr<<ss;	
	  p->addPointCloud(outcloudsparallel[i].cloud,ss,vp_2);
  }

			
  cerr<<"Registration Finished.";
   ParallelTime=GetTickCount()-dwStart;
  cerr<<"Parallel Time Taken "<<ParallelTime;
 

 }


 void checkMatrices(Eigen::Matrix4f Matrix1[],Eigen::Matrix4f Matrix2[],int size)
 {
	 for(int i=2;i<size;i++)
	 {
		 /*if(GlobalTransformSerial[i].coeff == GlobalTransformParallel[i].coeff)
					if(GlobalTransformSerial[i].data == GlobalTransformParallel[i].data)*/
						if (!(Matrix1[i].operator==( Matrix2[i])) )
				cerr<<"\nMissmatch"<<i;
			cerr<<"\nchecked"<<i;				
	 }
	
 }


int main (int argc, char** argv)
{
	
	float speedUp,efficiency;
  // Load data
  std::vector<PCD, Eigen::aligned_allocator<PCD> > data;
  
  std::cout<<"Data Loading.. ";
  //loadData (argc, argv, data);
  captureData(argc, argv, data);
 
  std::cout<<"Data collocted";
  // Check user input
  if (data.empty ())
  {
    PCL_ERROR ("Syntax is: %s <source.pcd> <target.pcd> [*]", argv[0]);
    PCL_ERROR ("[*] - multiple files can be added. The registration results of (i, i+1) will be registered against (i+2), etc");
    return (-1);
  }
  PCL_INFO ("Loaded %d datasets.", (int)data.size ());
  
  // Create a PCLVisualizer object
  
  /*p->createViewPort (0.0, 0, 0.5, 1.0, vp_1);
  p->createViewPort (0.5, 0, 1.0, 1.0, vp_2);
*/
  p = new pcl::visualization::PCLVisualizer (argc, argv, "Pairwise Incremental Registration example");
  p->createViewPort (0.0, 0, 0.33, 1.0, vp_1);
  p->createViewPort (0.33, 0, 0.66, 1.0, vp_2);
  p->createViewPort (0.66, 0, 1.0, 1.0, vp_3);

  //p->createViewPort (0.0, 0, 0.50, 1.0, vp_1);
  //p->createViewPort (0.50, 0, 1.0, 1.0, vp_2);


   PointCloud::Ptr inputcloud(new PointCloud);
   *inputcloud =*data[0].cloud;
  for(int i=1;i<data.size();i++)
  {
	  *inputcloud += *data[i].cloud;
  }

  p->addPointCloud(inputcloud,"Inputcloud",vp_1);
// callICP(data);
 //callICPNew(data);
  temp(data);
  //callICP_OMP_Sift(data);
   //callICP_OMP(data);
 //  callICP_OMP_New(data);
   cerr<<"Checking 1";
  // checkMatrices(GlobalTransformSerial,GlobalTransformParallel,data.size());
   /*cerr<<"Checking 2";
   checkMatrices(GlobalTransformParallel,GlobalTransformParallel2,data.size());
   cloudTransformation(GlobalTransformParallel,data);*/

  //serialTime=462432;

  cerr<<"\nSerial Time Taken "<<serialTime;
  cerr<<"\nParallel Time Taken "<<ParallelTime;
  speedUp=(serialTime/ParallelTime)*100;

  cerr<<"\nSpeedup"<<speedUp<<"%";
  efficiency=speedUp/8;
  cerr<<"\nEfficiency"<<efficiency<<"%";
  cerr<<"\nsizes\n";
  cerr<<outclouds.size()<<outclouds2.size()<<outcloudsparallel.size();
  p->spin();
  
 // callSACIA(data);
 
  getch();
	
//  pcl::registration::ELCH<PointType> elch;
//for (int i = 1; i < n; i++)
//{
//elch.addPointCloud (cloud[i]);
//}
//elch.setLoopStart (first);
//elch.setLoopEnd (last);
//elch.compute ();
}






/*
{
#include <pcl/filters/statistical_outlier_removal.h>
 // Create the filtering object
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud (cloud);
  sor.setMeanK (50);
  sor.setStddevMulThresh (1.0);
  sor.filter (*cloud_filtered);2222ji8gjkidinckcj5828281]
  {!
}*/


/* ]--- */