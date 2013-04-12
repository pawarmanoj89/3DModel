
#include <boost/make_shared.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>

#include <pcl/io/pcd_io.h>


#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>

#include <pcl/features/normal_3d.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>

#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/visualization/pcl_visualizer.h>



#include <time.h>
#include <pcl/registration/ia_ransac.h>
//#include <pcl/features/pfh.h>
#include <pcl/filters/passthrough.h>
//#include <pcl/visualization/cloud_viewer.h>
//#include <limits>
//#include <fstream>
//#include <vector>
//#include <Eigen/Core>
//#include "pcl/kdtree/kdtree_flann.h" 
#include "pcl/filters/passthrough.h" 
#include "pcl/filters/voxel_grid.h" 
#include "pcl/features/fpfh.h" 

#include <pcl/surface/mls.h>

const double FILTER_LIMIT = 1000.0; 
const int MAX_SACIA_ITERATIONS = 500; 

//units are meters: 
const float VOXEL_GRID_SIZE = 0.03; 
const double NORMALS_RADIUS = 0.04; 
const double FEATURES_RADIUS = 0.04; 
const double SAC_MAX_CORRESPONDENCE_DIST = 0.001; 


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

//convenient structure to handle our pointclouds
struct PCD
{
  PointCloud::Ptr cloud;
  std::string f_name;

  PCD() : cloud (new PointCloud) {};
};

struct PCD_OUTPUT
{
  PointCloud::Ptr cloud;
  std::string f_name;

  PCD_OUTPUT() : cloud (new PointCloud) {};
};

std::vector<PCD_OUTPUT , Eigen::aligned_allocator<PCD_OUTPUT> > outclouds;

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



















void view( const PointCloud::Ptr cloud ) { 

        p->addPointCloud(cloud,"vp1_target",vp_1);
		
		p->spin();

} 



void filterCloud( pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc ) { 

    pcl::PassThrough<pcl::PointXYZRGB> pass; 
    pass.setInputCloud(pc); 
    pass.setFilterFieldName("x"); 
    pass.setFilterLimits(0, FILTER_LIMIT); 
    pass.setFilterFieldName("y"); 
    pass.setFilterLimits(0, FILTER_LIMIT); 
    pass.setFilterFieldName("z"); 
    pass.setFilterLimits(0, FILTER_LIMIT); 
    pass.filter(*pc);   

} 


pcl::SampleConsensusInitialAlignment<pcl::PointXYZRGB, pcl::PointXYZRGB, pcl::FPFHSignature33>
         align( pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2, 
                        pcl::PointCloud<pcl::FPFHSignature33>::Ptr features1, pcl::PointCloud<pcl::FPFHSignature33>::Ptr features2 ) { 

         pcl::SampleConsensusInitialAlignment<pcl::PointXYZRGB, pcl::PointXYZRGB, pcl::FPFHSignature33> sac_ia; 
		 
         Eigen::Matrix4f final_transformation;	
         sac_ia.setInputCloud( cloud2 ); 
         sac_ia.setSourceFeatures( features2 ); 
         sac_ia.setInputTarget( cloud1 ); 
         sac_ia.setTargetFeatures( features1 ); 
         sac_ia.setMaximumIterations( MAX_SACIA_ITERATIONS ); 
		 pcl::PointCloud<pcl::PointXYZRGB>::Ptr finalcloud;	
         sac_ia.align( *finalcloud ); 
		 cerr<<"Alignment Done";
         return sac_ia; 
} 




pcl::PointCloud<pcl::FPFHSignature33>::Ptr getFeatures( const PointCloud::Ptr cloud, const PointCloudWithNormals::Ptr normals ) { 

        
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr features (new pcl::PointCloud<pcl::FPFHSignature33> ());
		pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
         
		 pcl::FPFHEstimation<pcl::PointXYZRGB, pcl::PointNormal, pcl::FPFHSignature33> fpfh_est;
        fpfh_est.setInputCloud( cloud ); 
        fpfh_est.setInputNormals( normals ); 
        fpfh_est.setSearchMethod( tree ); 
        fpfh_est.setRadiusSearch( FEATURES_RADIUS ); 
        fpfh_est.compute( *features ); 
        return features; 
} 







PointCloudWithNormals::Ptr getNormals( const pcl::PointCloud<pcl::PointXYZRGB>::Ptr incloud ) { 

        PointCloudWithNormals::Ptr normalsPtr = PointCloudWithNormals::Ptr (new PointCloudWithNormals); 
		pcl::NormalEstimation<PointT, PointNormalT> norm_est;
		pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
		norm_est.setSearchMethod(tree);
		
        norm_est.setInputCloud( incloud ); 
        norm_est.setRadiusSearch( NORMALS_RADIUS ); 
        norm_est.compute( *normalsPtr ); 
        return normalsPtr; 
} 


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

 /* for(int i=0 ; i< models[0].cloud->size(); i++ )
  {
	  cerr<<models[0].cloud->points[i].x;
  }

  PCD m;
   m.f_name = argv[0];
   //m.cloud=models[0].cloud;
   cerr<<"name done";
   m.cloud->points.resize(models[0].cloud->width*models[0].cloud->height);
  for(int i=0 ; i< models[0].cloud->size(); i++ )
  {
  m.cloud->points[i].x =models[0].cloud->points[i].x +0.05f ;
  m.cloud->points[i].y =models[0].cloud->points[i].y +0.02f;
  m.cloud->points[i].z =models[0].cloud->points[i].z +0.07f;
  m.cloud->points[i].rgba =models[0].cloud->points[i].rgba;
  }
  cerr<<"Data done";
  models.push_back(m);
  cerr<<" Second cloud loaded";
  */
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

 void smothClouds(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud )
 {
	 cerr<<cloud<<"Processing..";
	  pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointNormal> mls;
	  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
	  pcl::PointCloud<pcl::PointNormal> mls_points;
	  mls.setComputeNormals (true);

	  // Set parameters
	  mls.setInputCloud (cloud);
	  mls.setPolynomialFit (true);
	  mls.setSearchMethod (tree);
	  mls.setSearchRadius (0.03);

	  // Reconstruct
	  mls.process (mls_points);
		cerr<<"Done";
 }


////////////////////////////////////////////////////////////////////////////////
/** \brief Align a pair of PointCloud datasets and return the result
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

  pcl::NormalEstimation<PointT, PointNormalT> norm_est;
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
  
  norm_est.setSearchMethod (tree);
  norm_est.setKSearch (30);
  //norm_est.setRadiusSearch();

  //if(downsample)
  //norm_est.setSearchSurface(cloud_src);
  
  norm_est.setInputCloud (src);
  norm_est.compute (*points_with_normals_src);
  pcl::copyPointCloud (*src, *points_with_normals_src);

  cerr<<"Size of point cloud with normal src "<<points_with_normals_src->points.size()<<endl;

  //if(downsample)
  //norm_est.setSearchSurface(cloud_tgt);
  norm_est.setInputCloud (tgt);
  norm_est.compute (*points_with_normals_tgt);
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



  //
  // Run the same optimization in a loop and visualize the results
  Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), prev, targetToSource;
  PointCloudWithNormals::Ptr reg_result = points_with_normals_src;
  reg.setMaximumIterations(20);
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

    // visualize current state
    showCloudsRight(points_with_normals_tgt, points_with_normals_src);
	cerr<<"Coverage "<<reg.hasConverged()<< " score: " << reg.getFitnessScore() << std::endl;
  cerr<<Ti<<endl;
 /* if(reg.getFitnessScore()<0.0015)
	  break;
  }*/

	//
  // Get the transformation from target to source
  targetToSource = Ti.inverse();

  //
  // Transform target back in source frame
  pcl::transformPointCloud (*cloud_tgt, *output, targetToSource);

  p->removePointCloud ("source");
  p->removePointCloud ("target");
  p->removePointCloud ("last");
  PointCloudColorHandlerCustom<PointT> cloud_tgt_h (output, 0, 255, 0);
  PointCloudColorHandlerCustom<PointT> cloud_src_h (cloud_src, 255, 0, 0);
  p->addPointCloud (output, cloud_tgt_h, "target", vp_2);
  p->addPointCloud (cloud_src, cloud_src_h, "source", vp_2);



    //add the source to the transformed target to visualize output of this step
  *stepoutput= *cloud_src;
  *stepoutput += *output;

  
  //p->addPointCloud (output,  "target", vp_2);
  //p->addPointCloud (cloud_src, "source", vp_2);
  p->addPointCloud(stepoutput,"last",vp_3);

	//PCL_INFO ("Press q to continue the registration.\n");
   if(reg.getFitnessScore()<0.0015)
	  break;
  }
  p->spin ();
  cleanAllClouds();
  p->removePointCloud ("source"); 
  p->removePointCloud ("target");

  cerr<<"Coverage "<<reg.hasConverged()<< " score: " << reg.getFitnessScore() << std::endl;
  cerr<<targetToSource<<endl;
  
  final_transform = targetToSource;
 }

 void callSACIA(std::vector<PCD, Eigen::aligned_allocator<PCD> > &data)
 {
	 time_t starttime = time(NULL); 
	 pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZRGB>);     
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZRGB>);     
	cloud1=data[0].cloud;
	cloud2=data[1].cloud;
	    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1ds (new pcl::PointCloud<pcl::PointXYZRGB>);     
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2ds (new pcl::PointCloud<pcl::PointXYZRGB>);     
        pcl::VoxelGrid<pcl::PointXYZRGB> vox_grid; 
        vox_grid.setLeafSize( VOXEL_GRID_SIZE, VOXEL_GRID_SIZE, VOXEL_GRID_SIZE ); 
        vox_grid.setInputCloud( cloud1 ); 
        vox_grid.filter( *cloud1ds ); 
		
        vox_grid.setInputCloud( cloud2 ); 
        vox_grid.filter( *cloud2ds ); 

        cout << "done. Time elapsed: " << time(NULL) - starttime << " seconds\nCalculating normals..."; 
    cout.flush();     

        //compute normals 
        PointCloudWithNormals::Ptr normals1 = getNormals( cloud1ds ); 
        PointCloudWithNormals::Ptr normals2 = getNormals( cloud2ds ); 

        cout << "done. Time elapsed: " << time(NULL) - starttime << " seconds\nComputing local features..."; 
    cout.flush(); 

        //compute local features 
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr features1 = getFeatures( cloud1ds, normals1 ); 
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr features2 = getFeatures( cloud2ds, normals2 ); 
		cerr<<"Feature1 size"<<features1->points.size();
		cerr<<"Feature2 size"<<features2->points.size();
        cout << "done. Time elapsed: " << time(NULL) - starttime << " seconds\nComputing initial alignment using SAC..."; 
    cout.flush(); 

        //Get an initial estimate for the transformation using SAC 
        //returns the transformation for cloud2 so that it is aligned with cloud1 
        pcl::SampleConsensusInitialAlignment<pcl::PointXYZRGB, pcl::PointXYZRGB, pcl::FPFHSignature33> sac_ia = align( cloud1ds, cloud2ds, features1, features2 ); 
        Eigen::Matrix4f	init_transform = sac_ia.getFinalTransformation(); 
        transformPointCloud( *cloud2, *cloud2, init_transform ); 
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr final;
		final = cloud1; 
        *final += *cloud2; 

        cout << "done. Time elapsed: " << time(NULL) - starttime << " seconds\n"; 
        cout << "Opening aligned cloud; will return when viewer window is closed."; 
    cout.flush(); 
	view(final);
 }

 void callICP(std::vector<PCD, Eigen::aligned_allocator<PCD> > &data)
 {
 
  PointCloud::Ptr result (new PointCloud), source, target;
  Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity (), pairTransform;
  
  //add first frame to output cloud
  PCD_OUTPUT oc;
  oc.cloud=data[0].cloud;
  outclouds.push_back(oc);

  for (size_t i = 1; i < data.size (); ++i)
	  {
			source = data[i-1].cloud;
			target = data[i].cloud;
			// Add visualization data
			showCloudsLeft(source, target);

			PointCloud::Ptr temp (new PointCloud);
			PCL_INFO ("Aligning %d (%d) with %d (%d).\n", i-1, source->points.size (), i, target->points.size ());
			pairAlign (source, target, temp, pairTransform, true);
			//myPairAlign (source, target, temp, pairTransform, true);

			PCL_INFO("Alignment Complete");
			std::cout<<"Alignment Done";
			//transform current pair into the global transform
			pcl::transformPointCloud (*temp, *result, GlobalTransform);

				// Pushtransformed cloud into outclouds
				PCD_OUTPUT oc;
				oc.cloud=result;
				outclouds.push_back(oc);
			//update the global transform
			GlobalTransform = pairTransform * GlobalTransform;
			//std::stringstream ss;
			//ss << i << ".pcd";
			//pcl::io::savePCDFile (ss.str (), *result, true);

	  }
  cerr<<"Registration Finished.";
  cleanAllClouds();
  for (int i=0;i<outclouds.size();i++)
  {		
	  std::string ss=i+"_thCloudBigStringForCloudIdentifier_ ";
	   //   std::stringstream ss;
		//	 ss << i << "_thCloud";
		cerr<<ss;
	  p->addPointCloud(outclouds[i].cloud,ss,vp_2);//,"_"+i,"vp_3");
  }

  cerr<<"Smoothning...";
  for (int i=0;i<outclouds.size();i++)
  {
	  smothClouds(outclouds[i].cloud);
  }
  for (int i=0;i<outclouds.size();i++)
  {		
	  std::string ss=i+"_thCloudBigStringForCloudIdentifier_ ";
	   //   std::stringstream ss;
		//	 ss << i << "_thCloud";
		cerr<<ss;
	  p->addPointCloud(outclouds[i].cloud,ss,vp_3);//,"_"+i,"vp_3");
  }

  p->spin();
 }



// void myPairAlign (const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt, PointCloud::Ptr output, Eigen::Matrix4f &final_transform, bool downsample = false)
//{
//  //
//  // Downsample for consistency and speed
//  // \note enable this for large datasets
//  PointCloud::Ptr src (new PointCloud);
//  PointCloud::Ptr tgt (new PointCloud);
//  pcl::VoxelGrid<PointT> grid;
//  if (downsample)
//  {
//    grid.setLeafSize (0.05, 0.05, 0.05);
//    grid.setInputCloud (cloud_src);
//    grid.filter (*src);
//
//    grid.setInputCloud (cloud_tgt);
//    grid.filter (*tgt);
//  }
//  else
//  {
//    src = cloud_src;
//    tgt = cloud_tgt;
//  }
//
//   /*
//  // Compute surface normals and curvature
//  PointCloudWithNormals::Ptr points_with_normals_src (new PointCloudWithNormals);
//  PointCloudWithNormals::Ptr points_with_normals_tgt (new PointCloudWithNormals);
//
//  pcl::NormalEstimation<PointT, PointNormalT> norm_est;
//  pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBA> ());
//  norm_est.setSearchMethod (tree);
//  norm_est.setKSearch (30);
//  
//  norm_est.setInputCloud (src);
//  norm_est.compute (*points_with_normals_src);
//  pcl::copyPointCloud (*src, *points_with_normals_src);
//
//  norm_est.setInputCloud (tgt);
//  norm_est.compute (*points_with_normals_tgt);
//  pcl::copyPointCloud (*tgt, *points_with_normals_tgt);*/
//
//
//  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr points_with_normals_src(new pcl::PointCloud<pcl::PointXYZRGBA>);
//  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr points_with_normals_tgt(new pcl::PointCloud<pcl::PointXYZRGBA>);	
//  pcl::SIFTKeypoint<PointT, PointT> sift;
//	//PointCloud<PointWithScale>::Ptr sifts (new PointCloud<PointWithScale>);
//	  const float min_scale = 0.0005; 
//	const int nr_octaves = 4; 
//	const int nr_scales_per_octave = 5; 
//	const float min_contrast = 1; 
//	pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBA> ());
//	sift.setSearchMethod(tree);
//	sift.setScales(min_scale, nr_octaves, nr_scales_per_octave);
//	sift.setMinimumContrast(min_contrast);
//	sift.setInputCloud(src);
//	sift.compute (*points_with_normals_src);	
//	//pcl::copyPointCloud (*src, *points_with_normals_src);
//
//	sift.setInputCloud(tgt);
//	sift.compute (*points_with_normals_tgt);
//	//pcl::copyPointCloud (*tgt, *points_with_normals_tgt);
//  //
//  // Instantiate our custom point representation (defined above) ...
//  MyPointRepresentation point_representation;
//  // ... and weight the 'curvature' dimension so that it is balanced against x, y, and z
//  float alpha[4] = {1.0, 1.0, 1.0, 1.0};
//  point_representation.setRescaleValues (alpha);
//
//  //
//  // Align
//  //pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;
//  pcl::IterativeClosestPoint<PointCloud,PointCloud> reg;
//  //reg.setTransformationEpsilon (1e-6);
//  
//  // Set the maximum distance between two correspondences (src<->tgt) to 10cm
//  // Note: adjust this based on the size of your datasets
//  
//  //reg.setMaxCorrespondenceDistance (0.1);  
//  
//  // Set the point representation
////  reg.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation));
//
//  reg.setInputCloud (points_with_normals_src);
//  reg.setInputTarget (points_with_normals_tgt);
//
//
//
//  //
//  // Run the same optimization in a loop and visualize the results
//  Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), prev, targetToSource;
//  pcl::PointCloud<pcl::PointXYZ>::Ptr reg_result (new pcl::PointCloud<pcl::PointXYZ>);// = points_with_normals_src;
//  //reg.setMaximumIterations (2);
//
//  reg.align (*reg_result);
//  points_with_normals_src = reg_result;
//  Ti = reg.getFinalTransformation () ;
//
//  //for (int i = 0; i < 30; ++i)
//  //{
//  //  PCL_INFO ("Iteration Nr. %d.\n", i);
//
//  //  // save cloud for visualization purpose
//  //  points_with_normals_src = reg_result;
//
//  //  // Estimate
//  //  reg.setInputCloud (points_with_normals_src);
//  //  reg.align (*reg_result);
//	
//		////accumulate transformation between each Iteration
//  //  Ti = reg.getFinalTransformation () * Ti;
//
//		////if the difference between this transformation and the previous one
//		////is smaller than the threshold, refine the process by reducing
//		////the maximal correspondence distance
//  //  if (fabs ((reg.getLastIncrementalTransformation () - prev).sum ()) < reg.getTransformationEpsilon ())
//  //    reg.setMaxCorrespondenceDistance (reg.getMaxCorrespondenceDistance () - 0.001);
//  //  
//  //  prev = reg.getLastIncrementalTransformation ();
//
//  //  // visualize current state
//  //  showCloudsRight(points_with_normals_tgt, points_with_normals_src);
//  //}
//
//	//
//  // Get the transformation from target to source
//  targetToSource = Ti.inverse();
//
//  //
//  // Transform target back in source frame
//  pcl::transformPointCloud (*cloud_tgt, *output, targetToSource);
//
//  p->removePointCloud ("source");
//  p->removePointCloud ("target");
//  p->removePointCloud ("last");
//
//  PointCloudColorHandlerCustom<PointT> cloud_tgt_h (output, 0, 255, 0);
//  PointCloudColorHandlerCustom<PointT> cloud_src_h (cloud_src, 255, 0, 0);
//  p->addPointCloud (output, cloud_tgt_h, "target", vp_2);
//  p->addPointCloud (cloud_src, cloud_src_h, "source", vp_2);
//
//    //add the source to the transformed target
//  *output += *cloud_src;
//  
//  //p->addPointCloud (output,  "target", vp_2);
//  //p->addPointCloud (cloud_src, "source", vp_2);
//  p->addPointCloud(output,"last",vp_3);
//
//	PCL_INFO ("Press q to continue the registration.\n");
//  p->spin ();
//
//  p->removePointCloud ("source"); 
//  p->removePointCloud ("target");
//
//
//  
//  final_transform = targetToSource;
// }
//
/* ---[ */


int main (int argc, char** argv)
{
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
  p = new pcl::visualization::PCLVisualizer (argc, argv, "Pairwise Incremental Registration example");
  /*p->createViewPort (0.0, 0, 0.5, 1.0, vp_1);
  p->createViewPort (0.5, 0, 1.0, 1.0, vp_2);
*/
  p->createViewPort (0.0, 0, 0.33, 1.0, vp_1);
  p->createViewPort (0.33, 0, 0.66, 1.0, vp_2);
  p->createViewPort (0.66, 0, 1.0, 1.0, vp_3);
  // p->createViewPort (0.0, 0, 0.5, 0.5, vp_1);
  //p->createViewPort (0.5, 0, 1.0, 0.5, vp_2);
  //p->createViewPort (0.0, 0.5, 0.5, 1.0, vp_3);
  //p->createViewPort (0.5, 0.5, 1.0, 1.0, vp_4);
  callICP(data);
 // callSACIA(data);
 
  
	
  
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