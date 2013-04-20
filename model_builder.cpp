#include "model_builder.h"
#include "openni_capture.h"
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/transforms.h>
#include <time.h>
#include<conio.h>
#include <omp.h>
#include "initial_guess.h";
#include "pairwise_registration.h"

//#define MAX_FRAME 32				//Moved to init_guess.h


const float VOXEL_GRID_SIZE = 0.05; 
pcl::visualization::PCLVisualizer *p3;
//	//its left and right viewports
//	int vp_1, vp_2, vp_3, vp_4;
	
//convenient structure to handle our pointclouds
//struct PCD
//{
//  PointCloud::Ptr cloud;
//  std::string f_name;
//
//  PCD() : cloud (new PointCloud) {};
//};

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

void filterCloud(std::vector<PCD, Eigen::aligned_allocator<PCD> > data,  std::vector<PCD,Eigen::aligned_allocator<PCD>> &filteredData )
{
	 
	#pragma omp parallel for shared(data,filteredData)  
	for(int i=0;i<data.size();i++)
	{
		pcl::VoxelGrid<pcl::PointXYZRGB> vox_grid; 
	   vox_grid.setLeafSize( VOXEL_GRID_SIZE, VOXEL_GRID_SIZE, VOXEL_GRID_SIZE );
	   vox_grid.setInputCloud(data[i].cloud);
	   PointCloudPtr temp (new PointCloud);
	   vox_grid.filter(*filteredData[i].cloud);
	   	   
	     
	}
}

void filterCloud2(std::vector<PCD, Eigen::aligned_allocator<PCD> > data, struct PCD *filteredData )
{
	 
    #pragma omp parallel for shared(data,filteredData) 
	for(int i=0;i<data.size();i++)
	{
		pcl::VoxelGrid<pcl::PointXYZRGB> vox_grid; 
		vox_grid.setLeafSize( VOXEL_GRID_SIZE, VOXEL_GRID_SIZE, VOXEL_GRID_SIZE ); 
	   vox_grid.setInputCloud(data[i].cloud);
	   vox_grid.filter(*filteredData[i].cloud);
	}
}

 //void cloudTransformation(Eigen::Matrix4f TransMatrix[],std::vector<PCD, Eigen::aligned_allocator<PCD> > &data)
 //{
	// PCD_OUTPUT poc;
	// poc.cloud=data[0].cloud;
	// outcloudsparallel.push_back(poc);
	// PointCloud::Ptr result (new PointCloud ); 
	//
	// for(int i=1;i<data.size();i++)
	// {
	//
	//			cerr<<"\n_____ "<<i;
	//			pcl::transformPointCloud (*data[i].cloud, *result, TransMatrix[i]);
	//			cerr<<"\n^^^^ "<<i;
	//			// Pushtransformed cloud into outclouds
	//			PCD_OUTPUT poc;
	//			poc.cloud=result;
	//			outcloudsparallel.push_back(poc);
	// }
 //
	// for (int i=0;i<outcloudsparallel.size();i++)
 // {		
	//  std::string ss=i+"zxcvbnmasdfghjklqwertyuiop";
	//  	cerr<<ss;	
	//  p->addPointCloud(outcloudsparallel[i].cloud,ss,vp_2);
 // }
 //}

int main (int argc, char** argv)
{
	std::vector<PCD, Eigen::aligned_allocator<PCD> > data;
	//Eigen::Matrix4f *guessTransformation[MAX_FRAME],*finalTransformation[MAX_FRAME];
	Eigen::Matrix4f *guessTransformation=new Eigen::Matrix4f[MAX_FRAME];
	Eigen::Matrix4f *finalTransformation=new Eigen::Matrix4f[MAX_FRAME];
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

  for(int i=0;i<data.size();i++)
  {
	  
	  cerr<<"\n"<<data[i].cloud->points.size();
	  	
  }
  for(int i=0;i<data.size();i++)
  {   std::stringstream s("file");
		s<<i;
		s<<".pcd";
		pcl::io::savePCDFileBinary<pcl::PointXYZRGB>(s.str(),*data[i].cloud);
		cerr<<"\n"<<s.str()<<"saved";
  }
  std::vector<PCD, Eigen::aligned_allocator<PCD> > filteredData(data.size()) ;
  std::vector<PCD, Eigen::aligned_allocator<PCD> > outputData(data.size()) ;
  
  filterCloud(data,filteredData);
  for(int i=0;i<data.size();i++)
	  cerr<<"\n Size"<<data[i].cloud->points.size()<<"  "<<filteredData[i].cloud->points.size();
  cerr<<"\nFiltered";
  
  //InitialGuess guess;
  //guess.computeInitialGuess(filteredData,guessTransformation);
  //for(int i=0;i<data.size()-1;i++)
	 // cerr<<"\n guess"<<guessTransformation[i];
  //cerr<<"\nGuessDone";
  PairwiseRegistration pairRegister;
  pairRegister.registerCloud(filteredData,guessTransformation,finalTransformation);
  for(int i=0;i<data.size();i++)
	  cerr<<"\n finalTransformation"<<finalTransformation[i];
  //Eigen::Matrix4f a=Eigen::Matrix4f::Identity();
  //Eigen::Matrix4f b=Eigen::Matrix4f::Random();
  //cerr<<"\n b"<<b;
  PointCloud::Ptr final (new PointCloud);
  PointCloud::Ptr input (new PointCloud);
  PointCloud::Ptr outCloud (new PointCloud);

  *input=*data[0].cloud;
  *final=*data[0].cloud;
   int view1,view2,view3;
  p3 = new pcl::visualization::PCLVisualizer("3DViewer3");
  p3->createViewPort (0.0, 0, 0.33, 1.0, view1);
  p3->createViewPort (0.33, 0, 0.66, 1.0, view2);
  p3->createViewPort (0.66, 0, 1.0, 1.0, view3);
   //p3->addPointCloud(data[0].cloud,"in1",view2);
  for(int i=1;i<data.size();i++)
  {
	  *input += *data[i].cloud;
  }

  p3->addPointCloud(input,"inpt",view1);
 /* for(int i=0;i<data.size();i=i+7)
	{
		std::stringstream s("inclds");
		s<<i;
		p3->addPointCloud(data[i].cloud,s.str(),view2);
	}*/
  

  cerr<<"\nParr trans";
  //#pragma omp parallel for  
    for(int i=0;i<data.size();i++)
  {
	  
	  pcl::transformPointCloud (*data[i].cloud, *outputData[i].cloud, finalTransformation[i]);
	  cerr<<"\n"<<data[i].cloud->points.size()<<" "<<outputData[i].cloud->points.size()<<"\n";
	  cerr<<finalTransformation[i];
	  std::stringstream s1("orig");
	  std::stringstream s2("trans");
	  s1<<i;
	  s2<<i;
	  p3->addPointCloud(data[i].cloud,s1.str(),view2);
	  p3->addPointCloud(outputData[i].cloud,s2.str(),view3);
	  p3->spin();
  }
	p3->removeAllPointClouds(view2);
	p3->removeAllPointClouds(view3);
	p3->spin();
	for(int i=0;i<data.size();i=i+7)
	{
		std::stringstream s("clds");
		s<<i;
		cerr<<" CLoud "<<i;
		p3->addPointCloud(outputData[i].cloud,s.str(),view3);
		p3->spin();
	}
	
	*outCloud= *outputData[0].cloud;
	
     for(int i=1;i<data.size();i++)
  {
	  *outCloud += *outputData[i].cloud;
  }
	 cerr<<"\nVisual..";
	// p3->addPointCloud(outCloud,"outCloud",view2); 
	 cerr<<"done";
	p3->spin();
	p3->removeAllPointClouds(view1);
	p3->removeAllPointClouds(view2);
	p3->addPointCloud(data[0].cloud,"cld1",view1);
	p3->addPointCloud(data[data.size()-1].cloud,"cldlst",view2);
	p3->spin();

	 //cerr<<"\nWan to save PLY(y/n) ";
  //  char reply;
	 //std::cin>>reply;
	 //if(reply='y')
		//   pcl::io::savePLYFileASCII<pcl::PointXYZRGB>("result.ply",*final);
  
  
  getch();

}