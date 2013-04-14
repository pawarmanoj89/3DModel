#include "openni_capture.h"
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <time.h>
#include<conio.h>
#include <omp.h>
#include "initial_guess.h";

//#define MAX_FRAME 32				//Moved to init_guess.h


const float VOXEL_GRID_SIZE = 0.03; 
pcl::visualization::PCLVisualizer *p;
	//its left and right viewports
	int vp_1, vp_2, vp_3, vp_4;
	
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
	 
	//filteredData.reserve(32);
	
    #pragma omp parallel for shared(data,filteredData)  
	for(int i=0;i<data.size();i++)
	{
		pcl::VoxelGrid<pcl::PointXYZRGB> vox_grid; 
	   vox_grid.setLeafSize( VOXEL_GRID_SIZE, VOXEL_GRID_SIZE, VOXEL_GRID_SIZE );
	   vox_grid.setInputCloud(data[i].cloud);
	   PointCloudPtr temp (new PointCloud);
	   vox_grid.filter(*filteredData[i].cloud);
	   	   
	   /*PCD m;
	   m.cloud=temp;
	   filteredData[i]=m;
	   filteredData.insert(it+i,m);*/
	   
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
	    cerr<<"\n Orignla Size"<<data[i].cloud->points.size();
	    cerr<<"\n Size"<<filteredData[i].cloud->points.size();
	}
}
int main (int argc, char** argv)
{
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

  std::vector<PCD, Eigen::aligned_allocator<PCD> > filteredData(data.size()) ;
  
  filterCloud(data,filteredData);
  for(int i=0;i<data.size();i++)
	  cerr<<"\n Size"<<data[i].cloud->points.size()<<"  "<<filteredData[i].cloud->points.size();
  cerr<<"\nFiltered";

  InitialGuess guess;
  guess.computeInitialGuess(filteredData);
  
  getch();

}