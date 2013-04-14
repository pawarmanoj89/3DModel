#ifndef OPENNI_CAPTURE_H
#define OPENNI_CAPTURE_H

//#include "typedefs.h"

#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/pcl_visualizer.h>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

typedef pcl::PointCloud<PointT>::Ptr PointCloudPtr;
typedef pcl::PointCloud<PointT>::ConstPtr PointCloudConstPtr;
/* A simple class for capturing data from an OpenNI camera */
class OpenNICapture
{
  public:
    OpenNICapture (const std::string& device_id = "");
    ~OpenNICapture ();
    
    void setTriggerMode (bool use_trigger);
    const PointCloudPtr snap ();
    const PointCloudPtr snapAndSave (const std::string & filename);
	//const PointCloudPtr captureNthFrame ();

  protected:
    void onNewFrame (const PointCloudConstPtr &cloud);
    void onKeyboardEvent (const pcl::visualization::KeyboardEvent & event);

    void waitForTrigger ();

    pcl::OpenNIGrabber grabber_;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> preview_;
    int frame_counter_;
    PointCloudPtr most_recent_frame_;
    bool use_trigger_, trigger_;
    boost::mutex mutex_;
};

#endif
