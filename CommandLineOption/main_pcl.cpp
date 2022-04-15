#include <pointcloudio.hpp>
#include <planedetector.h>
#include <normalestimator.h>
#include <boundaryvolumehierarchy.h>
#include <connectivitygraph.h>
#include <iostream>
#include <fstream>

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>

// #include <tbb/parallel_for.h>
// #include <tbb/blocked_range.h>
// #include <tbb/mutex.h>

#include "LineDetection3D.h"

void outlier_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src, 
					pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered, int search_num = 50)
{
	//创建滤波器对象
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
	sor.setInputCloud(cloud_src);
	sor.setMeanK(search_num);//寻找每个点的50个最近邻点
	sor.setStddevMulThresh(3.0);//一个点的最近邻距离超过全局平均距离的一个标准差以上，就会舍弃
	sor.filter(*cloud_filtered);
    std::cout << "cloud filterd: " << cloud_filtered->size() << std::endl;
}

void radius_outlier_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src, 
					pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered)
{
	//创建滤波器对象
	pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
	outrem.setInputCloud(cloud_src);
	outrem.setRadiusSearch(2);  //设置半径为0.8的范围内找临近点
    outrem.setMinNeighborsInRadius(3);  //设置查询点的邻域点集数小于5的删除
	outrem.filter (*cloud_filtered); 
    std::cout << "cloud filterd: " << cloud_filtered->size() << std::endl;
}

int main(int argc, char **argv)
{
    // if (argc < 3)
    // {
    //     std::cerr << "Usage: <input_point_cloud (XYZ format)> <output file (.txt)>" << std::endl;
    //     return -1;
    // }
    // std::string inputFileName(argv[1]);
    // std::string outputFileName(argv[2]);

    std::string inputFileName("/home/fan/code/3d/data/middel_150_1.pcd");
    std::string outputFileName("/home/fan/code/3d/data/middel_150_1");

    std::cout << "Reading the point cloud..." << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile(inputFileName, *cloud_input);
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src(new pcl::PointCloud<pcl::PointXYZ>);
	// voxel_filter(cloud_src, cloud_filtered);
	outlier_filter(cloud_input, cloud_src);

    std::vector<Point3d> points;
    int pointNumCur = cloud_src->size();
    for ( auto& pt : cloud_src->points ) {
        points.push_back(Point3d(Eigen::Vector3f(pt.x, pt.y, pt.z), Eigen::Vector3f(0, 0, 0)));
    }
    PointCloud3d *pointCloud = new PointCloud3d(points);


    // you can skip the normal estimation if you point cloud already have normals
    std::cout << "Estimating normals..." << std::endl;
    size_t normalsNeighborSize = 30;
    Octree octree(pointCloud);
    octree.partition(10, 30);
    ConnectivityGraph *connectivity = new ConnectivityGraph(pointCloud->size());
    pointCloud->connectivity(connectivity);
    
        nanoflann::PointCloud<double> pointData; 
        pointData.pts.reserve(pointCloud->size());
        for (int i = 0; i < pointCloud->size(); ++i) {
            auto pt = pointCloud->at(i).position();
            pointData.pts.push_back(nanoflann::PointCloud<double>::PtData(pt[0],pt[1],pt[2]));
        }

    NormalEstimator3d estimator(&octree, normalsNeighborSize, NormalEstimator3d::QUICK, &pointData);
    std::cout << pointCloud->size() << std::endl;

    // tbb::mutex mutex;
#pragma omp parallel for
    // tbb::parallel_for(tbb::blocked_range<size_t>(0, pointCloud->size()), [&](const tbb::blocked_range<size_t> &x) {
        // for (size_t i = x.begin(); i != x.end(); ++i)
        for (size_t i = 0; i < pointCloud->size(); i++)
        {
            if (i % 10000 == 0)
            {
                std::cout << i / float(pointCloud->size()) * 100 << "%..." << std::endl;
            }
            NormalEstimator3d::Normal normal = estimator.estimate_nanoflann(i);
            connectivity->addNode(i, normal.neighbors);
            (*pointCloud)[i].normal(normal.normal);
            (*pointCloud)[i].normalConfidence(normal.confidence);
            (*pointCloud)[i].curvature(normal.curvature);
        }
    // });


    // pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    // ne.setNumberOfThreads(10);
    // ne.setInputCloud(cloud_src);
    // pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    // ne.setSearchMethod(tree);
    // pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    // // ne.setRadiusSearch(1);
    // ne.setKSearch(50);
    // ne.compute(*normals);

    // Octree octree(pointCloud);
    // octree.partition(10, 30);
    // ConnectivityGraph *connectivity = new ConnectivityGraph(pointCloud->size());
    // pointCloud->connectivity(connectivity);
    // size_t normalsNeighborSize = 30;
    // NormalEstimator3d estimator(&octree, normalsNeighborSize, NormalEstimator3d::QUICK);
    // for (size_t i = 0; i < pointCloud->size(); i++)
    // {
    //     if (i % 10000 == 0)
    //     {
    //         std::cout << i / float(pointCloud->size()) * 100 << "%..." << std::endl;
    //     }
    //     NormalEstimator3d::Normal normal = estimator.estimate(i);
    //     connectivity->addNode(i, normal.neighbors);
    //     (*pointCloud)[i].normal(PointCloud3d::Vector(normals->points[i].normal_x, 
    //                                                  normals->points[i].normal_y, normals->points[i].normal_z));
    //     (*pointCloud)[i].normalConfidence(normals->points[i].data_c[3]);
    //     (*pointCloud)[i].curvature(normals->points[i].curvature);
    // }
            
    std::cout << "Detecting planes..." << std::endl;
    PlaneDetector detector(pointCloud);
    detector.minNormalDiff(0.5f);
    detector.maxDist(0.8f);  // 0.258819f
    detector.outlierRatio(0.75f);

    std::set<Plane*> planes = detector.detect();
    std::cout << planes.size() << std::endl;

    std::cout << "Saving results..." << std::endl;
    Geometry *geometry = pointCloud->geometry();
    for (Plane *plane : planes) 
    {
        geometry->addPlane(plane);
    }
    // many output formats are allowed. if you want to run our 'compare_plane_detector', uncomment the line below and comment the rest
    //pointCloudIO.saveGeometry(geometry, outputFileName);
    std::ofstream outputFile(outputFileName + ".txt");
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> pcl_planes;
    auto& src_pts = cloud_src->points;
    for (Plane *plane : planes)
    {
        Eigen::Vector3f v1 = plane->center() + plane->basisU() + plane->basisV();
        Eigen::Vector3f v2 = plane->center() + plane->basisU() - plane->basisV();
        Eigen::Vector3f v3 = plane->center() - plane->basisU() + plane->basisV();
        Eigen::Vector3f v4 = plane->center() - plane->basisU() - plane->basisV(); 
        outputFile << "Normal: [" << plane->normal()[0] << ", " << plane->normal()[1] << ", " << plane->normal()[2] << "]; " <<
                    "Center: [" << plane->center()[0] << ", " << plane->center()[1] << ", " << plane->center()[2] << "]; " <<
                    "Vertices: [[" << v1.x() << "," << v1.y() << "," << v1.z() << "], " <<
                                 "[" << v2.x() << "," << v2.y() << "," << v2.z() << "], " << 
                                 "[" << v3.x() << "," << v3.y() << "," << v3.z() << "], " << 
                                 "[" << v4.x() << "," << v4.y() << "," << v4.z() << "]]" << std::endl;

        pcl::PointCloud<pcl::PointXYZ>::Ptr plane_points(new pcl::PointCloud<pcl::PointXYZ>);
        plane_points->points.reserve(plane->inliers().size());
        for (auto& idx : plane->inliers()) {
            plane_points->points.emplace_back(src_pts[idx].x, src_pts[idx].y, src_pts[idx].z);
        }
        pcl_planes.push_back(plane_points);
    }

	int k = 50;
	LINEDETECTION3D::LineDetection3D line_detector;
	std::vector<LINEDETECTION3D::PLANE> planes_out;
	std::vector<std::vector<cv::Point3d> > lines;
	std::vector<double> ts;
    
	line_detector.run( pcl_planes, k, planes_out, lines, ts );


    boost::shared_ptr<pcl::visualization::PCLVisualizer> MView(new pcl::visualization::PCLVisualizer("boundary"));

    int v1(0);
    MView->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    MView->setBackgroundColor(0.3, 0.3, 0.3, v1);
    MView->addText("Raw point clouds", 10, 10, "v1_text", v1);
    MView->addPointCloud<pcl::PointXYZ>(cloud_src, "sample cloud", v1);
    MView->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "sample cloud", v1);

	double colors[][3] = {{0, 0.5, 0}, {0, 0, 0.5}, {0.5, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5},
                          {0.2, 0.5, 0.7}, {0.7, 0.5, 0.2}, {0.5, 0.2, 0.7}, {0.5, 0.7, 0.2}, {0.2, 0.7, 0.5}, {0.7, 0.2, 0.5}};
    int v2(1);
    MView->createViewPort(0.5, 0.0, 1, 1.0, v2);
    MView->setBackgroundColor(0.5, 0.5, 0.5, v2);
    MView->addText("Boudary point clouds", 80, 80, "v2_text", v2);

	for (int i = 0; i < pcl_planes.size(); ++i) {
		std::string plane_str = std::string("plane_") + std::to_string(i);
		std::string hull_str = std::string("hull_") + std::to_string(i);
		MView->addPointCloud<pcl::PointXYZ>(pcl_planes[i], plane_str, v2);
		MView->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, colors[i][0], colors[i][1], colors[i][2], plane_str, v2);
		// MView->addPointCloud<pcl::PointXYZ>(hulls[i], hull_str, v2);
		// MView->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0.2, 1, hull_str, v2);
		// MView->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, hull_str, v2);
	}

    // for (auto &plane_v : planes_out) {
    for (int i = 0; i < planes_out.size(); ++i) {
        auto &plane_v = planes_out[i];
        // float R = 1.0*(rand()%1000)/1000;
        // float G = 1.0*(rand()%1000)/1000;
        // float B = 1.0*(rand()%1000)/1000;
        float R = colors[i][0]+0.3;
        float G = colors[i][1]+0.3;
        float B = colors[i][2]+0.3;
        for (auto& lines_v : plane_v.lines3d) {
            for (auto& line_v : lines_v) {
                if (line_v.size() != 2) {
                    cout << "warning: line size:" << line_v.size() << endl;
                }
                auto& p1 = line_v[0];
                auto& p2 = line_v[1];
                double dist = sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y) + (p1.z - p2.z)*(p1.z - p2.z));
                cout << "dist: " << dist << endl;
                string rand_str = to_string(rand());
                MView->addLine(pcl::PointXYZ(p1.x,p1.y,p1.z), pcl::PointXYZ(p2.x,p2.y,p2.z), R, G, B, rand_str, v2);
                MView->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, rand_str, v2);
                // break;
            }
            // break;
        }
    }

    MView->spin();

    delete pointCloud;
    return 0;
}