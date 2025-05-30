#include <CL/opencl.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;
using namespace std;
using namespace cv;
using namespace std::chrono;

string loadkernel(const string& filepath) 
{
    ifstream file(filepath);
    
    if (!file.is_open()) 
    {
        cerr << "error: could not open kernel file!" << endl;
        exit(1);
    }
    
    string loadedkernel = string((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    
    return loadedkernel;
}

int main() 
{
    string datasetpath = "/home/anas/Documents/Parallel and Distributed Computing/Assignment 3/Solution/q1/DataSet";
    string outputfolder = "/home/anas/Documents/Parallel and Distributed Computing/Assignment 3/Solution/q1/OutputOpenCL/";
    
    fs::create_directory(outputfolder);

    string kernelsource = loadkernel("convolution.cl");
    cl::Program::Sources sourcecode;
    sourcecode.push_back({kernelsource.c_str(), kernelsource.length()});

    vector<cl::Platform> allplatforms;
    cl::Platform::get(&allplatforms);
    cl::Platform myplatform = allplatforms[0];

    vector<cl::Device> devices;
    myplatform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue queue(context, device);
    cl::Program program(context, sourcecode);

    if (program.build({device}) != CL_SUCCESS) 
    {
        cerr << "error: failed to compile OpenCL program!" << endl;
        cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
        
        return -1;
    }

    vector<float> sobelfilter = {1, 0, -1, 1, 0, -1, 1, 0, -1};
    int filtersize = 3;

    for (const auto& file : fs::directory_iterator(datasetpath)) 
    {
        if (file.is_regular_file()) 
        {
            string inputpath = file.path().string();
            
            string outputpath = outputfolder + "Processed_" + file.path().stem().string() + ".jpg";

            Mat grayscaleimage = imread(inputpath, IMREAD_GRAYSCALE);
            
            if (grayscaleimage.empty()) 
            {
                cerr << "error: failed to load image " << inputpath << endl;
                continue;
            }

            int imagewidth = grayscaleimage.cols;
            int imageheight = grayscaleimage.rows;
            int imagesize = imagewidth * imageheight;
            
            vector<float> inputimage(imagesize);
            for (int y = 0; y < imageheight; y++) 
            {
                for (int x = 0; x < imagewidth; x++) 
                {
                    inputimage[y * imagewidth + x] = static_cast<float>(grayscaleimage.at<uchar>(y, x));
                }
            }

            vector<float> outputimage(imagesize, 0);
            
            cl::Buffer bufferinput(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * imagesize, inputimage.data());
            cl::Buffer bufferoutput(context, CL_MEM_WRITE_ONLY, sizeof(float) * imagesize);
            cl::Buffer bufferfilter(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * sobelfilter.size(), sobelfilter.data());
            
            cl::Kernel convolutionkernel(program, "convolution");
            convolutionkernel.setArg(0, bufferinput);
            convolutionkernel.setArg(1, bufferoutput);
            convolutionkernel.setArg(2, bufferfilter);
            convolutionkernel.setArg(3, imagewidth);
            convolutionkernel.setArg(4, imageheight);
            convolutionkernel.setArg(5, filtersize);

            cl::NDRange globaldimension(imagewidth, imageheight);
            
            auto starttime = high_resolution_clock::now();
            
            queue.enqueueNDRangeKernel(convolutionkernel, cl::NullRange, globaldimension, cl::NullRange);
            queue.finish();
            
            auto endtime = high_resolution_clock::now();
            
            duration<double> totaltime = endtime - starttime;
            cout << "Processed " << inputpath << " in " << totaltime.count() << " seconds." << endl;

            queue.enqueueReadBuffer(bufferoutput, CL_TRUE, 0, sizeof(float) * imagesize, outputimage.data());

            float maximum = *max_element(outputimage.begin(), outputimage.end());
            float minimum = *min_element(outputimage.begin(), outputimage.end());
            
            if (maximum > minimum) 
            {
                for (int i = 0; i < imagesize; i++) 
                {
                    outputimage[i] = ((outputimage[i] - minimum) / (maximum - minimum)) * 255.0f;
                }
            }

            Mat finalresult(imageheight, imagewidth, CV_8U);
            
            for (int y = 0; y < imageheight; y++) 
            {
                for (int x = 0; x < imagewidth; x++) 
                {
                    finalresult.at<uchar>(y, x) = static_cast<uchar>(outputimage[y * imagewidth + x]);
                }
            }

            imwrite(outputpath, finalresult);
            cout << "Saved: " << outputpath << endl;
        }
    }

    return 0;
}

