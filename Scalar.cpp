#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std;
using namespace cv;
using namespace chrono;
namespace fs = std::filesystem;

const int kerneldimensions = 3;

const float kernel[kerneldimensions][kerneldimensions] = 
{
    { 1,  0, -1 },
    { 1,  0, -1 },
    { 1,  0, -1 }
};

Mat processimage(const Mat& input)
{
    int height = input.rows;
    int width = input.cols;
    Mat output(height, width, CV_32F, Scalar(0));
    
    Mat floatinput;
    input.convertTo(floatinput, CV_32F);
    
    int pad = kerneldimensions / 2;
    Mat paddedinput;
    copyMakeBorder(floatinput, paddedinput, pad, pad, pad, pad, BORDER_REPLICATE);
    
    for (int y = 0; y < height; y++) 
    {
        for (int x = 0; x < width; x++) 
        {
            float sum = 0.0;
            
            for (int ky = 0; ky < kerneldimensions; ky++) 
            {
                for (int kx = 0; kx < kerneldimensions; kx++) 
                {
                    sum += paddedinput.at<float>(y + ky, x + kx) * kernel[ky][kx];
                }
            }
            
            output.at<float>(y, x) = sum;
        }
    }
    
    output = abs(output);
    Mat finalnormalizedoutput;
    normalize(output, finalnormalizedoutput, 0, 255, NORM_MINMAX);
    finalnormalizedoutput.convertTo(finalnormalizedoutput, CV_8U);
    
    return finalnormalizedoutput;
}

int main()
{
    string datasetpath = "/home/anas/Documents/Parallel and Distributed Computing/Assignment 3/Solution/q1/DataSet";
    string outputfolder = "/home/anas/Documents/Parallel and Distributed Computing/Assignment 3/Solution/q1//Output/";
    
    fs::create_directories(outputfolder);
    
    int processedimages = 0;
    
    for (const auto& file : fs::directory_iterator(datasetpath)) 
    {
        if (file.is_regular_file()) 
        {
            string inputpath = file.path().string();
            string filename = file.path().filename().string();
            string outputname = outputfolder + "Processed_" + filename;
            
            Mat image = imread(inputpath, IMREAD_GRAYSCALE);
            
            if (image.empty()) 
            {
                cerr << "Failed to open: " << inputpath << endl;
                continue;
            }
            
            auto starttime = high_resolution_clock::now();
            Mat processedimage = processimage(image);
            auto endtime = high_resolution_clock::now();
            duration<double> totaltime = endtime - starttime;
            
            cout << "Time taken: " << totaltime.count() << " seconds" << endl;
            
            imwrite(outputname, processedimage);
            cout << "Saved: " << outputname << endl;
            processedimages++;
        }
    }
    
    cout << "\nTotal images processed: " << processedimages << endl;
    
    return 0;
}

