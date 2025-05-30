__kernel void convolution(__global const float* input, __global float* output, __constant float* filter, int width, int height, int filtersize) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int halffilter = filtersize / 2;

    if (x >= width || y >= height)
    {
    	return;
    }

    float4 sum = (float4)(0.0f);

    for (int fy = -halffilter; fy <= halffilter; fy++) 
    {
        for (int fx = -halffilter; fx <= halffilter; fx++) 
        {
            int nx = x + fx;
            int ny = y + fy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) 
            {
                int imageindex = ny * width + nx;
                int filterindex = (fy + halffilter) * filtersize + (fx + halffilter);
                
                float4 pixel = vload4(0, &input[imageindex]);
                sum += pixel * (float4)(filter[filterindex]);
            }
        }
    }

    float totalsum = sum.x + sum.y + sum.z + sum.w;
    totalsum = fabs(totalsum);
    output[y * width + x] = totalsum;
} 

