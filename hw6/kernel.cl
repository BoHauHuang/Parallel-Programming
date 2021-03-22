__kernel void convolution(const __global float *inputImage, __global float4 *outputImage, __constant float *filter, const int w, const int h, const int hf){
        float4 sum = 0.0;
        const int gid = get_global_id(0)<<2;
        int now_x = gid%w;
	int now_y = gid/w;
        int yy, fidx, xx, k, l, wy, pos;
	float4 cal, f;
        fidx = 0;

        for (k = -hf; k <= hf; k++)
        {
		yy = now_y+k;
                if(yy >= 0 && yy < h)
                {
                        wy = yy*w;
                        for (l = -hf; l <= hf; l++)
                        {
				if(filter[fidx] == 0) ;
				else{
					xx = now_x + l;
                	                if (xx >= 0 && xx < w)
                        	        {
						pos = xx+wy;
						cal = (float4)(inputImage[pos], inputImage[pos+1], inputImage[pos+2], inputImage[pos+3]);
						//cal = inputImage[pos];

						f = filter[fidx];
						sum += cal*f;
        	                        }
				}
				fidx++;
                        }
                }
        }
	outputImage[gid>>2] = sum;
}

