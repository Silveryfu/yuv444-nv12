#include "utility.h"
#define BLOCK_SIZE 512

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

//  Subsampling from YUV444 to NV12
//	YUV 4:2:0 image with a plane of 8 bit Y samples followed 
//	by an interleaved U/V plane containing 8 bit 2x2 subsampled 
//	colour difference samples.
// 						Horizontal	Vertical
//		Y	   Sample Period	1	1
//		U (Cb) Sample Period	2	2
//		V (Cr) Sample Period	2	2

__global__ void yuv2nv(unsigned char * y_in,unsigned char * u_in,unsigned char * v_in, 
	unsigned char * y_out,unsigned char * u_out,unsigned char * v_out, int imgSize){
		
		


}

//@@ CUDA kernel
__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

//Convert RGB to YUV, all components in [0, 255]
__global__ void rgb2yuv_cuda(unsigned char * img_r,unsigned char * img_g,unsigned char * img_b, 
	unsigned char * img_y,unsigned char * img_u,unsigned char * img_v, int imgSize)
{
    
	int gid = threadIdx.x+blockIdx.x*blockDim.x;

    if(gid < imgSize)
	{
        img_y[gid] = (unsigned char)( 0.299*img_r[gid] + 0.587*img_g[gid] +  0.114*img_b[gid]);
        img_u[gid] = (unsigned char)(-0.169*img_r[gid] - 0.331*img_g[gid] +  0.499*img_b[gid] + 128);
        img_v[gid] = (unsigned char)( 0.499*img_r[gid] - 0.418*img_g[gid] - 0.0813*img_b[gid] + 128);
    }
}

int main()
{
	clock_t start, end;

	unsigned char* host_img_y, *host_img_u, *host_img_v;
	unsigned char* device_img_y_in, *device_img_u_in, *device_img_v_in;
	unsigned char* device_img_y_out, *device_img_u_out, *device_img_v_out;

	PPM_IMG img_in;
	YUV_IMG img_yuv;

	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	queryDevice();

	img_in = read_ppm("test1.ppm");
	img_yuv = rgb2yuv(img_in);

	// Begin of conversion and subsampling
	start = clock();

	host_img_y = img_yuv.img_y;
	host_img_u = img_yuv.img_u;
	host_img_v = img_yuv.img_v;


	cudaMalloc((void **) &device_img_y_in, img_yuv.h * img_yuv.w * sizeof(unsigned char));
	cudaMalloc((void **) &device_img_u_in, img_yuv.h * img_yuv.w * sizeof(unsigned char));
	cudaMalloc((void **) &device_img_v_in, img_yuv.h * img_yuv.w * sizeof(unsigned char));

	cudaMalloc((void **) &device_img_y_out, img_yuv.h * img_yuv.w * sizeof(unsigned char));
	cudaMalloc((void **) &device_img_u_out, img_yuv.h * img_yuv.w * sizeof(unsigned char));
	cudaMalloc((void **) &device_img_v_out, img_yuv.h * img_yuv.w * sizeof(unsigned char));

	cudaMemcpy(device_img_y_in, host_img_y, img_yuv.h * img_yuv.w * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(device_img_u_in, host_img_u, img_yuv.h * img_yuv.w * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(device_img_v_in, host_img_v, img_yuv.h * img_yuv.w * sizeof(unsigned char), cudaMemcpyHostToDevice);

	dim3 dimGrid((img_yuv.w - 1)/BLOCK_SIZE + 1, (img_yuv.h - 1)/BLOCK_SIZE + 1, 1);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	//cudaMalloc();

	cudaDeviceSynchronize();

	cudaMemcpy(host_img_y, device_img_y_out, img_yuv.h * img_yuv.w * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_img_u, device_img_u_out, img_yuv.h * img_yuv.w * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_img_v, device_img_v_out, img_yuv.h * img_yuv.w * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	end = clock();
	printf("\nTime taken is: %d seconds %d milliseconds.\n", (end - start)/(CLOCKS_PER_SEC), (end - start)*1000/(CLOCKS_PER_SEC)%1000);
	// End of conversion and subsampling

	img_in = yuv2rgb(img_yuv);
	write_ppm(img_in, "test_out.ppm");

	cudaFree(device_img_y_in);
	cudaFree(device_img_u_in);
	cudaFree(device_img_v_in);
	cudaFree(device_img_y_out);
	cudaFree(device_img_u_out);
	cudaFree(device_img_v_out);

	free_ppm(img_in);


	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	getchar();
	getchar();

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
