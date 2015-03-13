#include "utility.h"
#define BLOCK_WIDTH 32
#ifndef __CUDACC__  
    #define __CUDACC__
#endif

//  Subsampling from YUV444 to NV12
//	YUV 4:2:0 image with a plane of 8 bit Y samples followed 
//	by an interleaved U/V plane containing 8 bit 2x2 subsampled 
//	colour difference samples.
// 						Horizontal	Vertical
//		Y	   Sample Period	1	1
//		U (Cb) Sample Period	2	2
//		V (Cr) Sample Period	2	2

__global__ void yuv2nv(unsigned char * y_in,unsigned char * u_in,unsigned char * v_in, 
		unsigned char * buf_out, int img_width, int img_height){

		int tx = threadIdx.x;
		int ty = threadIdx.y;

		int row = blockIdx.y * BLOCK_WIDTH + ty;
		int col = blockIdx.x * BLOCK_WIDTH + tx;

		unsigned int index = row * img_width + col;
		unsigned int start_uv = img_width * img_height;

		if(row < img_height && col < img_width) {
			buf_out[index] = y_in[index];
			if(tx % 2 == 0 && ty % 2 == 0) {
				unsigned int uv_index = row/2 * img_width +  col;
				buf_out[start_uv + uv_index] = u_in[index];
				buf_out[start_uv + uv_index + 1] = v_in[index];
			}
		}
}

__host__ void yuv2nv_cpu(unsigned char * y_in,unsigned char * u_in,unsigned char * v_in, 
		unsigned char * buf_out, int img_width, int img_height){

			unsigned int index, uv_index;
			unsigned int start_uv = img_width * img_height;

			for(int i = 0; i < img_height; i++) {
				for(int j = 0; j < img_width; j++) {
					index = i*img_width + j;
					buf_out[index] = y_in[index];
					if(i % 2 == 0 && j % 2 == 0) {
						uv_index = i/2 * img_width + j;
						buf_out[start_uv + uv_index] = u_in[index];
						buf_out[start_uv + uv_index + 1] = v_in[index];
					}
				}
			}

		//if(row < img_height && col < img_width) {
		//	buf_out[index] = y_in[index];
		//	if(tx % 2 == 0 && ty % 2 == 0) {
		//		unsigned int uv_index = row/2 * img_width +  col;
		//		buf_out[start_uv + uv_index] = u_in[index];
		//		buf_out[start_uv + uv_index + 1] = v_in[index];
		//	}
		//}
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
	printf("%d", sizeof(unsigned char));
	clock_t start, end;

	unsigned char* host_img_y, *host_img_u, *host_img_v;
	unsigned char* device_img_y_in, *device_img_u_in, *device_img_v_in;
	//unsigned char* device_img_y_out, *device_img_u_out, *device_img_v_out;
	unsigned char* device_nv_buf_out;

	PPM_IMG img_in;
	YUV_IMG img_yuv;
	NV_IMG img_nv;

	cudaError_t cudaStatus;

	queryDevice();

	img_in = read_ppm("test1.ppm");
	img_yuv = rgb2yuv(img_in);

	img_nv.w = img_yuv.w;
	img_nv.h = img_yuv.h;

	// Begin of conversion and subsampling
	start = clock();

	host_img_y = img_yuv.img_y;
	host_img_u = img_yuv.img_u;
	host_img_v = img_yuv.img_v;

	myCudaCheck(cudaMalloc((void **) &device_img_y_in, img_yuv.h * img_yuv.w * sizeof(unsigned char)));
	myCudaCheck(cudaMalloc((void **) &device_img_u_in, img_yuv.h * img_yuv.w * sizeof(unsigned char)));
	myCudaCheck(cudaMalloc((void **) &device_img_v_in, img_yuv.h * img_yuv.w * sizeof(unsigned char)));

	myCudaCheck(cudaMemcpy(device_img_y_in, host_img_y, img_yuv.h * img_yuv.w * sizeof(unsigned char), cudaMemcpyHostToDevice));
	myCudaCheck(cudaMemcpy(device_img_u_in, host_img_u, img_yuv.h * img_yuv.w * sizeof(unsigned char), cudaMemcpyHostToDevice));
	myCudaCheck(cudaMemcpy(device_img_v_in, host_img_v, img_yuv.h * img_yuv.w * sizeof(unsigned char), cudaMemcpyHostToDevice));

	myCudaCheck(cudaMalloc((void **) &device_nv_buf_out, (3 * img_yuv.h * img_yuv.w * sizeof(unsigned char) - 1)/2 + 1));

	dim3 dimGrid((img_yuv.w - 1)/BLOCK_WIDTH + 1, (img_yuv.w - 1)/BLOCK_WIDTH + 1, 1);
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

	
	// Real conversion
	yuv2nv<<<dimGrid, dimBlock>>>(device_img_y_in, device_img_u_in, device_img_v_in, 
								  device_nv_buf_out, img_yuv.w, img_yuv.h);
	cudaDeviceSynchronize();
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("Subsampling kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// Real conversion
	img_nv.buf = (unsigned char *)malloc((3 * img_nv.h * img_nv.w * sizeof(unsigned char) - 1)/2 + 1);
	myCudaCheck(cudaMemcpy(img_nv.buf, device_nv_buf_out, (3 * img_yuv.h * img_yuv.w * sizeof(unsigned char) - 1)/2 + 1, cudaMemcpyDeviceToHost));
	
	end = clock();
	printf("\nTime taken is: %d seconds %d milliseconds.\n", (end - start)/(CLOCKS_PER_SEC), (end - start)*1000/(CLOCKS_PER_SEC)%1000);
	printf("\nRaw time: %ld\n", end - start);
	// End of conversion and subsampling

	write_yuv(img_nv, "nv_out.yuv");

	free(img_nv.buf);
	// Start of cpu subsampling
	img_nv.buf = (unsigned char *)malloc((3 * img_nv.h * img_nv.w * sizeof(unsigned char) - 1)/2 + 1);
	start = clock();
	yuv2nv_cpu(host_img_y, host_img_u, host_img_v, img_nv.buf, img_yuv.w, img_yuv.h);
	end = clock();
	printf("\nTime taken (CPU) is: %d seconds %d milliseconds.\n", (end - start)/(CLOCKS_PER_SEC), (end - start)*1000/(CLOCKS_PER_SEC)%1000);
	printf("\nRaw time: %ld\n", end - start);
	// End of cpu subsampling

	write_yuv(img_nv, "nv_out_cpu.yuv");

	free(img_nv.buf);
	cudaFree(device_img_y_in);
	cudaFree(device_img_u_in);
	cudaFree(device_img_v_in);

	cudaFree(device_nv_buf_out);


	free_ppm(img_in);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	getchar();

	return 0;
}

//#include "utility.h"
//#define BLOCK_WIDTH 32
//#ifndef __CUDACC__  
//    #define __CUDACC__
//#endif
//
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//
////  Subsampling from YUV444 to NV12
////	YUV 4:2:0 image with a plane of 8 bit Y samples followed 
////	by an interleaved U/V plane containing 8 bit 2x2 subsampled 
////	colour difference samples.
//// 						Horizontal	Vertical
////		Y	   Sample Period	1	1
////		U (Cb) Sample Period	2	2
////		V (Cr) Sample Period	2	2
//
//__global__ void yuv_subsample(unsigned char * y_in,unsigned char * u_in,unsigned char * v_in, 
//	unsigned char * y_out,unsigned char * u_out,unsigned char * v_out, int img_width, int img_height){
//		__shared__ unsigned char sample[BLOCK_WIDTH][BLOCK_WIDTH][2];
//
//		int tx = threadIdx.x;
//		int ty = threadIdx.y;
//
//		int row = blockIdx.y * BLOCK_WIDTH + ty;
//		int col = blockIdx.x * BLOCK_WIDTH + tx;
//
//		unsigned int index = row * img_width + col;
//
//		//@@ TODO: need a better way to dispatch the memory access!
//		if(ty % 2 == 0 && tx % 2 == 0) {
//			if(row < img_height && col < img_width) {
//				sample[ty][tx][0] = u_in[index];
//				sample[ty][tx][1] = v_in[index];
//				sample[ty+1][tx+1][0] = u_in[index];
//				sample[ty+1][tx+1][1] = v_in[index];
//				sample[ty+1][tx][0] = u_in[index];
//				sample[ty+1][tx][1] = v_in[index];
//				sample[ty][tx+1][0] = u_in[index];
//				sample[ty][tx+1][1] = v_in[index];
//			} 
//		}
//
//		__syncthreads();
//
//		if(row < img_height && col < img_width) {
//			y_out[index] = y_in[index];
//			u_out[index] = sample[ty][tx][0];
//			v_out[index] = sample[ty][tx][1];
//		}
//}
//
//__global__ void yuv2nv(unsigned char * y_in,unsigned char * u_in,unsigned char * v_in, 
//		unsigned char * buf_out, int img_width, int img_height){
//
//		int tx = threadIdx.x;
//		int ty = threadIdx.y;
//
//		int row = blockIdx.y * BLOCK_WIDTH + ty;
//		int col = blockIdx.x * BLOCK_WIDTH + tx;
//
//		unsigned int index = row * img_width + col;
//		unsigned int start_uv = img_width * img_height;
//
//		if(row < img_height && col < img_width) {
//			buf_out[index] = y_in[index];
//			if(tx % 2 == 0 && ty % 2 == 0) {
//				unsigned int uv_index = row/2 * img_width +  col;
//				buf_out[start_uv + uv_index] = u_in[index];
//				buf_out[start_uv + uv_index + 1] = v_in[index];
//			}
//		}
//}
//
////@@ CUDA kernel
//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//	int i = threadIdx.x;
//	c[i] = a[i] + b[i];
//}
//
////Convert RGB to YUV, all components in [0, 255]
//__global__ void rgb2yuv_cuda(unsigned char * img_r,unsigned char * img_g,unsigned char * img_b, 
//	unsigned char * img_y,unsigned char * img_u,unsigned char * img_v, int imgSize)
//{
//    
//	int gid = threadIdx.x+blockIdx.x*blockDim.x;
//
//    if(gid < imgSize)
//	{
//        img_y[gid] = (unsigned char)( 0.299*img_r[gid] + 0.587*img_g[gid] +  0.114*img_b[gid]);
//        img_u[gid] = (unsigned char)(-0.169*img_r[gid] - 0.331*img_g[gid] +  0.499*img_b[gid] + 128);
//        img_v[gid] = (unsigned char)( 0.499*img_r[gid] - 0.418*img_g[gid] - 0.0813*img_b[gid] + 128);
//    }
//}
//
//int main()
//{
//	printf("%d", sizeof(unsigned char));
//	clock_t start, end;
//
//	unsigned char* host_img_y, *host_img_u, *host_img_v;
//	unsigned char* device_img_y_in, *device_img_u_in, *device_img_v_in;
//	//unsigned char* device_img_y_out, *device_img_u_out, *device_img_v_out;
//	unsigned char* device_nv_buf_out;
//
//	PPM_IMG img_in;
//	YUV_IMG img_yuv;
//	NV_IMG img_nv;
//
//	const int arraySize = 5;
//	const int a[arraySize] = { 1, 2, 3, 4, 5 };
//	const int b[arraySize] = { 10, 20, 30, 40, 50 };
//	int c[arraySize] = { 0 };
//
//	// Add vectors in parallel.
//	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "addWithCuda failed!");
//		return 1;
//	}
//
//	queryDevice();
//
//	img_in = read_ppm("test1.ppm");
//	img_yuv = rgb2yuv(img_in);
//
//	img_nv.w = img_yuv.w;
//	img_nv.h = img_yuv.h;
//
//	// Begin of conversion and subsampling
//	start = clock();
//
//	host_img_y = img_yuv.img_y;
//	host_img_u = img_yuv.img_u;
//	host_img_v = img_yuv.img_v;
//
//	printf("\nTEST: %d\n", host_img_y[10240]);
//	printf("TEST: %d\n", host_img_u[10240]);
//	printf("TEST: %d\n", host_img_v[10240]);
//
//	myCudaCheck(cudaMalloc((void **) &device_img_y_in, img_yuv.h * img_yuv.w * sizeof(unsigned char)));
//	myCudaCheck(cudaMalloc((void **) &device_img_u_in, img_yuv.h * img_yuv.w * sizeof(unsigned char)));
//	myCudaCheck(cudaMalloc((void **) &device_img_v_in, img_yuv.h * img_yuv.w * sizeof(unsigned char)));
//
//	//myCudaCheck(cudaMalloc((void **) &device_img_y_out, img_yuv.h * img_yuv.w * sizeof(unsigned char)));
//	//myCudaCheck(cudaMalloc((void **) &device_img_u_out, img_yuv.h * img_yuv.w * sizeof(unsigned char)));
//	//myCudaCheck(cudaMalloc((void **) &device_img_v_out, img_yuv.h * img_yuv.w * sizeof(unsigned char)));
//
//	myCudaCheck(cudaMemcpy(device_img_y_in, host_img_y, img_yuv.h * img_yuv.w * sizeof(unsigned char), cudaMemcpyHostToDevice));
//	myCudaCheck(cudaMemcpy(device_img_u_in, host_img_u, img_yuv.h * img_yuv.w * sizeof(unsigned char), cudaMemcpyHostToDevice));
//	myCudaCheck(cudaMemcpy(device_img_v_in, host_img_v, img_yuv.h * img_yuv.w * sizeof(unsigned char), cudaMemcpyHostToDevice));
//
//	myCudaCheck(cudaMalloc((void **) &device_nv_buf_out, (3 * img_yuv.h * img_yuv.w * sizeof(unsigned char) - 1)/2 + 1));
//
//	dim3 dimGrid((img_yuv.w - 1)/BLOCK_WIDTH + 1, (img_yuv.w - 1)/BLOCK_WIDTH + 1, 1);
//	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
//
//	//// Easy demo
//	//yuv_subsample<<<dimGrid, dimBlock>>>(device_img_y_in, device_img_u_in, device_img_v_in, 
//	//							  device_img_y_out, device_img_u_out, device_img_v_out, 
//	//							  img_yuv.w, img_yuv.h);
//	//// Check for any errors launching the kernel
//	//cudaStatus = cudaGetLastError();
//	//if (cudaStatus != cudaSuccess) {
//	//	printf("Subsampling kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//	//}
//
//	cudaDeviceSynchronize();
//
//	// Real conversion
//	yuv2nv<<<dimGrid, dimBlock>>>(device_img_y_in, device_img_u_in, device_img_v_in, 
//								  device_nv_buf_out, img_yuv.w, img_yuv.h);
//
//	// Check for any errors launching the kernel
//	cudaStatus = cudaGetLastError();
//	if (cudaStatus != cudaSuccess) {
//		printf("Subsampling kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//	}
//
//	cudaDeviceSynchronize();
//
//	//// Easy demo
//	//myCudaCheck(cudaMemcpy(host_img_y, device_img_y_out, img_yuv.h * img_yuv.w * sizeof(unsigned char), cudaMemcpyDeviceToHost));
//	//myCudaCheck(cudaMemcpy(host_img_u, device_img_u_out, img_yuv.h * img_yuv.w * sizeof(unsigned char), cudaMemcpyDeviceToHost));
//	//myCudaCheck(cudaMemcpy(host_img_v, device_img_v_out, img_yuv.h * img_yuv.w * sizeof(unsigned char), cudaMemcpyDeviceToHost));
//
//	// Real conversion
//	img_nv.buf = (unsigned char *)malloc((3 * img_nv.h * img_nv.w * sizeof(unsigned char) - 1)/2 + 1);
//	myCudaCheck(cudaMemcpy(img_nv.buf, device_nv_buf_out, (3 * img_yuv.h * img_yuv.w * sizeof(unsigned char) - 1)/2 + 1, cudaMemcpyDeviceToHost));
//	
//	end = clock();
//	printf("\nTime taken is: %d seconds %d milliseconds.\n", (end - start)/(CLOCKS_PER_SEC), (end - start)*1000/(CLOCKS_PER_SEC)%1000);
//	printf("\nRaw time: %ld\n", end - start);
//	// End of conversion and subsampling
//
//
//	printf("\nTEST: %d\n", img_yuv.img_y[10240]);
//	printf("TEST: %d\n", img_yuv.img_u[10240]);
//	printf("TEST: %d\n", img_yuv.img_v[10240]);
//
//
//	//img_in = yuv2rgb(img_yuv);
//	//write_ppm(img_in, "test_out.ppm");
//
//	// Real converted image in nv12 format 
//	write_yuv(img_nv, "nv_out.yuv");
//
//	//cudaFree(device_img_y_in);
//	//cudaFree(device_img_u_in);
//	//cudaFree(device_img_v_in);
//	//cudaFree(device_img_y_out);
//	//cudaFree(device_img_u_out);
//	//cudaFree(device_img_v_out);
//
//	free_ppm(img_in);
//
//	// cudaDeviceReset must be called before exiting in order for profiling and
//	// tracing tools such as Nsight and Visual Profiler to show complete traces.
//	cudaStatus = cudaDeviceReset();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaDeviceReset failed!");
//		return 1;
//	}
//
//	getchar();
//	getchar();
//
//	return 0;
//}
//
//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//	int *dev_a = 0;
//	int *dev_b = 0;
//	int *dev_c = 0;
//	cudaError_t cudaStatus;
//
//	// Choose which GPU to run on, change this on a multi-GPU system.
//	cudaStatus = cudaSetDevice(0);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//		goto Error;
//	}
//
//	// Allocate GPU buffers for three vectors (two input, one output)    .
//	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	// Copy input vectors from host memory to GPU buffers.
//	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	// Launch a kernel on the GPU with one thread for each element.
//	addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//	// Check for any errors launching the kernel
//	cudaStatus = cudaGetLastError();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//		goto Error;
//	}
//
//	// cudaDeviceSynchronize waits for the kernel to finish, and returns
//	// any errors encountered during the launch.
//	cudaStatus = cudaDeviceSynchronize();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//		goto Error;
//	}
//
//	// Copy output vector from GPU buffer to host memory.
//	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//Error:
//	cudaFree(dev_c);
//	cudaFree(dev_a);
//	cudaFree(dev_b);
//
//	return cudaStatus;
//}
