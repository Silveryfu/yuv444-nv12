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
}

//@@ Better kernel alleviates thread diversion
__global__ void yuv2nv_impv(unsigned char * y_in,unsigned char * u_in,unsigned char * v_in, 
		unsigned char * buf_out, int img_width, int img_height){

		int tx = threadIdx.x;
		int ty = threadIdx.y;

		int row = blockIdx.y * BLOCK_WIDTH + ty;
		int col = blockIdx.x * BLOCK_WIDTH + tx;

		unsigned int index = row * img_width + col;
		unsigned int start_uv = img_width * img_height;

		if(row < img_height && col < img_width) {
			buf_out[index] = y_in[index];
			if(row <= (img_height/2) && col <= (img_width/2)) {
				unsigned int uv_index = row * 2 * img_width + col * 2;
				unsigned int nv_index = row * img_width + col * 2;
				buf_out[start_uv + nv_index] = u_in[uv_index];
				buf_out[start_uv + nv_index + 1] = v_in[uv_index];
			}
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

	start = clock();
	// Real conversion
	yuv2nv<<<dimGrid, dimBlock>>>(device_img_y_in, device_img_u_in, device_img_v_in, 
								  device_nv_buf_out, img_yuv.w, img_yuv.h);
	//yuv2nv_impv<<<dimGrid, dimBlock>>>(device_img_y_in, device_img_u_in, device_img_v_in, 
	//							  device_nv_buf_out, img_yuv.w, img_yuv.h);
	cudaDeviceSynchronize();
	end = clock();

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("Subsampling kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// Real conversion
	img_nv.buf = (unsigned char *)malloc((3 * img_nv.h * img_nv.w * sizeof(unsigned char) - 1)/2 + 1);
	myCudaCheck(cudaMemcpy(img_nv.buf, device_nv_buf_out, (3 * img_yuv.h * img_yuv.w * sizeof(unsigned char) - 1)/2 + 1, cudaMemcpyDeviceToHost));
	
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
