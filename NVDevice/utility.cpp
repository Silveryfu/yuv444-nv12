#include "utility.h"

YUV_IMG rgb2yuv(PPM_IMG img_in) {
	YUV_IMG img_out;

    int i;//, j;
    unsigned char r, g, b;
    unsigned char y, cb, cr;
    
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_y = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_u = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_v = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);

    for(i = 0; i < img_out.w*img_out.h; i ++){
        r = img_in.img_r[i];
        g = img_in.img_g[i];
        b = img_in.img_b[i];
        
        y  = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
        cb = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
        cr = (unsigned char)( 0.499*r - 0.418*g - 0.0813*b + 128);
        
        img_out.img_y[i] = y;
        img_out.img_u[i] = cb;
        img_out.img_v[i] = cr;
    }
    
    return img_out;
}

PPM_IMG yuv2rgb(YUV_IMG img_in)
{
    PPM_IMG img_out;
    int i;
    int  rt,gt,bt;
    int y, cb, cr;
    
    
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_r = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_g = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_b = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);

    for(i = 0; i < img_out.w*img_out.h; i ++){
        y  = (int)img_in.img_y[i];
        cb = (int)img_in.img_u[i] - 128;
        cr = (int)img_in.img_v[i] - 128;
        
        rt  = (int)( y + 1.402*cr);
        gt  = (int)( y - 0.344*cb - 0.714*cr);
        bt  = (int)( y + 1.772*cb);

        img_out.img_r[i] = clip_rgb(rt);
        img_out.img_g[i] = clip_rgb(gt);
        img_out.img_b[i] = clip_rgb(bt);
    }
    
    return img_out;
}

__host__ PPM_IMG read_ppm(const char * path) {
	FILE * in_file;
	char sbuf[256];

	char *ibuf;
	PPM_IMG result;

	int v_max, i;

	in_file = fopen(path, "r");
	if (in_file == NULL) {
		printf("Input file not found!\n");
		exit(1);
	}

	printf("\nReading PPM.\n");
	
	/* Skip the magic number */
	fscanf(in_file, "%s", sbuf);
	fscanf(in_file, "%s", sbuf);
	fscanf(in_file, "%s", sbuf);
	fscanf(in_file, "%s", sbuf);
	fscanf(in_file, "%s", sbuf);
	fscanf(in_file, "%s", sbuf);
	fscanf(in_file, "%s", sbuf);
	fscanf(in_file, "%s", sbuf);
	fscanf(in_file, "%s", sbuf);

	printf("%s", sbuf);

	fscanf(in_file, "%d", &result.w);
	fscanf(in_file, "%d", &result.h);
	fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);

	result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    ibuf         = (char *)malloc(3 * result.w * result.h * sizeof(char));

	/* Read into the input buffer for rgb disperse */
	fread(ibuf,sizeof(unsigned char), 3 * result.w*result.h, in_file);

	for(i = 0; i < result.w*result.h; ++i) {
		result.img_r[i] = ibuf[3*i + 0];
		result.img_g[i] = ibuf[3*i + 1];
		result.img_b[i] = ibuf[3*i + 2];
	}
	
	fclose(in_file);
	free(ibuf);

	return result;
}

__host__ void write_ppm(PPM_IMG img, const char * path){
    FILE * out_file;
    int i;
    
    char * obuf = (char *)malloc(3 * img.w * img.h * sizeof(char));

    for(i = 0; i < img.w*img.h; i ++){
        obuf[3*i + 0] = img.img_r[i];
        obuf[3*i + 1] = img.img_g[i];
        obuf[3*i + 2] = img.img_b[i];
    }
    out_file = fopen(path, "wb");

	printf("\nWriting PPM.\n");
    fprintf(out_file, "P6\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(obuf,sizeof(unsigned char), 3*img.w*img.h, out_file);
    fclose(out_file);
    free(obuf);
}

__host__ void write_yuv(NV_IMG img, const char * path){
    FILE * out_file;
    
    out_file = fopen(path, "wb");

	printf("\nWriting .YUV.\n");
    fwrite(img.buf, sizeof(char), (3*img.w*img.h - 1)/2 + 1, out_file);
    fclose(out_file);
}

__host__ void free_ppm(PPM_IMG img)
{
    free(img.img_r);
    free(img.img_g);
    free(img.img_b);
}

__host__ bool myCudaCheck(cudaError_t stat) {
	if (stat != cudaSuccess) {
		printf("Cuda Error");
		return 1;
	}
	return 0;
}

__host__ void queryDevice() {
	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i=0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
	}
}

unsigned char clip_rgb(int x)
{
    if(x > 255)
        return 255;
    if(x < 0)
        return 0;

    return (unsigned char)x;
}