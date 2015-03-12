#ifndef UTILITY_H
#define UTILITY_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;    

typedef struct{
    int w;
    int h;
    unsigned char * img_r;
    unsigned char * img_g;
    unsigned char * img_b;
} PPM_IMG;

typedef struct{
    int w;
    int h;
    unsigned char * img_y;
    unsigned char * img_u;
    unsigned char * img_v;
} YUV_IMG;

PPM_IMG read_ppm(const char * path);
void write_ppm(PPM_IMG img, const char * path);
void free_ppm(PPM_IMG img);

PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
void free_pgm(PGM_IMG img);

YUV_IMG rgb2yuv(PPM_IMG img_in);
PPM_IMG yuv2rgb(YUV_IMG img_in);

unsigned char clip_rgb(int x);

__host__ void queryDevice();

#endif