const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

__kernel void image_scaling(__read_only image2d_t sourceImage,
                            __write_only image2d_t destinationImage,
                            const float widthNormalizationFactor,
                            const float heightNormalizationFactor)
{
    int2 coordinate = (int2)(get_global_id(0), get_global_id(1));

    float2 normalizedCoordinate = convert_float2(coordinate) * (float2)(widthNormalizationFactor, heightNormalizationFactor);

    float4 colour = read_imagef(sourceImage, sampler, normalizedCoordinate);

    write_imagef(destinationImage, coordinate, colour);
}
/*
#include "common_types.h"

#ifndef CL_DEVICE_LOCAL_MEM_SIZE
#error not defined CL_DEVICE_LOCAL_MEM_SIZE by complier with options -D
#endif
#ifndef SRC_TYPE
#error not defined SRC_TYPE by complier with options -D
#endif
#ifndef DST_TYPE
#error not defined DST_TYPE by complier with options -D
#endif
#ifndef INTEG_TYPE
#error not defined INTEG_TYPE by complier with options -D
#endif
#define V_TYPE 4
#define SHIFT_NUM 2
#define LOCAL_BUFFER_SIZE (CL_DEVICE_LOCAL_MEM_SIZE/sizeof(DST_TYPE))

#define _KERNEL_NAME(s,d,t) prefix_sum_col_and_transpose_##s##_##d##_##t
#define KERNEL_NAME(s,d,t) _KERNEL_NAME(s,d,t)

#define _KERNEL_NAME_INTEGRAL_BLOCK(s,d,t) integral_block_##s##_##d##_##t
#define KERNEL_NAME_INTEGRAL_BLOCK(s,d,t) _KERNEL_NAME_INTEGRAL_BLOCK(s,d,t)

#define _KERNEL_NAME_SCAN_V(s) integral_scan_v_##s
#define KERNEL_NAME_SCAN_V(s) _KERNEL_NAME_SCAN_V(s)
#define _KERNEL_NAME_COMBINE_V(s) integral_combine_v_##s
#define KERNEL_NAME_COMBINE_V(s) _KERNEL_NAME_COMBINE_V(s)
#define _KERNEL_NAME_SCAN_H(s) integral_scan_h_##s
#define KERNEL_NAME_SCAN_H(s) _KERNEL_NAME_SCAN_H(s)
#define _KERNEL_NAME_COMBINE_H(s) integral_combine_h_##s
#define KERNEL_NAME_COMBINE_H(s) _KERNEL_NAME_COMBINE_H(s)
#define _kernel_name_scan_v KERNEL_NAME_SCAN_V(DST_TYPE)
#define _kernel_name_scan_h KERNEL_NAME_SCAN_H(DST_TYPE)
#define _kernel_name_combine_v KERNEL_NAME_COMBINE_V(DST_TYPE)
#define _kernel_name_combine_h KERNEL_NAME_COMBINE_H(DST_TYPE)


#define VECTOR_SRC VECTOR(SRC_TYPE,V_TYPE)
#define VECTOR_DST VECTOR(DST_TYPE,V_TYPE)

#define VLOAD FUN_NAME(vload,V_TYPE)

#if INTEG_TYPE == INTEG_SQUARE
#define compute_src(src) src*src
#define _kernel_name_ KERNEL_NAME(SRC_TYPE,DST_TYPE,integ_square)
#define _kernel_name_integral_block KERNEL_NAME_INTEGRAL_BLOCK(SRC_TYPE,DST_TYPE,integ_square)
#elif INTEG_TYPE == INTEG_COUNT
#define compute_src(src) ((DST_TYPE)0!=src?(DST_TYPE)(1):(DST_TYPE)(0))
#define _kernel_name_ KERNEL_NAME(SRC_TYPE,DST_TYPE,integ_count)
#define _kernel_name_integral_block KERNEL_NAME_INTEGRAL_BLOCK(SRC_TYPE,DST_TYPE,integ_count)
#elif INTEG_TYPE == INTEG_DEFAULT
#define compute_src(src) src
#define _kernel_name_ KERNEL_NAME(SRC_TYPE,DST_TYPE,integ_default)
#define _kernel_name_integral_block KERNEL_NAME_INTEGRAL_BLOCK(SRC_TYPE,DST_TYPE,integ_default)
#else
#error unknow INTEG_TYPE by complier with options -D
#endif

///////////////////////////////////////////////////////////////////////////////
//! @brief  :   Calculates the integral of an image
////////////////////////////////////////////////////////////////////////////////
#define __SWAP(a,b) swap=a,a=b,b=swap;
// 4x4矩阵转置
inline void transpose( VECTOR_DST m[V_TYPE] ){
    DST_TYPE swap;
    __SWAP(m[0].s1,m[1].s0);
    __SWAP(m[0].s2,m[2].s0);
    __SWAP(m[0].s3,m[3].s0);
    __SWAP(m[1].s2,m[2].s1);
    __SWAP(m[1].s3,m[3].s1);
    __SWAP(m[2].s3,m[3].s2);
}
// 计算4x4的局部积分图
__kernel void _kernel_name_integral_block( __global SRC_TYPE *sourceImage, __global VECTOR_DST * dest, __constant integ_param* param){
    int pos_x=get_global_id(0)*V_TYPE,pos_y=get_global_id(1)*V_TYPE;
    if(pos_x>=param->width||pos_y>=param->height)return;
    int count_x=min(V_TYPE,param->width -pos_x);
    int count_y=min(V_TYPE,param->height-pos_y);
    VECTOR_DST sum;
    VECTOR_DST matrix[V_TYPE];
    //从原矩阵加载数据，并转为目标矩阵的数据向量类型(VECTOR_DST),
    //比如原矩阵是uchar，目标矩阵是float
    matrix[0]= 0<count_y ?
              count_x==V_TYPE?        VCONVERT(VECTOR_DST,)(VLOAD(0,sourceImage+(pos_y+0)*param->src_width_step+pos_x))
            :(count_x==1?(VECTOR_DST)(VCONVERT(VECTOR_DST,)(VLOAD(0,sourceImage+(pos_y+0)*param->src_width_step+param->width-V_TYPE)).w,0,0,0)
            :(count_x==2?(VECTOR_DST)(VCONVERT(VECTOR_DST,)(VLOAD(0,sourceImage+(pos_y+0)*param->src_width_step+param->width-V_TYPE)).zw,0,0)
                        :(VECTOR_DST)(VCONVERT(VECTOR_DST,)(VLOAD(0,sourceImage+(pos_y+0)*param->src_width_step+param->width-V_TYPE)).yzw,0)
            )
            ):0;
    matrix[1]= 1<count_y ?
              count_x==V_TYPE?        VCONVERT(VECTOR_DST,)(VLOAD(0,sourceImage+(pos_y+1)*param->src_width_step+pos_x))
            :(count_x==1?(VECTOR_DST)(VCONVERT(VECTOR_DST,)(VLOAD(0,sourceImage+(pos_y+1)*param->src_width_step+param->width-V_TYPE)).w,0,0,0)
            :(count_x==2?(VECTOR_DST)(VCONVERT(VECTOR_DST,)(VLOAD(0,sourceImage+(pos_y+1)*param->src_width_step+param->width-V_TYPE)).zw,0,0)
                        :(VECTOR_DST)(VCONVERT(VECTOR_DST,)(VLOAD(0,sourceImage+(pos_y+1)*param->src_width_step+param->width-V_TYPE)).yzw,0)
            )
            ):0;
    matrix[2]= 2<count_y ?
              count_x==V_TYPE?        VCONVERT(VECTOR_DST,)(VLOAD(0,sourceImage+(pos_y+2)*param->src_width_step+pos_x))
            :(count_x==1?(VECTOR_DST)(VCONVERT(VECTOR_DST,)(VLOAD(0,sourceImage+(pos_y+2)*param->src_width_step+param->width-V_TYPE)).w,0,0,0)
            :(count_x==2?(VECTOR_DST)(VCONVERT(VECTOR_DST,)(VLOAD(0,sourceImage+(pos_y+2)*param->src_width_step+param->width-V_TYPE)).zw,0,0)
                        :(VECTOR_DST)(VCONVERT(VECTOR_DST,)(VLOAD(0,sourceImage+(pos_y+2)*param->src_width_step+param->width-V_TYPE)).yzw,0)
            )
            ):0;
    matrix[3]= 3<count_y ?
              count_x==V_TYPE?        VCONVERT(VECTOR_DST,)(VLOAD(0,sourceImage+(pos_y+3)*param->src_width_step+pos_x))
            :(count_x==1?(VECTOR_DST)(VCONVERT(VECTOR_DST,)(VLOAD(0,sourceImage+(pos_y+3)*param->src_width_step+param->width-V_TYPE)).w,0,0,0)
            :(count_x==2?(VECTOR_DST)(VCONVERT(VECTOR_DST,)(VLOAD(0,sourceImage+(pos_y+3)*param->src_width_step+param->width-V_TYPE)).zw,0,0)
                        :(VECTOR_DST)(VCONVERT(VECTOR_DST,)(VLOAD(0,sourceImage+(pos_y+3)*param->src_width_step+param->width-V_TYPE)).yzw,0)
            )
            ):0;
    sum=0;
    //4x4矩阵纵向前缀和计算
    sum+=compute_src(matrix[0]),matrix[0]=sum;
    sum+=compute_src(matrix[1]),matrix[1]=sum;
    sum+=compute_src(matrix[2]),matrix[2]=sum;
    sum+=compute_src(matrix[3]),matrix[3]=sum;
    // 转置矩阵
    transpose(matrix);
    sum=0;
    //4x4矩阵横向前缀和计算
    sum+=matrix[0],matrix[0]=sum;
    sum+=matrix[1],matrix[1]=sum;
    sum+=matrix[2],matrix[2]=sum;
    sum+=matrix[3],matrix[3]=sum;
    // 第二次转置矩阵，将矩阵方向恢复正常
    transpose(matrix);
    // 计算结果将数据写到目标矩阵
    if(0<count_y)dest[((pos_y+0)*param->dst_width_step+pos_x)/V_TYPE]=matrix[0];
    if(1<count_y)dest[((pos_y+1)*param->dst_width_step+pos_x)/V_TYPE]=matrix[1];
    if(2<count_y)dest[((pos_y+2)*param->dst_width_step+pos_x)/V_TYPE]=matrix[2];
    if(3<count_y)dest[((pos_y+3)*param->dst_width_step+pos_x)/V_TYPE]=matrix[3];
}
#undef __SWAP
// 将第一个kernel计算的结果(4x4分块的局部积分图)作为输入输入矩阵(dest)
// 计算每个4x4块纵向结尾数据的前缀和，存入vert
__kernel void _kernel_name_scan_v( __global DST_TYPE * dest, __constant integ_param* param,__global DST_TYPE *vert,int vert_step){
    int gid_y=get_global_id(0);
    if(gid_y>=param->height)return;
    DST_TYPE sum=0;
    int dst_width_step=param->dst_width_step;
    for(int x=V_TYPE-1,end_x=param->width;x<end_x;x+=V_TYPE){
        sum+=dest[gid_y*dst_width_step+x];
        vert[gid_y*vert_step+(x/V_TYPE)]=sum;
    }
}

// 将上第一个kernel计算的结果(4x4分块的局部积分图)作为输入输入矩阵(dest)
// 将上第二个kernel计算的分组前缀和作为输入输入矩阵(vert)
// 对dest每个4x4块数据加上vert对应的上一组增量，结果输出到dest_out
__kernel void _kernel_name_combine_v( __global VECTOR_DST * dest, __constant integ_param* param,__global DST_TYPE *vert,int vert_step,__global VECTOR_DST * dest_out){
    int gid_x=get_global_id(0),gid_y=get_global_id(1);
    if(gid_x*V_TYPE>=param->width||gid_y>=param->height)return;
    int dest_index=(gid_y*param->dst_width_step)/V_TYPE+gid_x;
    VECTOR_DST m  = dest[dest_index];
    m += (VECTOR_DST)(gid_x>=1 ? vert[ gid_y*vert_step + gid_x-1]:0);
    dest_out [dest_index]=m;
}
// 将上一个kernel计算的结果(4x4分块的局部积分图)作为输入输入矩阵(dest)
// 计算每个4x4块横向结尾数据的前缀和，存入horiz
__kernel void _kernel_name_scan_h( __global VECTOR_DST * dest, __constant integ_param* param,__global VECTOR_DST *horiz,int horiz_step){
    int gid_x=get_global_id(0);
    if(gid_x*V_TYPE>=param->width)return;
    VECTOR_DST sum=0;
    int dst_width_step=param->dst_width_step;
    for(int y=V_TYPE-1,end_y=param->height;y<end_y;y+=V_TYPE){
        sum+=dest[y*dst_width_step/V_TYPE+gid_x];
        horiz[(y/V_TYPE)*horiz_step/V_TYPE+gid_x]=sum;
    }
}
// 将第三个kernel计算的结果作为输入输入矩阵(dest)
// 将第四个kernel计算的分组前缀和作为输入输入矩阵(vert)
// 对dest每个4x4块数据加上horiz对应的上一组增量，结果输出到dest_out
// dest_out就是最终的积分图
__kernel void _kernel_name_combine_h( __global VECTOR_DST * dest, __constant integ_param* param,__global VECTOR_DST *horiz,int horiz_step,__global VECTOR_DST * dest_out){
    int gid_x=get_global_id(0),gid_y=get_global_id(1);
    if(gid_x*V_TYPE>=param->width||gid_y>=param->height)return;
    VECTOR_DST m;
    int dest_index=(gid_y*param->dst_width_step)/V_TYPE+gid_x;
    m  = dest[dest_index];
    m += gid_y>=V_TYPE?horiz[((gid_y/V_TYPE)-1)*horiz_step/V_TYPE + gid_x  ]:(DST_TYPE)0;
    dest_out[dest_index]=m;
}
 */
