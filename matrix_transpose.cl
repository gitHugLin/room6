#define SRC_TYPE unsigned char
#define _KERNEL_NAME(s) matrix_transpose_##s
#define KERNEL_NAME(s) _KERNEL_NAME(s)

//kernel function的名字在编译期根据SRC_TYPE 加一个类型后缀

__kernel void KERNEL_NAME(SRC_TYPE)( __global SRC_TYPE *matrix_src,__global SRC_TYPE *matrix_dst,
            int width, int src_width_step,int dst_width_step ) {
    const int y = get_global_id(1);
    __global SRC_TYPE * src_ptr = matrix_src + y*src_width_step;

    for( int x = 0; x < width; ++x,++src_ptr ) {
        matrix_dst[x*dst_width_step + y] = *src_ptr;
    }
}
