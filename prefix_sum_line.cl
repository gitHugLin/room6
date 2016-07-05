#ifndef CL_DEVICE_LOCAL_MEM_SIZE //local memory的大小，由编译器提供
#error not defined CL_DEVICE_LOCAL_MEM_SIZE by complier with options -D
#endif

#define SRC_TYPE unsigned char
#define DST_TYPE unsigned int
#define LOCAL_BUFFER_SIZE (CL_DEVICE_LOCAL_MEM_SIZE/sizeof(DST_TYPE))   //编译时确定local buffer数组的大小
#define _KERNEL_NAME(s,d) prefix_sum_line_##s##_##d
#define KERNEL_NAME(s,d) _KERNEL_NAME(s,d)
//kernel function的名字在编译期根据SRC_TYPE 和DST_TYPE添加类型后缀

__kernel void KERNEL_NAME(SRC_TYPE,DST_TYPE)( __global SRC_TYPE *sourceImage,
                __global DST_TYPE * dest, int width, int width_step,int is_square )
{
    __local DST_TYPE local_block[ LOCAL_BUFFER_SIZE ];
    const int line_index = get_global_id(1)*width_step;// 计算当前行的起始位置
    __global SRC_TYPE * const src_ptr   = line_index + sourceImage;  // 源矩阵的起始指针
    __global DST_TYPE * const dst_ptr   = line_index + dest;         // 目标矩阵的起始指针
    __global SRC_TYPE * block_src_ptr   = src_ptr;
    __global DST_TYPE * block_dst_ptr   = dst_ptr;
    int block_size = 0;         // 块大小
    DST_TYPE last_sum=0;        // 上一块数组的前缀和
    //将一行数据按local_block数组的大小来分块处理
    for(int start_x = 0 ; start_x < width ;
            start_x         += LOCAL_BUFFER_SIZE,
            block_src_ptr   += LOCAL_BUFFER_SIZE,
            block_dst_ptr   += LOCAL_BUFFER_SIZE,
            last_sum        += local_block[block_size -1]) {
        block_size = min((int)LOCAL_BUFFER_SIZE, width - start_x);
        //compute prefix sum of a block with local memory
        if(is_square) {
            local_block[0] = last_sum + ((DST_TYPE)block_src_ptr[0])*((DST_TYPE)block_src_ptr[0]);
            for( int i=1; i<block_size; ++i)    local_block[i]=((DST_TYPE)block_src_ptr[i])*((DST_TYPE)block_src_ptr[i])+local_block[i-1];
        } else {
            local_block[0] = last_sum + (DST_TYPE)block_src_ptr[0];
            for( int i=1; i<block_size; ++i)    local_block[i]=block_src_ptr[i]+local_block[i-1];
        }
        //copy local_block to dest
        for(int i = 0 ; i < block_size; ++i) {
            block_dst_ptr[i]=local_block[i];
        }
    }
 }
