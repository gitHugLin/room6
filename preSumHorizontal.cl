


__kernel void preSumHorizontal( __global unsigned char* _pSrc ,
                            __global unsigned int* _pDst ,const int width ,const int height)
{
    const int line_index = get_global_id(0);
    __global unsigned char* pSrc = line_index*width + _pSrc;
    __global unsigned int* pDst = line_index*width + _pDst;

    pDst[0] = pSrc[0];
    for(int i = 1; i < height; i++) {
        pDst[i] = pSrc[i] + pSrc[i-1];
    }
}
