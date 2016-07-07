


__kernel void preSumVertical( __global unsigned int* _pSrc ,
                            __global unsigned int* _pDst ,const int width ,const int height)
{
    const int line_index = get_global_id(0);
    __global unsigned int* pDst = line_index*height + _pDst;
    __global unsigned int* pSrc = line_index*height + _pSrc;

    pDst[0] = pSrc[0];
    for(int i = 1; i < width; i++) {
        pDst[i*height] = pSrc[i*height] + pSrc[(i-1)*height];
    }
}
