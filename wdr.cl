
#define LOCAL_BUFFER_SIZE 64

#define min(a,b)        ((a) > (b)?(b):(a))
#define max(a,b)        ((a) > (b)?(a):(b))

__kernel void preSumHorizontal( __global unsigned char* _pSrc ,
                            __global unsigned int* _pDst ,const int width ,const int height)
{
    const int line_index = get_global_id(0);
    __global unsigned char* pSrc = line_index*width + _pSrc;
    __global unsigned int*  pDst = line_index*width + _pDst;

    pDst[0] = pSrc[0];
    for(int i = 1; i < height; i++) {
        pDst[i] = pSrc[i] + pSrc[i-1];
    }
}

__kernel void preSumVertical( __global unsigned int* _pSrc ,
                            __global unsigned int* _pDst ,const int width ,const int height)
{
    const int line_index = get_global_id(1);
    __global unsigned int* pDst = line_index*height + _pDst;
    __global unsigned int* pSrc = line_index*height + _pSrc;

    pDst[0] = pSrc[0];
    for(int i = 1; i < width; i++) {
        pDst[i*height] = pSrc[i*height] + pSrc[(i-1)*height];
    }
}


__kernel void toneMapping( __global unsigned int* pIntegral ,__global unsigned char* pGray,__global float* mToneMapLut,
                            __global unsigned char* pDst, const int nCols ,const int nRows,const int mBlkSize)
{
    const int y = get_global_id(0);
    const int x = get_global_id(1);

    int xMin = max(x - mBlkSize, 0);
    int yMin = max(y - mBlkSize, 0);
    int xMax = min(x + mBlkSize, nCols - 1);
    int yMax = min(y + mBlkSize, nRows - 1);

    int blockAvgLumi = *(pIntegral+xMax+yMax*nCols) - *(pIntegral+xMin+yMax*nCols) -
                *(pIntegral+xMax+yMin*nCols) + *(pIntegral+xMin+yMin*nCols);
    blockAvgLumi = blockAvgLumi/((yMax - yMin + 2)*(xMax - xMin + 2));
    //blockAvgLumi = blockAvgLumi>>10;
    int offsetGray = y*nCols+x;
    int indexX = blockAvgLumi;
    int indexY = *(pGray+offsetGray);
    float gain = mToneMapLut[indexY*256+indexX];
    int curPixel = gain*indexY;
    *(pDst+offsetGray) = min(curPixel, 255);
}

__kernel  void image_rotate(  __global int * iInbuf,
                             __global int * iOutbuf,
                             int iWidth , int iHeight,
                             float fSinTheta, float fCosTheta )   //Rotation Parameters
{
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    int iXc = iWidth/2;
    int iYc = iHeight/2;
    int iXpos = ( ix - iXc)*fCosTheta - (iy- iYc)*fSinTheta+ iXc;
    int iYpos = (ix - iXc)*fSinTheta + ( iy- iYc)*fCosTheta+ iYc;
    if ((iXpos >=0) && (iXpos < iWidth) && (iYpos >=0) && (iYpos < iHeight))
        iOutbuf[iYpos*iWidth + iXpos] = iInbuf[iy*iWidth +ix];
}
