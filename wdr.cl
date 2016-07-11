#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define min(a,b)        ((a) > (b)?(b):(a))
#define max(a,b)        ((a) > (b)?(a):(b))


__kernel void wdr( __global unsigned char* _pSrc,__global float* mToneMapLut,
                            __global unsigned char* _pDst, const int nCols ,const int nRows)
{
    const int x = get_global_id(0)*8;
    const int y = get_global_id(1);
    const int index = y*nCols + x;
    // const xMax = min(x,nRows - 16);
    // const xMin = max(x,nRows - 16);
    // const yMax = min(y,nCols - 16);
    // const yMin = max(y,nCols - 16);

    __global uchar* pSrc = y*nCols + x + _pSrc;
    __global uchar* pDst = y*nCols + x + _pDst;

    if(x >= 8 && y >= 8 && x <= nCols-8 && y <= nRows-8) {
        __global uchar* origin = pSrc-3*nCols;
        uchar16 line1   = vload16(0,origin - 4);
        uchar16 line2   = vload16(0,origin + nCols - 4);
        uchar16 line3   = vload16(0,origin + 2*nCols - 4);
        uchar16 line4   = vload16(0,origin + 3*nCols - 4);
        uchar16 line5   = vload16(0,origin + 4*nCols - 4);
        uchar16 line6   = vload16(0,origin + 5*nCols - 4);
        uchar16 line7   = vload16(0,origin + 6*nCols - 4);
        uchar16 line8   = vload16(0,origin + 7*nCols - 4);
        // uchar16 line9   = vload16(0,origin + 8*nCols - 8);
        // uchar16 line10  = vload16(0,origin + 9*nCols - 8);
        // uchar16 line11  = vload16(0,origin + 10*nCols - 8);
        // uchar16 line12  = vload16(0,origin + 11*nCols - 8);
        // uchar16 line13  = vload16(0,origin + 12*nCols - 8);
        // uchar16 line14  = vload16(0,origin + 13*nCols - 8);
        // uchar16 line15  = vload16(0,origin + 14*nCols - 8);
        // uchar16 line16  = vload16(0,origin + 15*nCols - 8);
        uchar8 curLumi;
        uchar8 pixel = vload8(0,pSrc);
        int8 pixelLumi = convert_int8(pixel);
        //part 1
        int8 sum = convert_int8(line1.s01234567) + convert_int8(line2.s01234567) +
          convert_int8(line3.s01234567)+ convert_int8(line4.s01234567)+ convert_int8(line5.s01234567)
          + convert_int8(line6.s01234567) + convert_int8(line7.s01234567) + convert_int8(line8.s01234567);
        int blockAvgLumi = (sum.s0 + sum.s1 + sum.s2 +sum.s3 + sum.s4 + sum.s5 + sum.s6 + sum.s7)>>6;
        //int pixelLumi = (int)*pSrc;
        int indexX = blockAvgLumi;
        int finalPxiel = convert_int(mToneMapLut[pixelLumi.s0*256+indexX]);
        //int finalPxiel = pixelLumi.s0*(1.18+pixelLumi.s0*blockAvgLumi)/(pixelLumi.s0+blockAvgLumi+0.18);
        curLumi.s0 = (uchar)min(finalPxiel,255);
        //part 2
        sum = convert_int8(line1.s12345678) + convert_int8(line2.s12345678) +
          convert_int8(line3.s12345678)+ convert_int8(line4.s12345678)+ convert_int8(line5.s12345678)
          + convert_int8(line6.s12345678) + convert_int8(line7.s12345678) + convert_int8(line8.s12345678);
        blockAvgLumi = (sum.s0 + sum.s1 + sum.s2 +sum.s3 + sum.s4 + sum.s5 + sum.s6 + sum.s7)>>6;
        //pixelLumi = (int)*(pSrc+1);
        indexX = blockAvgLumi;
        finalPxiel = convert_int(mToneMapLut[pixelLumi.s1*256+indexX]);
        //finalPxiel = pixelLumi.s1*(1.18+pixelLumi.s1*blockAvgLumi)/(pixelLumi.s1+blockAvgLumi+0.18);
        curLumi.s1 = (uchar)min(finalPxiel,255);
        //part 3
        sum = convert_int8(line1.s3456789a) + convert_int8(line2.s3456789a) +
          convert_int8(line3.s3456789a)+ convert_int8(line4.s3456789a)+ convert_int8(line5.s3456789a)
          + convert_int8(line6.s3456789a) + convert_int8(line7.s3456789a) + convert_int8(line8.s3456789a);
        blockAvgLumi = (sum.s0 + sum.s1 + sum.s2 +sum.s3 + sum.s4 + sum.s5 + sum.s6 + sum.s7)>>6;
        //pixelLumi = (int)*(pSrc+2);
        indexX = blockAvgLumi;
        finalPxiel = convert_int(mToneMapLut[pixelLumi.s2*256+indexX]);
        // finalPxiel = pixelLumi.s2*(1.18+pixelLumi.s2*blockAvgLumi)/(pixelLumi.s2+blockAvgLumi+0.18);
        curLumi.s2 = (uchar)min(finalPxiel,255);
        //part 4
        sum = convert_int8(line1.s456789ab) + convert_int8(line2.s456789ab) +
          convert_int8(line3.s456789ab)+ convert_int8(line4.s456789ab)+ convert_int8(line5.s456789ab)
          + convert_int8(line6.s456789ab) + convert_int8(line7.s456789ab) + convert_int8(line8.s456789ab);
        blockAvgLumi = (sum.s0 + sum.s1 + sum.s2 +sum.s3 + sum.s4 + sum.s5 + sum.s6 + sum.s7)>>6;
        //pixelLumi = (int)*(pSrc+3);
        indexX = blockAvgLumi;
        finalPxiel = convert_int(mToneMapLut[pixelLumi.s3*256+indexX]);
        // finalPxiel = pixelLumi.s3*(1.18+pixelLumi.s3*blockAvgLumi)/(pixelLumi.s3+blockAvgLumi+0.18);
        curLumi.s3 = (uchar)min(finalPxiel,255);
        //part 5
        sum = convert_int8(line1.s56789abc) + convert_int8(line2.s56789abc) +
          convert_int8(line3.s56789abc)+ convert_int8(line4.s56789abc)+ convert_int8(line5.s56789abc)
          + convert_int8(line6.s56789abc) + convert_int8(line7.s56789abc) + convert_int8(line8.s56789abc);
        blockAvgLumi = (sum.s0 + sum.s1 + sum.s2 +sum.s3 + sum.s4 + sum.s5 + sum.s6 + sum.s7)>>6;
        //pixelLumi = (int)*(pSrc+4);
        indexX = blockAvgLumi;
        finalPxiel = convert_int(mToneMapLut[pixelLumi.s4*256+indexX]);
        // finalPxiel = pixelLumi.s4*(1.18+pixelLumi.s4*blockAvgLumi)/(pixelLumi.s4+blockAvgLumi+0.18);
        curLumi.s4 = (uchar)min(finalPxiel,255);
        //part 6
        sum = convert_int8(line1.s6789abcd) + convert_int8(line2.s6789abcd) +
          convert_int8(line3.s6789abcd)+ convert_int8(line4.s6789abcd)+ convert_int8(line5.s6789abcd)
          + convert_int8(line6.s6789abcd) + convert_int8(line7.s6789abcd) + convert_int8(line8.s6789abcd);
        blockAvgLumi = (sum.s0 + sum.s1 + sum.s2 +sum.s3 + sum.s4 + sum.s5 + sum.s6 + sum.s7)>>6;
        //pixelLumi = (int)*(pSrc+5);
        indexX = blockAvgLumi;
        finalPxiel = convert_int(mToneMapLut[pixelLumi.s5*256+indexX]);
        // finalPxiel = pixelLumi.s5*(1.18+pixelLumi.s5*blockAvgLumi)/(pixelLumi.s5+blockAvgLumi+0.18);
        curLumi.s5 = (uchar)min(finalPxiel,255);
        //part 7
        sum = convert_int8(line1.s789abcde) + convert_int8(line2.s789abcde) +
          convert_int8(line3.s789abcde)+ convert_int8(line4.s789abcde)+ convert_int8(line5.s789abcde)
          + convert_int8(line6.s789abcde) + convert_int8(line7.s789abcde) + convert_int8(line8.s789abcde);
        blockAvgLumi = (sum.s0 + sum.s1 + sum.s2 +sum.s3 + sum.s4 + sum.s5 + sum.s6 + sum.s7)>>6;
        //pixelLumi = (int)*(pSrc+6);
        indexX = blockAvgLumi;
        finalPxiel = convert_int(mToneMapLut[pixelLumi.s6*256+indexX]);
        // finalPxiel = pixelLumi.s6*(1.18+pixelLumi.s6*blockAvgLumi)/(pixelLumi.s6+blockAvgLumi+0.18);
        curLumi.s6 = (uchar)min(finalPxiel,255);
        //part 8
        sum = convert_int8(line1.s89abcdef) + convert_int8(line2.s89abcdef) +
          convert_int8(line3.s89abcdef)+ convert_int8(line4.s89abcdef)+ convert_int8(line5.s89abcdef)
          + convert_int8(line6.s89abcdef) + convert_int8(line7.s89abcdef) + convert_int8(line8.s89abcdef);
        blockAvgLumi = (sum.s0 + sum.s1 + sum.s2 +sum.s3 + sum.s4 + sum.s5 + sum.s6 + sum.s7)>>6;
        //pixelLumi = (int)*(pSrc+7);
        indexX = blockAvgLumi;
        finalPxiel = convert_int(mToneMapLut[pixelLumi.s7*256+indexX]);
        // finalPxiel = pixelLumi.s7*(1.18+pixelLumi.s7*blockAvgLumi)/(pixelLumi.s7+blockAvgLumi+0.18);
        curLumi.s7 = (uchar)min(finalPxiel,255);

        vstore8(curLumi,0,pDst);
        //*pDst = curLumi;
    } else {

        // int pixelLumi = (int)*pSrc;
        // float gain = mToneMapLut[pixelLumi*256+pixelLumi];
        // int finalPxiel = gain*pixelLumi;
        // uchar curLumi = (uchar)min(finalPxiel,255);
        // *pDst = curLumi;

    }

    // uchar16 line9  = vload16(0,pSrc + 8*origin);
    // uchar16 line10 = vload16(0,pSrc + 9*origin);
    // uchar16 line11 = vload16(0,pSrc + 10*origin);
    // uchar16 line12 = vload16(0,pSrc + 11*origin);
    // uchar16 line13 = vload16(0,pSrc + 12*origin);
    // uchar16 line14 = vload16(0,pSrc + 13*origin);
    // uchar16 line15 = vload16(0,pSrc + 14*origin);
    // uchar16 line16 = vload16(0,pSrc + 15*origin);

}


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
    blockAvgLumi = blockAvgLumi/((yMax - yMin)*(xMax - xMin));

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
