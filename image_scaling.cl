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
