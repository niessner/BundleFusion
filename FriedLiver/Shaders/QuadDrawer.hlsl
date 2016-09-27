Texture2D<float> inputTexture : register (t10);
Texture2D<float4> inputTexture2 : register (t10);


sampler g_PointSampler : register (s10);


cbuffer cbPerFrame : register( b10 )
{
	float		m_WidthoverNextPowOfTwo;
	float		m_HeightoverNextPowOfTwo;
	float		g_Scale;
	uint		dummy1;
};

struct VS_INPUT
{
    float3 vPosition        : POSITION;
	float2 vTexcoord		: TEXCOORD;
};

struct VS_OUTPUT {
	float4 vPosition	: SV_POSITION;
	float2 vTexcoord	: TEXCOORD;
};

VS_OUTPUT QuadVS( VS_INPUT Input )
{
    VS_OUTPUT Output;
	Output.vPosition = float4(Input.vPosition, 1.0f);
	Output.vTexcoord = float2(m_WidthoverNextPowOfTwo*Input.vTexcoord.x, m_HeightoverNextPowOfTwo*Input.vTexcoord.y);

    return Output;
}

static float4 c0 = float4(0.6, 0.6, 0.0, 1.0);
static float4 c1 = float4(0.0, 0.0, 2.0, 1.0);
static float4 c2 = float4(0.0, 0.0, 2.0, 1.0);


float4 QuadFloatPS( VS_OUTPUT Input ) : SV_TARGET
{
	float r = inputTexture.Sample(g_PointSampler, Input.vTexcoord);
	r *= g_Scale;
	return float4(r, r, r, 1.0f);


	//if (r > 0.0)
	//{
	//	float s = 1.0-r;
	//	return r*r*c0 + 2.0*r*s*c1 + s*s*c2;
	//}
	//else {
	//	return (float4)0;
	//}
}




float4 QuadRGBAPS( VS_OUTPUT Input ) : SV_TARGET
{
	float4 res = inputTexture2.Sample(g_PointSampler, Input.vTexcoord);
	res.xyz *= g_Scale;
	return res;
}


float4 QuadPS3( VS_OUTPUT Input ) : SV_TARGET
{
	float4 res = inputTexture2.Sample(g_PointSampler, Input.vTexcoord);
	
	return res;
}

