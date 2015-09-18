#define MINF asfloat(0xff800000)

Texture2D<float4> inputPositions : register (t10);
Texture2D<float4> inputNormals : register (t11);
Texture2D<float4> inputColors : register (t12);

sampler g_PointSampler : register (s10);

cbuffer cbPerFrame : register( b0 )
{
	uint  g_useMaterial; 
	float dummy0;
	uint  dummy1;
	uint  dummy2;
};

cbuffer cbLight : register( b1 )
{
	float4 lightAmbient;
	float4 lightDiffuse;
	float4 lightSpecular;
	float3 lightDir;
	float  materialShininess;
	float4 materialAmbient;
	float4 materialSpecular;
	float4 materialDiffuse;
};

cbuffer cbPerFrame : register( b10 )
{
	float m_WidthoverNextPowOfTwo;
	float m_HeightoverNextPowOfTwo;
	float g_Scale;
	uint dummy10;
};

struct VS_INPUT
{
    float3 vPosition		: POSITION;
	float2 vTexcoord		: TEXCOORD;
};

struct VS_OUTPUT
{
	float4 vPosition		: SV_POSITION;
	float2 vTexcoord		: TEXCOORD;
};

float4 PhongPS(VS_OUTPUT Input) : SV_TARGET
{
	const float3 position = inputPositions.Sample(g_PointSampler, Input.vTexcoord).xyz;
	const float3 normal = inputNormals.Sample(g_PointSampler, Input.vTexcoord).xyz;
	const float3 color = inputColors.Sample(g_PointSampler, Input.vTexcoord).xyz;

	if(position.x != MINF && color.x != MINF && normal.x != MINF)
	{
		float4 material;
		if(g_useMaterial == 1)  material = float4(color, 1.0f);
		else					material = materialDiffuse;
		
		const float3 eyeDir = normalize(position);
		const float3 R = normalize(reflect(-normalize(lightDir), normal));
	
		return		lightAmbient  * materialAmbient														  // Ambient
				  + lightDiffuse  * material * max(dot(normal, -normalize(lightDir)), 0.0)				  // Diffuse
				  + lightSpecular * materialSpecular * pow(max(dot(R, eyeDir), 0.0f), materialShininess); // Specular
	}
	else
	{
		return float4(MINF, MINF, MINF, MINF);
	}
}
