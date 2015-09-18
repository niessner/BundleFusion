
//Buffer<float4>	g_vertexBuffer	: register(t0);


//struct GS_INPUT
//{
//};

struct VS_OUTPUT
{
	float4 position	: SV_POSITION;
	float depth : DEPTH_WORLD;
};

VS_OUTPUT VS( float4 position : POSITION )
{
	VS_OUTPUT output;
	output.position = float4(position.xyz, 1.0f);
	output.depth = position.w;

	return output;
}

//[maxvertexcount(4)]
//void GS(point GS_INPUT points[1], uint primID : SV_PrimitiveID, inout TriangleStream<PS_INPUT> triStream)
//{
//	PS_INPUT output;
//
//	uint idx = primID * 4;
//
//	output.position = float4(g_vertexBuffer[idx].xyz, 1.0f);
//	output.depth = g_vertexBuffer[idx].w;
//	triStream.Append(output);				
//
//	output.position = float4(g_vertexBuffer[idx+1].xyz, 1.0f);
//	output.depth = g_vertexBuffer[idx+1].w;
//	triStream.Append(output);				
//
//	output.position = float4(g_vertexBuffer[idx+2].xyz, 1.0f);
//	output.depth = g_vertexBuffer[idx+2].w;
//	triStream.Append(output);				
//
//	output.position = float4(g_vertexBuffer[idx+3].xyz, 1.0f);
//	output.depth = g_vertexBuffer[idx+3].w;
//	triStream.Append(output);
//}

//void PS(PS_INPUT input) 
//{
//}
float PS( VS_OUTPUT In) : SV_TARGET
{
	return In.depth;
}