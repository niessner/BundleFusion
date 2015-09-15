#ifndef _EIGENVALUEDECOMPOSITION_H_
#define _EIGENVALUEDECOMPOSITION_H_

#include "cuda_SimpleMatrixUtil.h"

/*https://en.wikipedia.org/wiki/Eigenvalue_algorithm#Direct_calculation*/
/*	Computes the Eigenvalues of a symmetric 3x3 Matrix aTa  */
/*	the eigenvalues satisfy eig.z <= eig.y <= eig.x			*/
__host__ __device__ inline float3 computeEigenValues(const float3x3& aTa) {
	const float PI = 3.14159265f;
	float3 eig;
	float p = aTa(0, 1) * aTa(0, 1) + aTa(0, 2) * aTa(0, 2) + aTa(1, 2) * aTa(1, 2);
	if (p == 0){//aTa is diagonal
		eig.x = aTa(0, 0);
		eig.y = aTa(1, 1);
		eig.z = aTa(2, 2);
	}
	else
	{
		float q = (aTa(0, 0) + aTa(1, 1) + aTa(2, 2)) / 3.0f; //trace(aTa)/3
		p = (aTa(0, 0) - q)*(aTa(0, 0) - q) + (aTa(1, 1) - q)*(aTa(1, 1) - q) + (aTa(2, 2) - q)*(aTa(2, 2) - q) + 2.0f * p;
		p = sqrt(p / 6.0f);
		float3x3 B = (aTa - float3x3(q, 0.0f, 0.0f, 0.0f, q, 0.0f, 0.0f, 0.0f, q)) * (1.0f / p);//aTa-q*I
		float r = B.det() / 2.0f;
		// In exact arithmetic for a symmetric matrix  -1 <= r <= 1  but computation error can leave it slightly outside this range.
		float phi;
		if (r <= -1.0f)
			phi = PI / 3.0f;
		else if (r >= 1)
			phi = 0;
		else
			phi = acos(r) / 3.0f;
		//the eigenvalues satisfy eig3 <= eig2 <= eig1
		eig.x = q + 2.0f * p * cos(phi);
		eig.z = q + 2.0f * p * cos(phi + PI * (2.0f / 3.0f));
		eig.y = 3.0f * q - eig.x - eig.z;    //since trace(A) = eig1 + eig2 + eig3;
	}
	return eig;
}

/* Computes the EigenVector of eigenValue eig (Vielfachheit 1!!) and symmetric 3x3 Matrix aTa  */
__device__ __host__ inline float3 computeEigenVector(const float3x3& aTa, float eig) {
	float3 res = make_float3(aTa(0, 1) * aTa(1, 2) - aTa(0, 2) * (aTa(1, 1) - eig),
		aTa(0, 1) * aTa(0, 2) - aTa(1, 2) * (aTa(0, 0) - eig),
		(aTa(0, 0) - eig) * (aTa(1, 1) - eig) - aTa(0, 1) * aTa(0, 1));
	return normalize(res);
}
/* Computes the EigenVector of eigenValue eig (Vielfachheit 2!!) and symmetric 3x3 Matrix aTa  */
//todo 
/* Computes the EigenVector of eigenValue eig (Vielfachheit 3!!) and symmetric 3x3 Matrix aTa  */
//trivial 






#endif //_EIGENVALUEDECOMPOSITION_H_