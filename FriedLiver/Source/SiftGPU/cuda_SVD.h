#pragma once

#ifndef _CUDA_SVD_H_
#define _CUDA_SVD_H_

#include "cuda_SimpleMatrixUtil.h"

#include <cutil_inline.h>
#include <cutil_math.h>


#define M_DIM 3

class MYEIGEN {
public:

	static inline bool __device__ __host__ eigenSystem(const float3x3& m, float3& eigenvalues, float3& ev0, float3& ev1, float3& ev2, bool sort = true) {
		float* evs[3] = { (float*)&ev0, (float*)&ev1, (float*)&ev2 };
		return eigenSystem<3>((const float*)&m, (float*)&eigenvalues, evs, sort);
	}

	//static inline bool __device__ __host__ eigenSystem3x3(const float3x3& m, float3& eigenvalues, float3& ev0, float3& ev1, float3& ev2, bool sort = true) {

	//	float3x3 eigenvectors;
	//	bool res = eigenSystem3x3(m, (float*)&eigenvalues, eigenvectors, sort);
	//	ev0 = eigenvectors.getRow(0);
	//	ev1 = eigenvectors.getRow(1);
	//	ev2 = eigenvectors.getRow(2);
	//	return res;
	//}
private:


	//static inline bool __device__ __host__ eigenSystem3x3(const float3x3& m, float* eigenvalues, float3x3& eigenvectors, bool sort = true) {

	//	int num_of_required_jabobi_rotations;
	//	float3x3 input = m;
	//	if (!jacobi3(input.entries2, eigenvalues, eigenvectors.entries2, &num_of_required_jabobi_rotations)) {
	//		return false;
	//	}

	//	if (sort) {
	//		//simple selection sort
	//		for (unsigned int i = 0; i < 3; i++) {
	//			float currMax = 0.0f;
	//			unsigned int currMaxIdx = (unsigned int)-1;
	//			for (unsigned int j = i; j < 3; j++) {
	//				if (fabsf(eigenvalues[j]) > currMax) {
	//					currMax = fabsf(eigenvalues[j]);
	//					currMaxIdx = j;
	//				}
	//			}

	//			if (currMaxIdx != i && currMaxIdx != (unsigned int)-1) {
	//				swap(eigenvalues[i], eigenvalues[currMaxIdx]);
	//				for (unsigned int j = 0; j < 3; j++) {
	//					//swap(eigenvectors(i, j), eigenvectors(currMaxIdx, j));
	//					swap(eigenvectors(i,j), eigenvectors(currMaxIdx,j));
	//				}
	//			}
	//		}
	//	}

	//	return true;
	//}



	template<unsigned int n>
	static inline bool __device__ __host__ eigenSystem(const float* m, float* eigenvalues, float** eigenvectors, bool sort = true) {
		//TODO ONLY WORKS for n==3
		float tmpCV0[n + 1], tmpCV1[n + 1], tmpCV2[n + 1], tmpCV3[n + 1];
		float* tmpCV[n + 1] = { tmpCV0, tmpCV1, tmpCV2, tmpCV3 };
		float** CV = tmpCV;

		float tmpLambda[n + 1];
		float* lambda = tmpLambda;

		float tmpV0[n + 1], tmpV1[n + 1], tmpV2[n + 1], tmpV3[n + 1];
		float* tmpV[n + 1] = { tmpV0, tmpV1, tmpV2, tmpV3 };
		float** v = tmpV;

		for (unsigned int i = 0; i < n; i++) {
			for (unsigned int j = 0; j < n; j++) {
				CV[i + 1][j + 1] = m[i + n*j];
			}
		}

		int num_of_required_jabobi_rotations;

		if (!jacobi<n>(CV, lambda, v, &num_of_required_jabobi_rotations)) {
			return false;
		}

		for (unsigned int i = 0; i < n; i++) {
			eigenvalues[i] = lambda[i + 1];
			for (unsigned int j = 0; j < n; j++) {
				eigenvectors[i][j] = v[i + 1][j + 1];
			}
		}

		if (sort) {
			//simple selection sort
			for (unsigned int i = 0; i < n; i++) {
				float currMax = 0.0f;
				unsigned int currMaxIdx = (unsigned int)-1;
				for (unsigned int j = i; j < n; j++) {
					if (fabsf(eigenvalues[j]) > currMax) {
						currMax = fabsf(eigenvalues[j]);
						currMaxIdx = j;
					}
				}

				if (currMaxIdx != i && currMaxIdx != (unsigned int)-1) {
					swap(eigenvalues[i], eigenvalues[currMaxIdx]);
					for (unsigned int j = 0; j < n; j++) {
						//swap(eigenvectors(i, j), eigenvectors(currMaxIdx, j));
						swap(eigenvectors[i][j], eigenvectors[currMaxIdx][j]);
					}
				}
			}
		}

		return true;
	}

	template<class T>
	__host__ __device__ inline static void swap(T& a, T& b) {
		T tmp = a;
		a = b;
		b = tmp;
	}

#define CUDA_SVD_ROTATE(a,i,j,k,l) {g=a[i][j];h=a[k][l];a[i][j]=g-s*(h+g*tau);a[k][l]=h+s*(g-h*tau); }


	template<unsigned int n>
	static inline __device__ __host__ bool jacobi(float **a, float d[], float **v, int *nrot) {
		int j, iq, ip, i;
		float tresh, theta, tau, t, sm, s, h, g, c;
		float b[n + 1];
		float z[n + 1];

		for (ip = 1; ip <= n; ip++) {
			for (iq = 1; iq <= n; iq++) v[ip][iq] = (float)0.0;
			v[ip][ip] = (float)1.0;
		}
		for (ip = 1; ip <= n; ip++) {
			b[ip] = d[ip] = a[ip][ip];
			z[ip] = (float)0.0;
		}
		*nrot = 0;
		for (i = 1; i <= 50; i++) {
			sm = (float)0.0;
			for (ip = 1; ip <= n - 1; ip++) {
				for (iq = ip + 1; iq <= n; iq++)
					sm += fabs(a[ip][iq]);
			}
			if (sm == 0.0) {
				return true;
			}
			if (i < 4)
				tresh = (float)0.2*sm / (n*n);
			else
				tresh = (float)0.0;
			for (ip = 1; ip <= n - 1; ip++) {
				for (iq = ip + 1; iq <= n; iq++) {
					g = (float)100.0*fabs(a[ip][iq]);
					if (i > 4 && (float)(fabs(d[ip]) + g) == (float)fabs(d[ip]) && (float)(fabs(d[iq]) + g) == (float)fabs(d[iq]))
						a[ip][iq] = (float)0.0;
					else if (fabs(a[ip][iq]) > tresh) {
						h = d[iq] - d[ip];
						if ((float)(fabs(h) + g) == (float)fabs(h))
							t = (a[ip][iq]) / h;
						else {
							theta = (float)0.5*h / (a[ip][iq]);
							t = (float)1.0 / (fabs(theta) + sqrt((float)1.0 + theta*theta));
							if (theta < 0.0)
								t = -t;
						}
						c = (float)1.0 / sqrt(1 + t*t);
						s = t*c;
						tau = s / ((float)1.0 + c);
						h = t*a[ip][iq];
						z[ip] -= (float)h;
						z[iq] += (float)h;
						d[ip] -= (float)h;
						d[iq] += (float)h;
						a[ip][iq] = (float)0.0;
						for (j = 1; j <= ip - 1; j++) {
							CUDA_SVD_ROTATE(a, j, ip, j, iq);
						}
						for (j = ip + 1; j <= iq - 1; j++) {
							CUDA_SVD_ROTATE(a, ip, j, j, iq);
						}
						for (j = iq + 1; j <= n; j++) {
							CUDA_SVD_ROTATE(a, ip, j, iq, j);
						}
						for (j = 1; j <= n; j++) {
							CUDA_SVD_ROTATE(v, j, ip, j, iq);
						}
						++(*nrot);
					}
				}
			}
			for (ip = 1; ip <= n; ip++) {
				b[ip] += z[ip];
				d[ip] = b[ip];
				z[ip] = (float)0.0;
			}
		}
		return false;
	}



	//static inline __device__ __host__ bool jacobi3(float a[3][3], float d[3], float v[3][3], int* n_rot)
	//{
	//	int count, k, i, j;
	//	float tresh, theta, tau, t, sum, s, h, g, c, b[3], z[3];

	//	/*Initialize v to the identity matrix.*/
	//	for (i = 0; i < 3; i++)
	//	{
	//		for (j = 0; j < 3; j++)
	//			v[i][j] = 0.0f;
	//		v[i][i] = 1.0f;
	//	}

	//	/* Initialize b and d to the diagonal of a */
	//	for (i = 0; i < 3; i++)
	//		b[i] = d[i] = a[i][i];

	//	/* z will accumulate terms */
	//	for (i = 0; i < 3; i++)
	//		z[i] = 0.0f;

	//	*n_rot = 0;

	//	/* 50 tries */
	//	for (count = 0; count < 50; count++)
	//	{

	//		/* sum off-diagonal elements */
	//		sum = 0.0f;
	//		for (i = 0; i < 2; i++)
	//		{
	//			for (j = i + 1; j < 3; j++)
	//				sum += fabs(a[i][j]);
	//		}

	//		/* if converged to machine underflow */
	//		if (sum == 0.0)
	//			return(1);

	//		/* on 1st three sweeps... */
	//		if (count < 3)
	//			tresh = sum * 0.2f / 9.0f;
	//		else
	//			tresh = 0.0f;

	//		for (i = 0; i < 2; i++)
	//		{
	//			for (j = i + 1; j < 3; j++)
	//			{
	//				g = 100.0f * fabs(a[i][j]);

	//				/*  after four sweeps, skip the rotation if
	//				*   the off-diagonal element is small
	//				*/
	//				if (count > 3 && fabs(d[i]) + g == fabs(d[i])
	//					&& fabs(d[j]) + g == fabs(d[j]))
	//				{
	//					a[i][j] = 0.0f;
	//				}
	//				else if (fabs(a[i][j]) > tresh)
	//				{
	//					h = d[j] - d[i];

	//					if (fabs(h) + g == fabs(h))
	//					{
	//						t = a[i][j] / h;
	//					}
	//					else
	//					{
	//						theta = 0.5f * h / (a[i][j]);
	//						t = 1.0f / (fabs(theta) +
	//							sqrtf(1.0f + theta*theta));
	//						if (theta < 0.0f)
	//							t = -t;
	//					}

	//					c = 1.0f / sqrtf(1 + t*t);
	//					s = t * c;
	//					tau = s / (1.0f + c);
	//					h = t * a[i][j];

	//					z[i] -= h;
	//					z[j] += h;
	//					d[i] -= h;
	//					d[j] += h;

	//					a[i][j] = 0.0;

	//					for (k = 0; k <= i - 1; k++)
	//						CUDA_SVD_ROTATE(a, k, i, k, j)
	//						for (k = i + 1; k <= j - 1; k++)
	//							CUDA_SVD_ROTATE(a, i, k, k, j)
	//							for (k = j + 1; k < 3; k++)
	//								CUDA_SVD_ROTATE(a, i, k, j, k)
	//								for (k = 0; k < 3; k++)
	//									CUDA_SVD_ROTATE(v, k, i, k, j)
	//									++(*n_rot);
	//				}
	//			}
	//		}

	//		for (i = 0; i < 3; i++)
	//		{
	//			b[i] += z[i];
	//			d[i] = b[i];
	//			z[i] = 0.0;
	//		}
	//	}
	//	return false;
	//}

};





class SVD {
public:

	static inline __device__ __host__ bool svd(const float3x3& m, float3x3& u, float3x3& s, float3x3& v)
	{
		float3 _s;
		u = m;
		bool res = decompose3x3((float*)&u, (float*)&v, (float*)&_s);
		s.setZero();
		s(0, 0) = _s.x;
		s(1, 1) = _s.y;
		s(2, 2) = _s.z;
		return res;
	}

	//Implementation based on Numerical Recipies
	//Given Matrix A, this routine computes its singular value decomposition, A = U * W * V_trans
	//and stores results in the matrices u, v and the vector w (sigma): note that v returns really v - NOT v_trans
	static inline __device__ __host__ bool decompose3x3(float* u, float* v, float* w) {

		//mxn -> here always 3x3
		const int n = 3;
		const int m = 3;

		int flag;
		int i, its, j, jj, k, l, nm;
		float anorm, c, f, g, h, s, scale, x, y, z;
		float rv1[n];

		g = scale = anorm = 0.0f; //Householder reduction to bidiagonal form.

		for (i = 0; i < n; i++) {
			l = i + 2;
			rv1[i] = scale * g;
			g = s = scale = 0.0f;
			if (i < m){
				for (k = i; k < m; k++){
					scale += fabsf(getElem(u, k, i));
				}
				if (scale){
					for (k = i; k < m; k++){
						//u[k][i] /= scale;
						divFromElem(u, k, i, scale);
						//s += u[k][i]*u[k][i];
						s += getElem(u, k, i) * getElem(u, k, i);
					}
					//f=u[i][i];
					f = getElem(u, i, i);
					g = -SIGN(sqrt(s), f);
					h = f*g - s;
					//u[i][i]=f-g;
					setElem(u, i, i, f - g);
					for (j = l - 1; j < n; j++) {
						for (s = 0.0f, k = i; k < m; k++) s += getElem(u, k, i)*getElem(u, k, j); //u[k][i]*u[k][j];
						f = (float)s / h;
						for (k = i; k < m; k++)  addToElem(u, k, j, f*getElem(u, k, i));//u[k][j] += f*u[k][i];
					}

					for (k = i; k < m; k++) multToElem(u, k, i, scale); //u[k][i] *= scale;
				}
			}

			w[i] = scale*g;

			g = s = scale = 0.0f;
			if (i + 1 <= m && i + 1 != n){
				for (k = l - 1; k < n; k++) scale += fabsf(getElem(u, i, k)); //u[i][k]);
				if (scale != 0.0f) {
					for (k = l - 1; k < n; k++){
						//u[i][k] /= scale;
						divFromElem(u, i, k, scale);
						//s += u[i][k]*u[i][k];
						s += getElem(u, i, k) * getElem(u, i, k);
					}
					f = getElem(u, i, l - 1);//u[i][l-1];
					g = -SIGN(sqrt(s), f);
					h = f*g - s;
					//u[i][l-1]=f-g;
					setElem(u, i, l - 1, (f - g));
					for (k = l - 1; k < n; k++) rv1[k] = getElem(u, i, k) / h;//u[i][k]/h;
					for (j = l - 1; j < m; j++){
						for (s = 0.0f, k = l - 1; k < n; k++) s += getElem(u, j, k) * getElem(u, i, k);//u[j][k]*u[i][k];
						for (k = l - 1; k < n; k++) addToElem(u, j, k, s*rv1[k]); //u[j][k] += s*rv1[k];
					}
					for (k = l - 1; k < n; k++) multToElem(u, i, k, scale);//u[i][k] *= scale;
				}
			}
			anorm = /*MAX*/fmaxf(anorm, (fabsf(w[i]) + fabsf(rv1[i])));
		}
		for (i = n - 1; i >= 0; i--){ //Accumulation of right-hand transformations.
			if (i < n - 1){
				if (g){
					for (j = l; j < n; j++){
						float myX = getElem(u, i, j) / getElem(u, i, l) / g; //v[j][i]=(u[i][j]/u[i][l])/g;
						setElem(v, j, i, myX);
					}
					for (j = l; j < n; j++){
						for (s = 0.0f, k = l; k < n; k++) s += getElem(u, i, k)*getElem(v, k, j);//s+=u[i][k]*v[k][j];
						for (k = l; k < n; k++) addToElem(v, k, j, s*getElem(v, k, i));//v[k][j] += s*v[k][i];
					}
				}
				for (j = l; j < n; j++){
					//v[i][j]=v[j][i]=0.0;
					setElem(v, i, j, 0.0f);
					setElem(v, j, i, 0.0f);
				}
			}
			//v[i][i]=1.0;
			setElem(v, i, i, 1.0f);
			g = rv1[i];
			l = i;
		}
		for (i = /*MIN*/min(m, n) - 1; i >= 0; i--){ //Accumulation of left-hand transformations.
			l = i + 1;
			g = w[i];
			for (j = l; j < n; j++) setElem(u, i, j, 0.0f);//u[i][j]=0.0;
			if (g != 0.0f) {
				g = 1.0f / g;
				for (j = l; j < n; j++){
					for (s = 0.0f, k = l; k < m; k++) s += getElem(u, k, i)*getElem(u, k, j); //s += u[k][i]*u[k][j];
					f = (s / getElem(u, i, i))*g;	//f = (s/u[i][i])*g;
					for (k = i; k < m; k++) addToElem(u, k, j, f*getElem(u, k, i));//u[k][j] += f*u[k][i];
				}
				for (j = i; j < m; j++) multToElem(u, j, i, g);//u[j][i] *= g;
			}
			else for (j = i; j < m; j++) setElem(u, j, i, 0.0f);//u[j][i]=0.0;
			addToElem(u, i, i, 1.0f);//++u[i][i];
		}
		for (k = n - 1; k >= 0; k--) { //Diagonalization of the bidiagonal form: Loop over singular values, and over allowed iterations. for (its=0;its<30;its++) {
			for (its = 0; its < 30; its++){
				//flag = true;
				flag = 1;
				for (l = k; l >= 0; l--){ //Test for splitting.
					nm = l - 1;
					//if(l == 0 || abs(rv1[l]) <= /*eps* */anorm) {
					if ((float)(fabsf(rv1[l]) + anorm) == anorm){
						//flag = false;
						flag = 0;
						break;
					}
					//if(abs(w[nm]) <= /*eps* */anorm) break;
					if ((float)(fabsf(w[nm]) + anorm) == anorm) break;
				}
				if (flag){
					c = 0.0f; //Cancellation of rv1[l], if l > 0.
					s = 1.0f;

					for (i = l; i < k + 1; i++){
						f = s*rv1[i];
						rv1[i] = c*rv1[i];
						//if(abs(f) <= /*eps* */anorm) break;
						if ((float)(fabsf(f) + anorm) == anorm) break;

						g = w[i];
						h = pythag(f, g);
						w[i] = h;
						h = 1.0f / h;
						c = g*h;
						s = -f*h;
						for (j = 0; j < m; j++){
							y = getElem(u, j, nm);//u[j][nm];
							z = getElem(u, j, i);//u[j][i];
							//u[j][nm]=y*c+z*s;
							setElem(u, j, nm, y*c + z*s);
							//u[j][i]=z*c-y*s;
							setElem(u, j, i, z*c - y*s);
						}
					}
				}
				z = w[k];
				if (l == k){ //Convergence.
					if (z < 0.0f) { //Singular value is made nonnegative.
						w[k] = -z;
						for (j = 0; j < n; j++) setElem(v, j, k, -getElem(v, j, k));//v[j][k] = -v[j][k];
					}
					break;
				}
				if (its == 29){
					return false;
				}
				x = w[l]; //Shift from bottom 2-by-2 minor.
				nm = k - 1;
				y = w[nm];
				g = rv1[nm];
				h = rv1[k];
				f = ((y - z)*(y + z) + (g - h)*(g + h)) / (2.0f*h*y);
				g = pythag(f, 1.0f);
				f = ((x - z)*(x + z) + h*((y / (f + SIGN(g, f))) - h)) / x;
				c = s = 1.0f; //Next QR transformation:
				for (j = l; j <= nm; j++){
					i = j + 1;
					g = rv1[i];
					y = w[i];
					h = s*g;
					g = c*g;
					z = pythag(f, h);
					rv1[j] = z;
					c = f / z;
					s = h / z;
					f = x*c + g*s;
					g = g*c - x*s;
					h = y*s;
					y *= c;
					for (jj = 0; jj < n; jj++){
						//x = v[jj][j];
						x = getElem(v, jj, j);
						//z = v[jj][i];
						z = getElem(v, jj, i);
						//v[jj][j]=x*c+z*s;
						setElem(v, jj, j, x*c + z*s);
						//v[jj][i]=z*c-x*s;
						setElem(v, jj, i, z*c - x*s);
					}
					z = pythag(f, h);
					w[j] = z; //Rotation can be arbitrary if z D 0.
					if (z){
						z = 1.0f / z;
						c = f*z;
						s = h*z;
					}
					f = c*g + s*y;
					x = c*y - s*g;
					for (jj = 0; jj < m; jj++){
						y = getElem(u, jj, j);//u[jj][j];
						z = getElem(u, jj, i);//u[jj][i];

						//u[jj][j]=y*c+z*s;
						setElem(u, jj, j, y*c + z*s);
						//u[jj][i]=z*c-y*s;
						setElem(u, jj, i, z*c - y*s);
					}
				}
				rv1[l] = 0.0f;
				rv1[k] = f;
				w[k] = x;
			}
		}
		return true;
	}

private:

	//helper functions - maybe clean up...
	static inline __device__ __host__ float SIGN(float a, float b) {
		return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);
	}

	static inline __device__ __host__ float getElem(float* m, int i, int j) {
		return m[i * M_DIM + j];
	}

	static inline __device__ __host__ void setElem(float* m, int i, int j, float v){
		m[i * M_DIM + j] = v;
	}

	static inline __device__ __host__ void addToElem(float* m, int i, int j, float v){
		m[i * M_DIM + j] += v;
	}

	static inline __device__ __host__ void multToElem(float* m, int i, int j, float v){
		m[i * M_DIM + j] *= v;
	}

	static inline __device__ __host__  void divFromElem(float* m, int i, int j, float v){
		m[i * M_DIM + j] /= v;
	}

	static inline __device__ __host__ float SQR(const float a){
		return a*a;
	}

	static inline __device__ __host__ float pythag(float a, float b){
		float absa = fabsf(a);
		float absb = fabsf(b);
		return	(absa > absb ? absa*sqrt(1.0f + SQR(absb / absa)) :
			(absb == 0.0f ? 0.0f : absb*sqrt(1.0f + SQR(absa / absb))));
	}
};





#endif
