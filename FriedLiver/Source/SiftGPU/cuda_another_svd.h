/*-
* Copyright (c) 2010 Nathan Lay
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
* IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
* OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
* IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
* NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
* DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
* THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
* THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef ANOTHER_SVD3_H
#define ANOTHEr_SVD3_H

#include <stdio.h>
#include <math.h>


class ANOTHER_SVD {
public:
	/*
	* NOTE: All computations are done COLUMN MAJOR!
	*/

#if defined(MATLAB_FMT)
#define printmat3_fmt(A)        #A " = [ %g %g %g; %g %g %g; %g %g %g ]\n"
#define readmat3_fmt(A)         "%lf %lf %lf %lf %lf %lf %lf %lf %lf"
#define printcol3_fmt(A,j)      #A " = [ %g %g %g ]'\n"
#define printcolp3_fmt(A,P,j)   #P #A " = [ %g %g %g ]'\n"
#define printrow3_fmt(A,i)      #A " = [ %g %g %g ]\n"
#define printL3_fmt(LDU,P)      "L = [ 1 0 0; %g 1 0; %g %g 1 ]\n"
#define printD3_fmt(LDU,P)      "D = [ %g 0 0; 0 %g 0; 0 0 %g ]\n"
#define printU3_fmt(LDU,P)      "U = [ 1 %g %g; 0 1 %g; 0 0 1 ]\n"
#define printP3_fmt(P)          #P " = [ %d %d %d; %d %d %d; %d %d %d ]\n"
#define printmatp3_fmt(A,P)     #A #P " = [ %g %g %g; %g %g %g; %g %g %g ]\n"
#else /* MATLAB_FMT */
#define printmat3_fmt(A)        #A " =\n%g %g %g\n%g %g %g\n%g %g %g\n"
#define readmat3_fmt(A)         "%lf %lf %lf %lf %lf %lf %lf %lf %lf"
#define printcol3_fmt(A,j)      #A " =\n%g\n%g\n%g\n"
#define printcolp3_fmt(A,P,j)   #P #A " =\n%g\n%g\n%g\n"
#define printrow3_fmt(A,i)      #A " =\n%g %g %g\n"
#define printL3_fmt(LDU,P)      "L =\n1 0 0\n%g 1 0\n%g %g 1\n"
#define printD3_fmt(LDU,P)      "D =\n%g 0 0\n0 %g 0\n0 0 %g\n"
#define printU3_fmt(LDU,P)      "U =\n1 %g %g\n0 1 %g\n0 0 1\n"
#define printP3_fmt(P)          #P " =\n%d %d %d\n%d %d %d\n%d %d %d\n"
#define printmatp3_fmt(A,P)     #A #P " =\n%g %g %g\n%g %g %g\n%g %g %g\n"
#endif /* !MATLAB_FMT */

#define printmat3(A)    ( printf(printmat3_fmt(A),                                      \
        ((float *)(A))[3*0+0], ((float *)(A))[3*1+0], ((float *)(A))[3*2+0],         \
        ((float *)(A))[3*0+1], ((float *)(A))[3*1+1], ((float *)(A))[3*2+1],         \
        ((float *)(A))[3*0+2], ((float *)(A))[3*1+2], ((float *)(A))[3*2+2]) )

#define readmat3(A)     ( scanf(readmat3_fmt(A),                                \
        (float *)(A)+(3*0+0), (float *)(A)+(3*1+0), (float *)(A)+(3*2+0),    \
        (float *)(A)+(3*0+1), (float *)(A)+(3*1+1), (float *)(A)+(3*2+1),    \
        (float *)(A)+(3*0+2), (float *)(A)+(3*1+2), (float *)(A)+(3*2+2)) )

#define printcol3(A,j)  ( printf(printcol3_fmt(A,j),    \
        ((float *)(A))[3*(j)+0],                       \
        ((float *)(A))[3*(j)+1],                       \
        ((float *)(A))[3*(j)+2]) )

#define printcolp3(A,P,j)       ( printf(printcolp3_fmt(A,P,j), \
        ((float *)(A))[3*(j)+(P)[0]],                          \
        ((float *)(A))[3*(j)+(P)[1]],                          \
        ((float *)(A))[3*(j)+(P)[2]]) )

#define printrow3(A,i)  ( printf(printrow3_fmt(A,i)                                     \
        ((float *)(A))[3*0+(i)], ((float *)(A))[3*1+(i)], ((float *)(A))[3*2+(i)]) )

#define printL3(LDU,P)  ( printf(printL3_fmt(LDU,P),                    \
        ((float *)(LDU))[3*(P)[0]+1],                                  \
        ((float *)(LDU))[3*(P)[0]+2], ((float *)(LDU))[3*(P)[1]+2]) )

#define printD3(LDU,P)  ( printf(printD3_fmt(LDU,P),    \
        ((float *)(LDU))[3*(P)[0]+0],                  \
        ((float *)(LDU))[3*(P)[1]+1],                  \
        ((float *)(LDU))[3*(P)[2]+2]) )

#define printU3(LDU,P)  ( printf(printU3_fmt(LDU,P),                    \
        ((float *)(LDU))[3*(P)[1]+0], ((float *)(LDU))[3*(P)[2]+0],   \
        ((float *)(LDU))[3*(P)[2]+1]) )

#define printP3(P)      ( printf(printP3_fmt(P),        \
        (P)[0]==0, (P)[1]==0, (P)[2]==0,                \
        (P)[0]==1, (P)[1]==1, (P)[2]==1,                \
        (P)[0]==2, (P)[1]==2, (P)[2]==2) )

#define printmatp3(A,P) ( printf(printmatp3_fmt(A,P),                                           \
        ((float *)(A))[3*(P)[0]+0], ((float *)(A))[3*(P)[1]+0], ((float *)(A))[3*(P)[2]+0],  \
        ((float *)(A))[3*(P)[0]+1], ((float *)(A))[3*(P)[1]+1], ((float *)(A))[3*(P)[2]+1],  \
        ((float *)(A))[3*(P)[0]+2], ((float *)(A))[3*(P)[1]+2], ((float *)(A))[3*(P)[2]+2]) )


	///* Computes cross product of 3D vectors x, y and stores the result in z */
	//static inline void cross(float * z, const float *   x,
	//	const float *   y);

	///* Sorts 3 elements */
	//static inline void sort3(float *   x);

	///* Normalizes a 3D vector (with respect to L2) */
	//static inline void unit3(float *   x);

	///*
	//* Solves for the roots of a monic cubic polynomial with 3 coefficients
	//* ordered by degree that is assumed to have 3 real roots (D <= 0)
	//*/
	//void solvecubic(float *   c);

	///* Computes the LDUP decomposition in-place */
	//void ldu3(float *   A, int *   P);

	///* Does the backward-solve step, or U*x = y */
	//static inline void ldubsolve3(float *   x, const float *   y,
	//	const float *   LDU, const int *   P);

	///* Explicitly computes the SVD of a 3x3 matrix */
	//void svd3(float *   U, float *   S, float *   V,
	//	const float *   A);

	///* Computes the matrix multiplication C = A*B */
	//static inline void matmul3(float *   C, const float *   A,
	//	const float *   B);

	///* Computes the matrix multiplication y = A*x */
	//static inline void matvec3(float *   y, const float *   A,
	//	const float *   x);

	///* Computes the matrix multiplication AA = A^T*A */
	//static inline void ata3(float *   AA, const float *   A);

	///* Computes the matrix multiplication AA = A*A^T */
	//static inline void aat3(float *   AA, const float *   A);

	///* Computes the matrix transpose of A */
	//static inline void trans3(float *   A);

	static inline void cross(float *   z, const float *   x,
		const float *   y) {
		z[0] = x[1] * y[2] - x[2] * y[1];
		z[1] = -(x[0] * y[2] - x[2] * y[0]);
		z[2] = x[0] * y[1] - x[1] * y[0];
	}

	static inline void sort3(float *   x) {
		float tmp;

		if (x[0] < x[1]) {
			tmp = x[0];
			x[0] = x[1];
			x[1] = tmp;
		}
		if (x[1] < x[2]) {
			if (x[0] < x[2]) {
				tmp = x[2];
				x[2] = x[1];
				x[1] = x[0];
				x[0] = tmp;
			}
			else {
				tmp = x[1];
				x[1] = x[2];
				x[2] = tmp;
			}
		}
	}

	static inline void unit3(float *   x) {
		float tmp = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
		x[0] /= tmp;
		x[1] /= tmp;
		x[2] /= tmp;
	}

	static inline void ldubsolve3(float *   x, const float *   y,
		const float *   LDU, const int *   P) {
		x[P[2]] = y[2];
		x[P[1]] = y[1] - LDU[3 * P[2] + 1] * x[P[2]];
		x[P[0]] = y[0] - LDU[3 * P[2] + 0] * x[P[2]] - LDU[3 * P[1] + 0] * x[P[1]];

#ifdef DEBUG
		puts("\nbsolve");
		printcol3(y, 0);
		putchar('\n');
		printcolp3(x, P, 0);
		putchar('\n');
#endif
	}

	static inline void matmul3(float *   C, const float *   A,
		const float *   B) {
		C[3 * 0 + 0] = A[3 * 0 + 0] * B[3 * 0 + 0] + A[3 * 1 + 0] * B[3 * 0 + 1] + A[3 * 2 + 0] * B[3 * 0 + 2];
		C[3 * 1 + 0] = A[3 * 0 + 0] * B[3 * 1 + 0] + A[3 * 1 + 0] * B[3 * 1 + 1] + A[3 * 2 + 0] * B[3 * 1 + 2];
		C[3 * 2 + 0] = A[3 * 0 + 0] * B[3 * 2 + 0] + A[3 * 1 + 0] * B[3 * 2 + 1] + A[3 * 2 + 0] * B[3 * 2 + 2];

		C[3 * 0 + 1] = A[3 * 0 + 1] * B[3 * 0 + 0] + A[3 * 1 + 1] * B[3 * 0 + 1] + A[3 * 2 + 1] * B[3 * 0 + 2];
		C[3 * 1 + 1] = A[3 * 0 + 1] * B[3 * 1 + 0] + A[3 * 1 + 1] * B[3 * 1 + 1] + A[3 * 2 + 1] * B[3 * 1 + 2];
		C[3 * 2 + 1] = A[3 * 0 + 1] * B[3 * 2 + 0] + A[3 * 1 + 1] * B[3 * 2 + 1] + A[3 * 2 + 1] * B[3 * 2 + 2];

		C[3 * 0 + 2] = A[3 * 0 + 2] * B[3 * 0 + 0] + A[3 * 1 + 2] * B[3 * 0 + 1] + A[3 * 2 + 2] * B[3 * 0 + 2];
		C[3 * 1 + 2] = A[3 * 0 + 2] * B[3 * 1 + 0] + A[3 * 1 + 2] * B[3 * 1 + 1] + A[3 * 2 + 2] * B[3 * 1 + 2];
		C[3 * 2 + 2] = A[3 * 0 + 2] * B[3 * 2 + 0] + A[3 * 1 + 2] * B[3 * 2 + 1] + A[3 * 2 + 2] * B[3 * 2 + 2];
	}

	static inline void matvec3(float *   y, const float *   A,
		const float *   x) {
		y[0] = A[3 * 0 + 0] * x[0] + A[3 * 1 + 0] * x[1] + A[3 * 2 + 0] * x[2];
		y[1] = A[3 * 0 + 1] * x[0] + A[3 * 1 + 1] * x[1] + A[3 * 2 + 1] * x[2];
		y[2] = A[3 * 0 + 2] * x[0] + A[3 * 1 + 2] * x[1] + A[3 * 2 + 2] * x[2];
	}

	static inline void ata3(float *   AA, const float *   A) {
		AA[3 * 0 + 0] = A[3 * 0 + 0] * A[3 * 0 + 0] + A[3 * 0 + 1] * A[3 * 0 + 1] + A[3 * 0 + 2] * A[3 * 0 + 2];
		AA[3 * 1 + 0] = A[3 * 0 + 0] * A[3 * 1 + 0] + A[3 * 0 + 1] * A[3 * 1 + 1] + A[3 * 0 + 2] * A[3 * 1 + 2];
		AA[3 * 2 + 0] = A[3 * 0 + 0] * A[3 * 2 + 0] + A[3 * 0 + 1] * A[3 * 2 + 1] + A[3 * 0 + 2] * A[3 * 2 + 2];

		AA[3 * 0 + 1] = AA[3 * 1 + 0];
		AA[3 * 1 + 1] = A[3 * 1 + 0] * A[3 * 1 + 0] + A[3 * 1 + 1] * A[3 * 1 + 1] + A[3 * 1 + 2] * A[3 * 1 + 2];
		AA[3 * 2 + 1] = A[3 * 1 + 0] * A[3 * 2 + 0] + A[3 * 1 + 1] * A[3 * 2 + 1] + A[3 * 1 + 2] * A[3 * 2 + 2];

		AA[3 * 0 + 2] = AA[3 * 2 + 0];
		AA[3 * 1 + 2] = AA[3 * 2 + 1];
		AA[3 * 2 + 2] = A[3 * 2 + 0] * A[3 * 2 + 0] + A[3 * 2 + 1] * A[3 * 2 + 1] + A[3 * 2 + 2] * A[3 * 2 + 2];
	}

	static inline void aat3(float *   AA, const float *   A) {
		AA[3 * 0 + 0] = A[3 * 0 + 0] * A[3 * 0 + 0] + A[3 * 1 + 0] * A[3 * 1 + 0] + A[3 * 2 + 0] * A[3 * 2 + 0];
		AA[3 * 1 + 0] = A[3 * 0 + 0] * A[3 * 0 + 1] + A[3 * 1 + 0] * A[3 * 1 + 1] + A[3 * 2 + 0] * A[3 * 2 + 1];
		AA[3 * 2 + 0] = A[3 * 0 + 0] * A[3 * 0 + 2] + A[3 * 1 + 0] * A[3 * 1 + 2] + A[3 * 2 + 0] * A[3 * 2 + 2];

		AA[3 * 0 + 1] = AA[3 * 1 + 0];
		AA[3 * 1 + 1] = A[3 * 0 + 1] * A[3 * 0 + 1] + A[3 * 1 + 1] * A[3 * 1 + 1] + A[3 * 2 + 1] * A[3 * 2 + 1];
		AA[3 * 2 + 1] = A[3 * 0 + 1] * A[3 * 0 + 2] + A[3 * 1 + 1] * A[3 * 1 + 2] + A[3 * 2 + 1] * A[3 * 2 + 2];

		AA[3 * 0 + 2] = AA[3 * 2 + 0];
		AA[3 * 1 + 2] = AA[3 * 2 + 1];
		AA[3 * 2 + 2] = A[3 * 0 + 2] * A[3 * 0 + 2] + A[3 * 1 + 2] * A[3 * 1 + 2] + A[3 * 2 + 2] * A[3 * 2 + 2];
	}

	static inline void trans3(float *   A) {
		float tmp;

		tmp = A[3 * 1 + 0];
		A[3 * 1 + 0] = A[3 * 0 + 1];
		A[3 * 0 + 1] = tmp;

		tmp = A[3 * 2 + 0];
		A[3 * 2 + 0] = A[3 * 0 + 2];
		A[3 * 0 + 2] = tmp;

		tmp = A[3 * 2 + 1];
		A[3 * 2 + 1] = A[3 * 1 + 2];
		A[3 * 1 + 2] = tmp;
	}










	void solvecubic(float *   c) {
		const float sq3d2 = 0.86602540378443864676f, c2d3 = c[2] / 3,
			c2sq = c[2] * c[2], Q = (3 * c[1] - c2sq) / 9,
			R = (c[2] * (9 * c[1] - 2 * c2sq) - 27 * c[0]) / 54;
		float tmp, t, sint, cost;

		if (Q < 0) {
			/*
			* Instead of computing
			* c_0 = A cos(t) - B
			* c_1 = A cos(t + 2 pi/3) - B
			* c_2 = A cos(t + 4 pi/3) - B
			* Use cos(a+b) = cos(a) cos(b) - sin(a) sin(b)
			* Keeps t small and eliminates 1 function call.
			* cos(2 pi/3) = cos(4 pi/3) = -0.5
			* sin(2 pi/3) = sqrt(3)/2
			* sin(4 pi/3) = -sqrt(3)/2
			*/

			tmp = 2 * sqrt(-Q);
			t = acos(R / sqrt(-Q*Q*Q)) / 3;
			cost = tmp*cos(t);
			sint = tmp*sin(t);

			c[0] = cost - c2d3;

			cost = -0.5f*cost - c2d3;
			sint = sq3d2*sint;

			c[1] = cost - sint;
			c[2] = cost + sint;
		}
		else {
			tmp = cbrt(R);
			c[0] = -c2d3 + 2 * tmp;
			c[1] = c[2] = -c2d3 - tmp;
		}
	}

	void ldu3(float *   A, int *   P) {
		int tmp;

#ifdef DEBUG
		printmat3(A);
		putchar('\n');
#endif

		P[1] = 1;
		P[2] = 2;

		P[0] = fabs(A[3 * 1 + 0]) > fabs(A[3 * 0 + 0]) ?
			(fabs(A[3 * 2 + 0]) > fabs(A[3 * 1 + 0]) ? 2 : 1) :
			(fabs(A[3 * 2 + 0]) > fabs(A[3 * 0 + 0]) ? 2 : 0);
		P[P[0]] = 0;

		if (fabs(A[3 * P[2] + 1]) > fabs(A[3 * P[1] + 1])) {
			tmp = P[1];
			P[1] = P[2];
			P[2] = tmp;
		}

#ifdef DEBUG
		printmatp3(A, P);
		putchar('\n');
#endif

		if (A[3 * P[0] + 0] != 0) {
			A[3 * P[1] + 0] = A[3 * P[1] + 0] / A[3 * P[0] + 0];
			A[3 * P[2] + 0] = A[3 * P[2] + 0] / A[3 * P[0] + 0];
			A[3 * P[0] + 1] = A[3 * P[0] + 1] / A[3 * P[0] + 0];
			A[3 * P[0] + 2] = A[3 * P[0] + 2] / A[3 * P[0] + 0];
		}

		A[3 * P[1] + 1] = A[3 * P[1] + 1] - A[3 * P[0] + 1] * A[3 * P[1] + 0] * A[3 * P[0] + 0];

		if (A[3 * P[1] + 1] != 0) {
			A[3 * P[2] + 1] = (A[3 * P[2] + 1] - A[3 * P[0] + 1] * A[3 * P[2] + 0] * A[3 * P[0] + 0]) / A[3 * P[1] + 1];
			A[3 * P[1] + 2] = (A[3 * P[1] + 2] - A[3 * P[0] + 2] * A[3 * P[1] + 0] * A[3 * P[0] + 0]) / A[3 * P[1] + 1];
		}

		A[3 * P[2] + 2] = A[3 * P[2] + 2] - A[3 * P[0] + 2] * A[3 * P[2] + 0] * A[3 * P[0] + 0] - A[3 * P[1] + 2] * A[3 * P[2] + 1] * A[3 * P[1] + 1];

#ifdef DEBUG
		printL3(A, P);
		putchar('\n');
		printD3(A, P);
		putchar('\n');
		printU3(A, P);
		putchar('\n');
		printP3(P);
#endif
	}

	void svd3(float *   U, float *   S, float *   V,
		const float *   A) {
		const float thr = 1e-10f;
		int P[3], k;
		float y[3], AA[3][3], LDU[3][3];

		/*
		* Steps:
		* 1) Use eigendecomposition on A^T A to compute V.
		* Since A = U S V^T then A^T A = V S^T S V^T with D = S^T S and V the
		* eigenvalues and eigenvectors respectively (V is orthogonal).
		* 2) Compute U from A and V.
		* 3) Normalize columns of U and V and root the eigenvalues to obtain
		* the singular values.
		*/

		/* Compute AA = A^T A */
		ata3((float *)AA, A);

		/* Form the monic characteristic polynomial */
		S[2] = -AA[0][0] - AA[1][1] - AA[2][2];
		S[1] = AA[0][0] * AA[1][1] + AA[2][2] * AA[0][0] + AA[2][2] * AA[1][1] -
			AA[2][1] * AA[1][2] - AA[2][0] * AA[0][2] - AA[1][0] * AA[0][1];
		S[0] = AA[2][1] * AA[1][2] * AA[0][0] + AA[2][0] * AA[0][2] * AA[1][1] + AA[1][0] * AA[0][1] * AA[2][2] -
			AA[0][0] * AA[1][1] * AA[2][2] - AA[1][0] * AA[2][1] * AA[0][2] - AA[2][0] * AA[0][1] * AA[1][2];

		/* Solve the cubic equation. */
		solvecubic(S);

		/* All roots should be positive */
		if (S[0] < 0)
			S[0] = 0;
		if (S[1] < 0)
			S[1] = 0;
		if (S[2] < 0)
			S[2] = 0;

		/* Sort from greatest to least */
		sort3(S);

		/* Form the eigenvector system for the first (largest) eigenvalue */
		memcpy(LDU, AA, sizeof(LDU));
		LDU[0][0] -= S[0];
		LDU[1][1] -= S[0];
		LDU[2][2] -= S[0];

		/* Perform LDUP decomposition */
		ldu3((float *)LDU, P);

		/*
		* Write LDU = AA-I*lambda.  Then an eigenvector can be
		* found by solving LDU x = LD y = L z = 0
		* L is invertible, so L z = 0 implies z = 0
		* D is singular since det(AA-I*lambda) = 0 and so
		* D y = z = 0 has a non-unique solution.
		* Pick k so that D_kk = 0 and set y = e_k, the k'th column
		* of the identity matrix.
		* U is invertible so U x = y has a unique solution for a given y.
		* The solution for U x = y is an eigenvector.
		*/

		/* Pick the component of D nearest to 0 */
		y[0] = y[1] = y[2] = 0;
		k = fabs(LDU[P[1]][1]) < fabs(LDU[P[0]][0]) ?
			(fabs(LDU[P[2]][2]) < fabs(LDU[P[1]][1]) ? 2 : 1) :
			(fabs(LDU[P[2]][2]) < fabs(LDU[P[0]][0]) ? 2 : 0);
		y[k] = 1;

		/* Do a backward solve for the eigenvector */
		ldubsolve3(V + (3 * 0 + 0), y, (float *)LDU, P);

		/* Form the eigenvector system for the last (smallest) eigenvalue */
		memcpy(LDU, AA, sizeof(LDU));
		LDU[0][0] -= S[2];
		LDU[1][1] -= S[2];
		LDU[2][2] -= S[2];

		/* Perform LDUP decomposition */
		ldu3((float *)LDU, P);

		/*
		* NOTE: The arrangement of the ternary operator output is IMPORTANT!
		* It ensures a different system is solved if there are 3 repeat eigenvalues.
		*/

		/* Pick the component of D nearest to 0 */
		y[0] = y[1] = y[2] = 0;
		k = fabs(LDU[P[0]][0]) < fabs(LDU[P[2]][2]) ?
			(fabs(LDU[P[0]][0]) < fabs(LDU[P[1]][1]) ? 0 : 1) :
			(fabs(LDU[P[1]][1]) < fabs(LDU[P[2]][2]) ? 1 : 2);
		y[k] = 1;

		/* Do a backward solve for the eigenvector */
		ldubsolve3(V + (3 * 2 + 0), y, (float *)LDU, P);

		/* The remaining column must be orthogonal (AA is symmetric) */
		cross(V + (3 * 1 + 0), V + (3 * 2 + 0), V + (3 * 0 + 0));

		/* Count the rank */
		k = (S[0] > thr) + (S[1] > thr) + (S[2] > thr);

		switch (k) {
		case 0:
			/*
			* Zero matrix.
			* Since V is already orthogonal, just copy it into U.
			*/
			memcpy(U, V, 9 * sizeof(float));
			break;
		case 1:
			/*
			* The first singular value is non-zero.
			* Since A = U S V^T, then A V = U S.
			* A V_1 = S_11 U_1 is non-zero. Here V_1 and U_1 are
			* column vectors. Since V_1 is known, we may compute
			* U_1 = A V_1.  The S_11 factor is not important as
			* U_1 will be normalized later.
			*/
			matvec3(U + (3 * 0 + 0), A, V + (3 * 0 + 0));

			/*
			* The other columns of U do not contribute to the expansion
			* and we may arbitrarily choose them (but they do need to be
			* orthogonal). To ensure the first cross product does not fail,
			* pick k so that U_k1 is nearest 0 and then cross with e_k to
			* obtain an orthogonal vector to U_1.
			*/
			y[0] = y[1] = y[2] = 0;
			k = fabs(U[3 * 0 + 0]) < fabs(U[3 * 0 + 2]) ?
				(fabs(U[3 * 0 + 0]) < fabs(U[3 * 0 + 1]) ? 0 : 1) :
				(fabs(U[3 * 0 + 1]) < fabs(U[3 * 0 + 2]) ? 1 : 2);
			y[k] = 1;

			cross(U + (3 * 1 + 0), y, U + (3 * 0 + 0));

			/* Cross the first two to obtain the remaining column */
			cross(U + (3 * 2 + 0), U + (3 * 0 + 0), U + (3 * 1 + 0));
			break;
		case 2:
			/*
			* The first two singular values are non-zero.
			* Compute U_1 = A V_1 and U_2 = A V_2. See case 1
			* for more information.
			*/
			matvec3(U + (3 * 0 + 0), A, V + (3 * 0 + 0));
			matvec3(U + (3 * 1 + 0), A, V + (3 * 1 + 0));

			/* Cross the first two to obtain the remaining column */
			cross(U + (3 * 2 + 0), U + (3 * 0 + 0), U + (3 * 1 + 0));
			break;
		case 3:
			/*
			* All singular values are non-zero.
			* We may compute U = A V. See case 1 for more information.
			*/
			matmul3(U, A, V);
			break;

		}

		/* Normalize the columns of U and V */
		unit3(V + (3 * 0 + 0));
		unit3(V + (3 * 1 + 0));
		unit3(V + (3 * 2 + 0));

		unit3(U + (3 * 0 + 0));
		unit3(U + (3 * 1 + 0));
		unit3(U + (3 * 2 + 0));

		/* S was initially the eigenvalues of A^T A = V S^T S V^T which are squared. */
		S[0] = sqrt(S[0]);
		S[1] = sqrt(S[1]);
		S[2] = sqrt(S[2]);
	}
};

#endif 
