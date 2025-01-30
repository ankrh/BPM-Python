/********************************************
 * FDBPMpropagator.c, in the C programming language, written for MATLAB MEX function generation
 *
 ** Compiling on Windows
 * Can be compiled with GCC using
 * "mex COPTIMFLAGS='$COPTIMFLAGS -Ofast -fopenmp -std=c11 -Wall' LDOPTIMFLAGS='$LDOPTIMFLAGS -Ofast -fopenmp -std=c11 -Wall' -outdir +BPMmatlab\@model\private .\src\FDBPMpropagator.c ".\src\libut.lib" -R2018a"
 * ... or the Microsoft Visual C++ compiler (MSVC) with
 * "copyfile ./src/FDBPMpropagator.c ./src/FDBPMpropagator.cpp; mex COMPFLAGS='/Zp8 /GR /EHs /nologo /MD /openmp /W4 /WX /wd4204 /wd4100' -outdir +BPMmatlab\@model\private .\src\FDBPMpropagator.cpp ".\src\libut.lib" -R2018a"
 *
 * The source code in this file is written is such a way that it is
 * compilable by either C or C++ compilers, either with GCC, MSVC or
 * the Nvidia CUDA compiler called NVCC, which is based on MSVC. To
 * compile with CUDA GPU acceleration support, you must have MSVC
 * installed. As of January 2020, mexcuda does not work with MSVC 2019,
 * so I'd recommend MSVC 2017. You also need the Parallel Computing
 * Toolbox, which you will find in the MATLAB addon manager. To compile, run:
 * "copyfile ./src/FDBPMpropagator.c ./src/FDBPMpropagator_CUDA.cu; mexcuda -llibut COMPFLAGS='-use_fast_math -res-usage $COMPFLAGS' -outdir +BPMmatlab\@model\private .\src\FDBPMpropagator_CUDA.cu -R2018a"
 *
 ** Compiling on macOS
 * As of March 2021, the macOS compiler doesn't support libut (for ctrl+c
 * breaking) or openmp (for multithreading).
 * "mex COPTIMFLAGS='$COPTIMFLAGS -Ofast -std=c11 -Wall' LDOPTIMFLAGS='$LDOPTIMFLAGS -Ofast -std=c11 -Wall' -outdir +BPMmatlab/@model/private ./src/FDBPMpropagator.c -R2018a"
 *
 * To get the MATLAB C compiler to work, try this:
 * 1. Install XCode from the App Store
 * 2. Type "mex -setup" in the MATLAB command window
 *
 ** Compiling on Linux
 * "mex COPTIMFLAGS='$COPTIMFLAGS -Ofast -fopenmp -std=c11 -Wall' LDOPTIMFLAGS='$LDOPTIMFLAGS -Ofast -fopenmp -std=c11 -Wall' -outdir +BPMmatlab/@model/private ./src/FDBPMpropagator.c -R2018a -lut"
 *
 * To get the MATLAB C compiler to work, try this:
 * 1. Use a package manager like apt to install GCC (on Ubuntu, part of the build-essential package)
 * 2. Type "mex -setup" in the MATLAB command window
 ********************************************/
 // printf("Reached line %d...\n",__LINE__);mexEvalString("drawnow; pause(.001);");mexEvalString("drawnow; pause(.001);");mexEvalString("drawnow; pause(.001);"); // For inserting into code for debugging purposes

#include <math.h>
#include <stdint.h>
#define PI acosf(-1.0f)
#include "omp.h"
#include <algorithm>
#include <complex>
typedef std::complex<float> floatcomplex;
#define I std::complex<float>{0,1}
#define CEXPF(x) (std::exp(x))
#define CREALF(x) (std::real(x))
#define CIMAGF(x) (std::imag(x))
#define MAX(x,y) (std::max(x,y))
#define MIN(x,y) (std::min(x,y))
#define FLOORF(x) (std::floor(x))

#include "print.h"

struct debug {
    double             dbls[3];
    unsigned long long ulls[3];
};

struct parameters {
    long Nx{};
    long Ny{};
    float dx{};
    float dy{};
    float dz{};
    long iz_start{};
    long iz_end{};
    unsigned char xSymmetry{};
    unsigned char ySymmetry{};
    float taperPerStep{};
    float twistPerStep{};
    float d{};
    float n_0{};
    floatcomplex* n_in{};
    long  Nx_n{};
    long  Ny_n{};
    long  Nz_n{};
    float dz_n{};
    floatcomplex* Efinal{};
    floatcomplex* E1{};
    floatcomplex* E2{};
    floatcomplex* Eyx{};
    floatcomplex* n_out{};
    floatcomplex* b{};
    float* multiplier{};
    floatcomplex ax{};
    floatcomplex ay{};
    float rho_e{};
    float RoC{};
    float sinBendDirection{};
    float cosBendDirection{};
    double precisePower{};
    float precisePowerDiff{};
    float EfieldPower{};
};

float sqrf(float x) { return x * x; }

void substep1a(struct parameters* P_global) {
    // Explicit part of substep 1 out of 2
    long ix, iy;
    struct parameters* P = P_global;
    bool xAntiSymm = P->xSymmetry == 2;
    bool yAntiSymm = P->ySymmetry == 2;
#pragma omp for schedule(dynamic)
    for (iy = 0; iy < P->Ny; iy++) {
        for (ix = 0; ix < P->Nx; ix++) {
            long i = ix + iy * P->Nx;

            P->E2[i] = P->E1[i];
            if (ix != 0) P->E2[i] += (P->E1[i - 1] - P->E1[i]) * P->ax;
            if (ix != P->Nx - 1 && (!yAntiSymm || ix != 0)) P->E2[i] += (P->E1[i + 1] - P->E1[i]) * P->ax;
            if (iy != 0) P->E2[i] += (P->E1[i - P->Nx] - P->E1[i]) * P->ay * 2.0f;
            if (iy != P->Ny - 1 && (!xAntiSymm || iy != 0)) P->E2[i] += (P->E1[i + P->Nx] - P->E1[i]) * P->ay * 2.0f;
        }
    }
}

void substep1b(struct parameters* P_global) {
    // Implicit part of substep 1 out of 2
    struct parameters* P = P_global;
    bool yAntiSymm = P->ySymmetry == 2;
    long i, ix, iy;
    long threadNum = omp_get_thread_num();
#pragma omp for schedule(dynamic)
    for (iy = 0; iy < P->Ny; iy++) {
        // Thomson algorithm, sweeps up from 0 to Nx-1 and then down from Nx-1 to 0:
        // Algorithm is taken from https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
        for (ix = 0; ix < P->Nx; ix++) {
            long ib = ix + threadNum * P->Nx;
            if (ix == 0 && yAntiSymm) P->b[ib] = 1.0f;
            else if (ix == 0) P->b[ib] = 1.0f + P->ax;
            else if (ix < P->Nx - 1) P->b[ib] = 1.0f + 2.0f * P->ax;
            else                          P->b[ib] = 1.0f + P->ax;

            if (ix > 0) {
                floatcomplex w = -P->ax / P->b[ib - 1];
                P->b[ib] += w * (ix == 1 && yAntiSymm ? 0 : P->ax);
                i = ix + iy * P->Nx;
                P->E2[i] -= w * P->E2[i - 1];
            }
        }

        for (ix = P->Nx - 1; ix >= 0 + yAntiSymm; ix--) {
            long ib = ix + threadNum * P->Nx;
            i = ix + iy * P->Nx;
            P->E2[i] = (P->E2[i] + (ix == P->Nx - 1 ? 0 : P->ax * P->E2[i + 1])) / P->b[ib];
        }
    }
}

void substep2a(struct parameters* P_global) {
    // Explicit part of substep 2 out of 2
    struct parameters* P = P_global;
    bool xAntiSymm = P->xSymmetry == 2;
    long i, ix, iy;
#pragma omp for schedule(dynamic)
    for (iy = 0; iy < P->Ny; iy++) {
        for (ix = 0; ix < P->Nx; ix++) {
            i = ix + iy * P->Nx;

            if (iy != 0) P->E2[i] -= (P->E1[i - P->Nx] - P->E1[i]) * P->ay;
            if (iy != P->Ny - 1 && (!xAntiSymm || iy != 0)) P->E2[i] -= (P->E1[i + P->Nx] - P->E1[i]) * P->ay;
        }
    }
}

void substep2b(struct parameters* P_global) {
    // Implicit part of substep 2 out of 2
    float EfieldPowerThread = 0.0f;
    struct parameters* P = P_global;
    bool xAntiSymm = P->xSymmetry == 2;
    long i, ix, iy;
    long threadNum = omp_get_thread_num();
#pragma omp for schedule(dynamic)
    for (ix = 0; ix < P->Nx; ix++) {
        for (iy = 0; iy < P->Ny; iy++) {
            long ib = iy + threadNum * P->Ny;
            if (iy == 0 && xAntiSymm) P->b[ib] = 1.0f;
            else if (iy == 0) P->b[ib] = 1.0f + P->ay;
            else if (iy < P->Ny - 1) P->b[ib] = 1.0f + 2.0f * P->ay;
            else                          P->b[ib] = 1.0f + P->ay;

            if (iy > 0) {
                floatcomplex w = -P->ay / P->b[ib - 1];
                P->b[ib] += w * (iy == 1 && xAntiSymm ? 0 : P->ay);
                i = ix + iy * P->Nx;
                P->E2[i] -= w * P->E2[i - P->Nx];
            }
        }

        for (iy = P->Ny - 1; iy >= 0 + xAntiSymm; iy--) {
            long ib = iy + threadNum * P->Ny;
            i = ix + iy * P->Nx;
            P->E2[i] = (P->E2[i] + (iy == P->Ny - 1 ? 0 : P->ay * P->E2[i + P->Nx])) / P->b[ib];
            EfieldPowerThread += sqrf(CREALF(P->E2[i])) + sqrf(CIMAGF(P->E2[i]));
        }
    }
#pragma omp atomic
    P->EfieldPower += EfieldPowerThread;
#pragma omp barrier
}

void applyMultiplier(struct parameters* P_global, long iz, struct debug* D) {
    float precisePowerDiffThread = 0.0f;
    struct parameters* P = P_global;
    float fieldCorrection = sqrtf((float)P->precisePower / P->EfieldPower);
    float cosvalue = cosf(-P->twistPerStep * iz); // Minus is because we go from the rotated frame to the source frame
    float sinvalue = sinf(-P->twistPerStep * iz);
    float scaling = 1 / (1 - P->taperPerStep * iz); // Take reciprocal because we go from scaled frame to unscaled frame
#pragma omp for schedule(dynamic)
    for (long i = 0;i < P->Nx * P->Ny;i++) {
        long ix = i % P->Nx;
        float x = P->dx * (ix - (P->Nx - 1) / 2.0f * (P->ySymmetry == 0));
        long iy = i / P->Nx;
        float y = P->dy * (iy - (P->Ny - 1) / 2.0f * (P->xSymmetry == 0));
        floatcomplex n = 0;
        if (P->taperPerStep || P->twistPerStep) { // Rotate, scale, interpolate. If we are tapering or twisting, we know that the RIP is 2D
            float x_src = scaling * (cosvalue * x - sinvalue * y);
            float y_src = scaling * (sinvalue * x + cosvalue * y);
            float ix_src = MIN(MAX(0.0f, x_src / P->dx + (P->Nx - 1) / 2.0f * (P->ySymmetry == 0)), (P->Nx - 1) * (1 - FLT_EPSILON)); // Fractional index, coerced to be within the source window
            float iy_src = MIN(MAX(0.0f, y_src / P->dy + (P->Ny - 1) / 2.0f * (P->xSymmetry == 0)), (P->Ny - 1) * (1 - FLT_EPSILON));
            long ix_low = (long)FLOORF(ix_src);
            long iy_low = (long)FLOORF(iy_src);
            float ix_frac = ix_src - FLOORF(ix_src);
            float iy_frac = iy_src - FLOORF(iy_src);
            n = P->n_in[ix_low + P->Nx * (iy_low)] * (1 - ix_frac) * (1 - iy_frac) +
                P->n_in[ix_low + 1 + P->Nx * (iy_low)] * (ix_frac) * (1 - iy_frac) +
                P->n_in[ix_low + P->Nx * (iy_low + 1)] * (1 - ix_frac) * (iy_frac)+
                P->n_in[ix_low + 1 + P->Nx * (iy_low + 1)] * (ix_frac) * (iy_frac); // Bilinear interpolation
        }
        else if (P->Nz_n == 1) { // 2D RIP
            n = P->n_in[i];
        }
        else { // 3D RIP
            float z = iz * P->dz;
            long ix_n = MIN(MAX(0L, ix - (P->Nx - P->Nx_n) / 2), P->Nx_n - 1);
            long iy_n = MIN(MAX(0L, iy - (P->Ny - P->Ny_n) / 2), P->Ny_n - 1);
            float iz_n = MIN(MAX(0.0f, z / P->dz_n), (P->Nz_n - 1) * (1 - FLT_EPSILON)); // Fractional index, coerced to be within the n window
            long iz_n_low = (long)FLOORF(iz_n);
            float iz_n_frac = iz_n - FLOORF(iz_n);
            n = P->n_in[ix_n + P->Nx_n * iy_n + P->Ny_n * P->Nx_n * (iz_n_low)] * (1 - iz_n_frac) +
                P->n_in[ix_n + P->Nx_n * iy_n + P->Ny_n * P->Nx_n * (iz_n_low + 1)] * (iz_n_frac); // Linear interpolation in z
        }
        if (iz == P->iz_end - 1) P->n_out[i] = n;
        float n_bend = CREALF(n) * (1 - (sqrf(CREALF(n)) * (x * P->cosBendDirection + y * P->sinBendDirection) / 2 / P->RoC * P->rho_e)) * exp((x * P->cosBendDirection + y * P->sinBendDirection) / P->RoC);
        floatcomplex a = P->multiplier[i] * CEXPF(P->d * (CIMAGF(n) + (sqrf(n_bend) - sqrf(P->n_0)) * I / (2 * P->n_0))); // Multiplier includes only the edge absorber
        P->E2[i] *= fieldCorrection * a;
        float anormsqr = sqrf(CREALF(a)) + sqrf(CIMAGF(a));
        if (anormsqr > 1 - 10 * FLT_EPSILON && anormsqr < 1 + 10 * FLT_EPSILON) anormsqr = 1; // To avoid accumulating power discrepancies due to rounding errors
        precisePowerDiffThread += (sqrf(CREALF(P->E2[i])) + sqrf(CIMAGF(P->E2[i]))) * (1 - 1 / anormsqr);
    }

#pragma omp atomic
    P->precisePowerDiff += precisePowerDiffThread;
#pragma omp barrier
    }

void updatePrecisePower(struct parameters* P) {
    P->precisePower += P->precisePowerDiff;
    P->precisePowerDiff = 0;
}

void swapEPointers(struct parameters* P, long iz) {
    P->EfieldPower = 0;
    if (iz > P->iz_start) { // Swap E1 and E2
        floatcomplex* temp = P->E1;
        P->E1 = P->E2;
        P->E2 = temp;
    }
    else if ((P->iz_end - P->iz_start) % 2) {
        P->E1 = P->E2;
        P->E2 = (floatcomplex*)malloc(P->Nx * P->Ny * sizeof(floatcomplex));
    }
    else {
        P->E1 = P->E2;
        P->E2 = P->Efinal;
    }
}

extern "C" __declspec(dllexport) void entryfunc(
        floatcomplex* E1, floatcomplex* Efinal,
        long Nx, long Ny,
        float dx, float dy, float dz,
        long iz_start, long iz_end,
        float taperPerStep, float twistPerStep,
        unsigned char xSymmetry, unsigned char ySymmetry,
        float d, float n_0,
        floatcomplex* n_in, floatcomplex* n_out,
        long Nx_n, long Ny_n, long Nz_n,
        float dz_n, float rho_e, float RoC,
        float bendDirection,
        double precisePower, double* outputPrecisePowerPtr,
        float* multiplier,
        floatcomplex ax, floatcomplex ay,
        bool useAllCPUs) {
	struct parameters P_var;
	struct parameters* P = &P_var;
    P->E1 = E1; P->Efinal = Efinal; // Input and output E fields
    printArray(P->E1,5);printArray(P->Efinal,5);
    P->Nx = Nx;	P->Ny = Ny;
	print(P->Nx);print(P->Ny);
	P->dx = dx;	P->dy = dy;	P->dz = dz;
	print(P->dx);print(P->dy);print(P->dz);
	P->iz_start = iz_start;	P->iz_end = iz_end;
    print(P->iz_start);print(P->iz_end);
    P->E2 = (floatcomplex*)((P->iz_end - P->iz_start) % 2 ? P->Efinal : malloc(P->Nx * P->Ny * sizeof(floatcomplex)));
    P->taperPerStep = taperPerStep;	P->twistPerStep = twistPerStep;
	print(P->taperPerStep);print(P->twistPerStep);
	P->xSymmetry = xSymmetry; P->ySymmetry = ySymmetry;
	print(P->xSymmetry);print(P->ySymmetry);
	P->d = d; P->n_0 = n_0;
	print(P->d);print(P->n_0);
    P->n_in = n_in; P->n_out = n_out; // Input and output refractive index
	printArray(P->n_in, 5);printArray(P->n_out, 5);
	P->Nx_n = Nx_n;	P->Ny_n = Ny_n;	P->Nz_n = Nz_n;
	print(P->Nx_n);print(P->Ny_n);print(P->Nz_n);
	P->dz_n = dz_n;	P->rho_e = rho_e; P->RoC = RoC;
	print(P->dz_n);print(P->rho_e);print(P->RoC);
	P->sinBendDirection = sin(bendDirection / 180 * PI);
	P->cosBendDirection = cos(bendDirection / 180 * PI);
	P->precisePower = precisePower;
	print(P->sinBendDirection);print(P->cosBendDirection);print(P->precisePower);
	P->multiplier = multiplier; // Array of multiplier values to apply to the E field after each step, due to the edge absorber outside the main simulation window
	P->ax = ax;
	P->ay = ay;
	printArray(P->multiplier, 5);print(P->ax);print(P->ay);
	bool ctrlc_caught = false;      // Has a ctrl+c been passed from MATLAB?
	P->EfieldPower = 0;
	P->precisePowerDiff = 0;
	long numThreads = useAllCPUs || omp_get_num_procs() == 1 ? omp_get_num_procs() : omp_get_num_procs() - 1;
	P->b = (floatcomplex*)malloc(numThreads * MAX(P->Nx, P->Ny) * sizeof(floatcomplex));
#pragma omp parallel num_threads(useAllCPUs || omp_get_num_procs() == 1? omp_get_num_procs(): omp_get_num_procs()-1)
	{
		for (long iz = P->iz_start; iz < P->iz_end; iz++) {
			if (ctrlc_caught) break;

			substep1a(P);
			substep1b(P);
			substep2a(P);
			substep2b(P);
			applyMultiplier(P, iz, NULL);

#pragma omp master
			{
				if (iz + 1 < P->iz_end) swapEPointers(P, iz);
				updatePrecisePower(P);
			}
#pragma omp barrier
		}
	}
	if (P->E1 != E1 && P->E1 != P->Efinal) free(P->E1); // Part of the reason for checking this is to properly handle ctrl-c cases
	free(P->b);
	*outputPrecisePowerPtr = P->precisePower;
	return;
}
