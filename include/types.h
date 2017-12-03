// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstdint>
#include <type_traits>

#include <mkl.h>

#include "hyperparams.h"

#ifdef LINUX
#include "float.h"
#endif


#define MKL_USE_DNSCSR

//#define MKL_LP64
//#define EIGEN_USE_MKL_ALL

// vocabSz_t and docsSz_t have to be unsigned
typedef MKL_UINT vocabSz_t;
typedef MKL_UINT docsSz_t;

typedef MKL_INT offset_t;

// Try to eliminate this out of code.
typedef uint32_t count_t;

#define FP_PRECISION SINGLE

#if FP_PRECISION == SINGLE
typedef float FPTYPE;
#elif FP_PRECISION == DOUBLE
typedef double FPTYPE;
#endif

#if USE_INT_NORMALIZED_COUNTS 
typedef count_t A_TYPE;
#else
typedef FPTYPE A_TYPE;
#endif


#if FP_PRECISION == SINGLE
#define FPTYPE_MAX FLT_MAX
#define asum	cblas_sasum
#define gemm	cblas_sgemm
#define gesvd	LAPACKE_sgesvd
#define axpy	cblas_saxpy
#define dot	cblas_sdot
#define nrm2	cblas_snrm2
#define blascopy cblas_scopy
#define imin	cblas_isamin   
#define dnscsr	mkl_sdnscsr
#define MatrixX MatrixXf
#define gemv	cblas_sgemv
#define symv	cblas_ssymv
#define scal	cblas_sscal
#define csrmm	mkl_scsrmm
#define cscmm	mkl_scscmm
#define omatcopy mkl_somatcopy
#define csrcsc	mkl_scsrcsc
#define csrgemv mkl_cspblas_scsrgemv
#elif FP_PRECISION == DOUBLE
#define FPTYPE	MAX DBL_MAX
#define asum	cblas_dasum
#define gemm	cblas_dgemm
#define gesvd	LAPACKE_dgesvd
#define axpy	cblas_daxpy
#define dot		cblas_ddot
#define nrm2	cblas_snrm2
#define blascopy  cblas_dcopy
#define imin	cblas_idamin
#define dnscsr	mkl_ddnscsr
#define MatrixX MatrixXd
#define gemv	cblas_dgemv
#define symv	cblas_dsymv
#define scal	cblas_dscal
#define csrmm	mkl_dcsrmm
#define cscmm	mkl_dcscmm
#define omatcopy mkl_domatcopy
#define csrcsc	mkl_dcsrcsc
#define csrgemv mkl_cspblas_dcsrgemv
#endif
