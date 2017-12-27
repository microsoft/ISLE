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
#define FPTYPE_asum	cblas_sasum
#define FPTYPE_gemm	cblas_sgemm
#define FPTYPE_gesvd	LAPACKE_sgesvd
#define FPTYPE_axpy	cblas_saxpy
#define FPTYPE_dot	cblas_sdot
#define FPTYPE_nrm2	cblas_snrm2
#define FPTYPE_blascopy cblas_scopy
#define FPTYPE_imin	cblas_isamin   
#define FPTYPE_dnscsr	mkl_sdnscsr
#define MatrixX MatrixXf
#define FPTYPE_gemv	cblas_sgemv
#define FPTYPE_symv	cblas_ssymv
#define FPTYPE_scal	cblas_sscal
#define FPTYPE_csrmm	mkl_scsrmm
#define FPTYPE_cscmm	mkl_scscmm
#define FPTYPE_omatcopy mkl_somatcopy
#define FPTYPE_csrcsc	mkl_scsrcsc
#define FPTYPE_csrgemv mkl_cspblas_scsrgemv
#elif FP_PRECISION == DOUBLE
#define FPTYPE	MAX DBL_MAX
#define FPTYPE_asum	cblas_dasum
#define FPTYPE_gemm	cblas_dgemm
#define FPTYPE_gesvd	LAPACKE_dgesvd
#define FPTYPE_axpy	cblas_daxpy
#define FPTYPE_dot		cblas_ddot
#define FPTYPE_nrm2	cblas_snrm2
#define FPTYPE_blascopy  cblas_dcopy
#define FPTYPE_imin	cblas_idamin
#define FPTYPE_dnscsr	mkl_ddnscsr
#define MatrixX MatrixXd
#define FPTYPE_gemv	cblas_dgemv
#define FPTYPE_symv	cblas_dsymv
#define FPTYPE_scal	cblas_dscal
#define FPTYPE_csrmm	mkl_dcsrmm
#define FPTYPE_cscmm	mkl_dcscmm
#define FPTYPE_omatcopy mkl_domatcopy
#define FPTYPE_csrcsc	mkl_dcsrcsc
#define FPTYPE_csrgemv mkl_cspblas_dcsrgemv
#endif
