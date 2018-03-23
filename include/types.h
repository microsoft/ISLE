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

namespace ISLE {
#define MKL_USE_DNSCSR
  //#define MKL_LP64
  //#define EIGEN_USE_MKL_ALL

  // word_id_t and doc_id_t have to be unsigned
  typedef MKL_UINT word_id_t;
  typedef MKL_UINT doc_id_t;
  typedef MKL_INT  offset_t;

  // Try to eliminate this out of code.
  typedef uint32_t count_t;

#define FP_PRECISION SINGLE

#if FP_PRECISION == SINGLE

  typedef float FPTYPE;
#define FP_MAX FLT_MAX
#define FPasum cblas_sasum
#define FPgemm cblas_sgemm
#define FPgesvd LAPACKE_sgesvd
#define FPaxpy cblas_saxpy
#define FPdot cblas_sdot
#define FPnrm2 cblas_snrm2
#define FPblascopy cblas_scopy
#define FPimin cblas_isamin
#define FPdnscsr mkl_sdnscsr
#define MatrixX MatrixXf
#define FPgemv cblas_sgemv
#define FPsymv cblas_ssymv
#define FPscal cblas_sscal
#define FPcsrmm mkl_scsrmm
#define FPcscmm mkl_scscmm
#define FPomatcopy mkl_somatcopy
#define FPcsrcsc mkl_scsrcsc
#define FPcsrgemv mkl_cspblas_scsrgemv

#elif FP_PRECISION == DOUBLE

  typedef double FPTYPE;
#define FP_MAX DBL_MAX
#define FPasum cblas_dasum
#define FPgemm cblas_dgemm
#define FPgesvd LAPACKE_dgesvd
#define FPaxpy cblas_daxpy
#define FPdot cblas_ddot
#define FPnrm2 cblas_snrm2
#define FPblascopy cblas_dcopy
#define FPimin cblas_idamin
#define FPdnscsr mkl_ddnscsr
#define MatrixX MatrixXd
#define FPgemv cblas_dgemv
#define FPsymv cblas_dsymv
#define FPscal cblas_dscal
#define FPcsrmm mkl_dcsrmm
#define FPcscmm mkl_dcscmm
#define FPomatcopy mkl_domatcopy
#define FPcsrcsc mkl_dcsrcsc
#define FPcsrgemv mkl_cspblas_dcsrgemv
#endif

#if USE_INT_NORMALIZED_COUNTS
  typedef count_t A_TYPE;
#else
  typedef FPTYPE A_TYPE;
#endif
};