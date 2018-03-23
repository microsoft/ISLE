// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "armadillo"
#include "mkl.h"

// Matrix primitives
typedef arma::fmat       ARMA_FPMAT;
typedef arma::cx_fmat    ARMA_FPCXMAT;
typedef arma::fvec       ARMA_FPVEC;
typedef arma::cx_fvec    ARMA_FPCXVEC;
typedef arma::frowvec    ARMA_FPRVEC;
typedef arma::cx_frowvec ARMA_FPCXRVEC;
typedef arma::uvec       ARMA_IDXVEC;