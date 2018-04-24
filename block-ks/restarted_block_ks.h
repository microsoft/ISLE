// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#define ARMA_DONT_USE_WRAPPER
#include <cmath>
#include <csignal>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include "armadillo"
#include "ks_types.h"
#include "timer.h"
#include "ks_utils.h"

using ISLE::Timer;

template<class ProdOp>
class BlockKs {
  // Problem Parameters
  ProdOp *op;
  uint64_t    nev, ncv, maxit, blk_size, dim;
  ARMA_FPTYPE  tol;

  // Problem Status
  ARMA_FPMAT V, H;
  uint64_t   nconv;

  // Submodules
  void expand();
  void truncate();

 public:
  BlockKs(
      ProdOp *op,
      uint64_t nev, 
      uint64_t ncv, 
      uint64_t maxit, 
      uint64_t blk_size,
      ARMA_FPTYPE tol);

  void init();
  void compute();

  ARMA_FPVEC eigenvalues(uint64_t num_evs = 0) const 
  {
    ARMA_FPVEC evs = arma::diagvec(H);
    return ((num_evs == 0) ? evs.head(nev) : evs.head(num_evs));
  }

  ARMA_FPMAT eigenvectors(uint64_t num_evecs = 0) const 
  {
    return ((num_evecs == 0) ? V.head_cols(nev) : V.head_cols(num_evecs));
  }

  uint64_t num_converged() const 
  {
    return nconv;
  }
};

template<class ProdOp>
void BlockKs<ProdOp>::expand()
{
  Timer timer;

  ARMA_FPMAT Vk, Hk, Ck, F, P, Q, R;
  uint64_t   rk;

  // Preallocate V, H is small enough for dynamic expansion
  V = arma::join_rows(V, arma::zeros<ARMA_FPMAT>(V.n_rows, ncv - H.n_rows));

  while (H.n_rows < ncv) {
    std::cout << "Block #" << (H.n_rows / blk_size) << std::endl;
    timer.start();
    // Expand & orthogonalize
    PRINT_SIZE(H);
    Vk = V.cols(H.n_cols, H.n_rows - 1);
    PRINT_NORM(Vk);
    F = this->op->multiply(Vk);
    timer.next_time_secs("Ax(Vk)");
    ARMA_FPMAT V_temp(V.memptr(), V.n_rows, H.n_rows, false, true);
    Hk = F.t() * V_temp;
    Hk = Hk.t();
    F = F - (V_temp * Hk);
    for (uint64_t j = 0; j < 2; j++) {
      Ck = F.t() * V_temp;
      Ck = Ck.t();
      F = F - (V_temp * Ck);
      Hk = Hk + Ck;
    }
    timer.next_time_secs("Orthogonalization");

    H = arma::join_rows(H, Hk);
    H = arma::join_cols(H, arma::zeros<ARMA_FPMAT>(blk_size, H.n_cols));
    utils::compute_qr(F, P, Q, R, &rk);

    // Copy all 'rk' vectors to V
    for (uint64_t j = 0; j < rk; j++)
      V.unsafe_col(H.n_cols + j) = Q.col(j);
    PRINT_NORM(V.cols(H.n_cols, H.n_rows - 1));
    // Resize R to be upper triangular, concentrated in top left corner
    R = arma::join_cols(R, arma::zeros<ARMA_FPMAT>(blk_size - rk, R.n_cols));
    H.submat(H.n_rows - blk_size, H.n_cols - blk_size, H.n_rows - 1,
             H.n_cols - 1) = R;
    if (rk < blk_size) {
      std::cout << "rank deficient" << std::endl;
      // Number of required new basis vectors
      uint64_t   nTries = 0;
      ARMA_FPMAT P2, Q2, R2, H2, F2;
      uint64_t   nvecs = H.n_cols + rk, rk2;
      while (nvecs < H.n_rows && nTries < 100) {
        nTries++;
        F2 = arma::randu<ARMA_FPMAT>(V.n_rows, blk_size - rk);
        H2 = (arma::trans(V.head_cols(nvecs)) * F2);
        F2 = F2 - (V.head_cols(nvecs) * H2);  // Add 1 DGKS correction step
        H2 = (arma::trans(V.head_cols(nvecs)) *
              F2);  // Destructive write to H2 as we don't need it
        F2 = F2 - (V.head_cols(nvecs) * H2);
        utils::compute_qr(F2, P2, Q2, R2, &rk2);
        if (rk2 > 0) {
          // Add rk2 vectors to V
          for (uint64_t l = 0; l < rk2; l++)
            V.unsafe_col(nvecs + l) = Q2.col(l);
          nvecs += rk2;
        }
      }
      PRINT_NORM(V.t() * V - arma::eye<ARMA_FPMAT>(V.n_cols, V.n_cols));
      if (nTries == 100 && nvecs < H.n_rows)
        std::runtime_error(
            "Unable to find new starting basis for Arnoldi expansion.");
    }
    timer.next_time_secs("Orthonormalize");
  }
  PRINT_NORM(V.t() * V - arma::eye<ARMA_FPMAT>(V.n_cols, V.n_cols));
}

template<class ProdOp>
void BlockKs<ProdOp>::truncate()
{
  PRINT_SIZE(V);
  PRINT_SIZE(H);
  PRINT(nconv);

  ARMA_FPMAT vH, subH;
  ARMA_FPVEC    eH;
  ARMA_IDXVEC idxeH;

  // Compute EVD of `H` & rearrange
  subH = H.submat(nconv, nconv, H.n_cols - 1, H.n_cols - 1);
  bool success = arma::eig_sym(eH, vH, subH);
  
  PRINT_NORM((vH.t() * subH * vH) - arma::diagmat(eH));
  PRINT_NORM((vH.t() * vH) - arma::eye<ARMA_FPMAT>(vH.n_cols, vH.n_cols));

  if (!success)
    throw std::runtime_error("evd(H) failed");

  idxeH = sort_index(eH, "descend");
  eH = eH.elem(idxeH);
  vH = vH.cols(idxeH);

  // Transform & truncate V
  ARMA_FPMAT new_starts = V.tail_cols(blk_size);
  ARMA_FPMAT preserve = V.head_cols(nconv);
  V = V.cols(nconv, V.n_cols - blk_size - 1);
  V = V * vH.head_cols(nev - nconv);
  V = arma::join_rows(preserve, V);
  V = arma::join_rows(V, new_starts);

  // Transform H
  H.submat(nconv, nconv, nev - 1, nev - 1) = diagmat(eH.head(nev - nconv));
  H.submat(nev, nconv, nev + blk_size - 1, H.n_cols - 1) =
      H.submat(H.n_rows - blk_size, H.n_cols - blk_size, H.n_rows - 1,
               H.n_cols - 1) *
      vH.tail_rows(blk_size);

  if (nconv > 0)
    H.submat(0, nconv, nconv - 1, H.n_cols - 1) =
        H.submat(0, nconv, nconv - 1, H.n_cols - 1) * vH;

  // Truncate H
  H = H.head_cols(nev);
  H = H.head_rows(nev + blk_size);

  PRINT_NORM(V.t() * V - arma::eye<ARMA_FPMAT>(V.n_cols, V.n_cols));
}

template<class ProdOp>
BlockKs<ProdOp>::BlockKs(
    ProdOp *op, uint64_t nev, uint64_t ncv, uint64_t maxit,
    uint64_t blk_size, ARMA_FPTYPE tol) 
{
  this->op = op;
  this->nev = nev;
  this->ncv = ncv;
  this->maxit = maxit;
  this->blk_size = blk_size;
  this->tol = tol;
  this->dim = op->rows();
}

template<class ProdOp>
void BlockKs<ProdOp>::init() 
{
  ARMA_FPMAT V_1, P, Q, R, C, alpha, blk, r1;
  ARMA_FPRVEC norms;
  uint64_t   rank;
  ARMA_IDXVEC start;

  // Start with random basis
  V = arma::randu<ARMA_FPMAT>(dim, blk_size);
  utils::compute_qr(V, P, Q, R, &rank);
  // Re-initialize till we obtain a random orthonomal basis
  while (rank < blk_size) {
    V = arma::randu<ARMA_FPMAT>(dim, blk_size);
    utils::compute_qr(V, P, Q, R, &rank);
  }

  // 1-step Block Arnoldi expansion (with 1-step DGKS correction)
  V = Q;

  std::cout << "KS: block0 computed" << std::endl;
  V_1 = this->op->multiply(V);
  std::cout << "KS: A*block0 computed" << std::endl;
  H = (V.t() * V_1);
  V_1 = V_1 - (V * H);
  C = (V.t() * V_1);
  H = H + C;
  V_1 = V_1 - (V * C);
  utils::compute_qr(V_1, P, Q, R, &rank);
  std::cout << "KS:qr(block1)" << std::endl;
  R = arma::join_cols(R, arma::zeros<ARMA_FPMAT>(blk_size - rank, R.n_cols));
  H = arma::join_cols(H, R);
  V = arma::join_rows(V, Q);

  // Fix any rank deficiency
  if (rank < blk_size) {
    uint64_t   nTries = 0, deficiency = blk_size - rank, rank2;
    ARMA_FPMAT P2, Q2, R2, H2, F2;
    while (deficiency > 0 && nTries < 100) {
      nTries++;
      F2 = arma::randu<ARMA_FPMAT>(V.n_rows, blk_size - rank);
      H2 = (arma::trans(V) * F2);
      F2 = F2 - (V * H2);  // Add 1 DGKS correction step
      H2 =
          (arma::trans(V) * F2);  // Destructive write to H2 as we don't need it
      F2 = F2 - (V * H2);
      utils::compute_qr(F2, P2, Q2, R2, &rank2);
      if (rank2 > 0) {
        V = arma::join_rows(V, Q2);
        deficiency -= rank2;
      }
    }
    if (nTries == 100 && deficiency > 0)
      std::runtime_error(
          "Unable to find new starting basis for Arnoldi expansion.");
  }
}

template<class ProdOp>
void BlockKs<ProdOp>::compute()
{
  ARMA_FPMAT residual_blk;
  ARMA_FPRVEC norms;
  ARMA_IDXVEC idxs;
  uint64_t   n_restarts = 0;
  this->nconv = 0;

  // Expand to a `ncv` size decomposition
  this->expand();

  while (n_restarts < this->maxit) {
    // Truncate to a size `nev` decomposition
    this->truncate();

    // Compute residuals & determine nconv
    residual_blk = H.tail_rows(blk_size);
    norms = arma::sum(arma::square(residual_blk), 0);
    ARMA_FPVEC evs = arma::diagvec(H);
    evs = evs.head(norms.n_cols);
    norms = norms / evs.t();
    std::cout << "Norms :: \n" << norms << "\n";
    PRINT(norms);
    idxs = find(norms >= tol, 1, "first");

    // Convergence checks
    if (idxs.n_cols == 0 || idxs.n_rows == 0) {
      nconv = norms.n_cols;
      break;
    } else {
      nconv = idxs(0);
    }

    std::cout << "nconv = " << nconv << std::endl;
    std::cout << "Restart #" << ++n_restarts << std::endl;

    // Expand to a size `ncv` decomposition
    this->expand();
  }

  // Exceeded `maxit` restarts
  if (n_restarts == this->maxit) {
    // Compute residuals & determine nconv
    residual_blk = H.tail_rows(blk_size);
    norms = arma::sum(arma::square(residual_blk), 0);
    PRINT(norms);
    idxs = find(norms >= tol, 1, "first");

    // Convergence checks
    if (idxs.n_cols == 0 || idxs.n_rows == 0) {
      nconv = norms.n_cols;
    } else {
      nconv = idxs(0);
    }
  }
  nconv = (nconv >= nev ? nev : nconv);

  std::printf("Completed with %llu restarts, nconv = %llu\n", n_restarts,
              nconv);
}
