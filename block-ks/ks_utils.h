// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <fstream>
#include <iostream>

//#include "types.h"
#include "ks_types.h"

typedef float ARMA_FPTYPE;

#ifdef DEBUG
#define PRINT(X) \
  { std::cerr << #X << "::\n" << X << std::endl; }
#define PRINT_NORM(X) \
  { std::cerr << "NORM(" << #X << "): " << std::sqrt(arma::dot(X,X)) << std::endl; }
#define PRINT_SIZE(X)                                                \
  {                                                                  \
    std::cerr << #X << ":: (" << X.n_rows << ", " << X.n_cols << ")" \
              << std::endl;                                          \
  }
#else
#define PRINT(X) \
  {}
#define PRINT_NORM(X) \
  {}
#define PRINT_SIZE(X) \
  {}
#endif

namespace utils {
  /* MGS implementation of column-pivoted rank-revealing QR factorization
   * using armadillo C++ linear algebra library
   * compute_qr()
   * A - MxN matrix
   * P - NxN permutation matrix
   * Q - MxR orthonormal matrix
   * R - NxR upper-triangular matrix
   * rank - optional, to obtain rank of matrix
   */
  void compute_qr(ARMA_FPMAT A, ARMA_FPMAT &P, ARMA_FPMAT &Q_out, ARMA_FPMAT &R_out,
                  uint64_t *rank = nullptr) {
    if (A.n_cols == A.n_rows && A.n_cols == 1) {
      P = {1};
      Q_out = {1};
      R_out = A;
      if (!rank)
        *rank = 1;
    }
#ifdef DEBUG
    arma::mat B = arma::conv_to<arma::mat>::from(A);
#endif
    arma::mat a = arma::conv_to<arma::mat>::from(A);
    // Allocate memory for Q and R
    arma::mat Q = arma::zeros<arma::mat>(A.n_rows, A.n_cols);
    arma::mat R = arma::zeros<arma::mat>(A.n_cols, A.n_cols);
    // idxs contains pivoting column indices
    ARMA_IDXVEC idxs(A.n_cols);
    // disc_idxs contains discarded/zeroed-out column indices
    ARMA_IDXVEC disc_idxs(A.n_cols);
    uint64_t    cur_rank = 0;
    for (uint64_t i = 0; i < A.n_cols; i++) {
      arma::vec v = a.col(i);
      ARMA_FPTYPE v_norm = std::sqrt(arma::dot(v, v));
      //ARMA_FPTYPE    v_norm = arma::norm(v);
      // Test if column is almost zero
      if (v_norm < 1e-6) {
        disc_idxs(i - cur_rank) = i;
        continue;
      }
      // Select as new pivot
      arma::vec q_v = v / v_norm;
      Q.col(cur_rank) = q_v;
      // MGS orthogonalization of q_v w.r.t remaining columns of A
      // Faster version using mat-vec products; using 1 count of DGKS correction
      arma::rowvec b = q_v.t() * a.tail_cols(A.n_cols - i);
      a.tail_cols(A.n_cols - i) = a.tail_cols(A.n_cols - i) - q_v * b;
      arma::rowvec c = q_v.t() * a.tail_cols(A.n_cols - i);
      a.tail_cols(A.n_cols - i) = a.tail_cols(A.n_cols - i) - q_v * c;
      R.row(cur_rank).tail(A.n_cols - i) = (b + c);

/* // Slower version using vec-vec products; without DGKS corrction
for(uint64_t j=i;j<A.n_cols;j++){
  r(i,j) = dot(q1, A.col(j));
  A.col(j) = A.col(j) - (r(i,j)*q1);
}*/

#ifdef DEBUG
      PRINT(i);
#endif
      idxs(cur_rank) = i;
      cur_rank++;
    }
    // Truncate Q & R
    Q = Q.head_cols(cur_rank);
    R = R.head_rows(cur_rank);
    Q_out = arma::conv_to<ARMA_FPMAT>::from(Q);
    R_out = arma::conv_to<ARMA_FPMAT>::from(R);

    // Generate permutation matrix P
    idxs.tail(A.n_cols - cur_rank) = disc_idxs.head(A.n_cols - cur_rank);
    P = arma::eye<ARMA_FPMAT>(A.n_cols, A.n_cols);
    P = P.cols(idxs);

    // Write rank
    if (rank != nullptr)
      *rank = cur_rank;

#ifdef DEBUG
    // PRINT(B);
    // PRINT(A);
    // PRINT(Q);
    // PRINT(arma::norm(Q.t() * Q - arma::eye<ARMA_FPMAT>(Q.n_cols, Q.n_cols)));
    // assert(arma::norm(Q.t() * Q - arma::eye<ARMA_FPMAT>(Q.n_cols, Q.n_cols)) <
    //      1e-4);
    // PRINT(R);
    // PRINT(P);
    PRINT(idxs.t());
    // PRINT(arma::norm(B - Q * R));
    // PRINT(arma::rank(B));
    // PRINT(arma::rank(R));
    PRINT(cur_rank);
    // assert(arma::norm(B - Q * R) < 1e-4);
#endif
  }

  /*
   * get_seed_eigs()
   * type - 0 : random eigenvalues in [0, 1]
   *      - 1 : Zipf distribution in [0, 1]
   *      - 2 : Zipf (sqrt) distribution in [0, 1]
   *      - 3 : evenly spaced in [0, 1]
   */
  void get_seed_eigs(uint64_t dim, uint64_t type, ARMA_FPVEC &evs) {
    switch (type) {
      case 0:
        evs = arma::randu<ARMA_FPVEC>(dim);
        break;
      case 1:
        evs = arma::zeros<ARMA_FPVEC>(dim);
        for (uint64_t i = 0; i < dim; i++) {
          evs(i) = 1.0f / (i + 1);
        }
        break;
      case 2:
        evs = arma::zeros<ARMA_FPVEC>(dim);
        for (uint64_t i = 0; i < dim; i++) {
          evs(i) = 1.0f / std::sqrt(i + 1);
        }
        break;
      case 3:
        evs = arma::zeros<ARMA_FPVEC>(dim);
        for (uint64_t i = 0; i < dim; i++) {
          evs(i) = (i + 1) / (dim * 1.0f);
        }
        break;
      default:
        std::runtime_error("Invalid matrix type");
    }

    return;
  }

   // BlockKs ProdOp class for Dense Armadillo matrices
  class ArmaMatProdOp {
    ARMA_FPMAT m;

   public:
    ArmaMatProdOp(ARMA_FPMAT mat) : m(mat) {
    }
    ARMA_FPMAT multiply(ARMA_FPMAT m_in) {
      return (this->m * m_in);
    }
    uint64_t rows() {
      return this->m.n_rows;
    }
    uint64_t cols() {
      return this->m.n_cols;
    }
  };
}  // namespace utils
