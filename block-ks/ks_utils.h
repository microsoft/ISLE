// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <fstream>
#include <iostream>
#include "ks_types.h"

#ifdef DEBUG
#define PRINT(X) \
  { std::cerr << #X << "::\n" << X << std::endl; }
#define PRINT_SIZE(X)                                                \
  {                                                                  \
    std::cerr << #X << ":: (" << X.n_rows << ", " << X.n_cols << ")" \
              << std::endl;                                          \
  }
#else
#define PRINT(X) \
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
    IDXVEC idxs(A.n_cols);
    // disc_idxs contains discarded/zeroed-out column indices
    IDXVEC disc_idxs(A.n_cols);
    uint64_t    cur_rank = 0;
    for (uint64_t i = 0; i < A.n_cols; i++) {
      arma::vec v = a.col(i);
      FPTYPE    v_norm = arma::norm(v);
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
    PRINT(arma::norm(Q.t() * Q - arma::eye<ARMA_FPMAT>(Q.n_cols, Q.n_cols)));
    assert(arma::norm(Q.t() * Q - arma::eye<ARMA_FPMAT>(Q.n_cols, Q.n_cols)) <
           1e-4);
    // PRINT(R);
    // PRINT(P);
    PRINT(idxs.t());
    PRINT(arma::norm(B - Q * R));
    // PRINT(arma::rank(B));
    // PRINT(arma::rank(R));
    PRINT(cur_rank);
    assert(arma::norm(B - Q * R) < 1e-4);
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

  // Assume n_rows == n_cols; Make sure tsvd_matrix_reader returns it that way.
  class TSVDMatrix {
    MKL_INT *ia, *ja, *ia2, *ja2;
    FPTYPE * a;
    uint64_t     n_dims, n_temp_dim, n_nzs;
    uint64_t     n_rows, n_cols;
    uint64_t     n_mms, n_mvs;
    bool     trans_first;  // we compute eigs(A*A^T) if [true] else eigs(A^T*A)
   public:
    TSVDMatrix(std::string &filename, uint64_t n_rows, uint64_t n_cols, uint64_t n_nzs) {
      n_dims = std::min(n_rows, n_cols);
      n_temp_dim = std::max(n_rows, n_cols);
      this->n_rows = n_rows;
      this->n_cols = n_cols;
      if (n_cols > n_rows)
        trans_first = true;
      else
        trans_first = false;
      ia = new MKL_INT[n_rows + 1];
      std::memset(ia, 0, (n_dims + 1) * sizeof(MKL_INT));
      ia2 = new MKL_INT[n_rows + 1];
      std::memset(ia, 0, (n_dims + 1) * sizeof(MKL_INT));
      ja = new MKL_INT[n_nzs];
      std::memset(ja, 0, n_nzs * sizeof(MKL_INT));
      ja2 = new MKL_INT[n_nzs];
      std::memset(ja, 0, n_nzs * sizeof(MKL_INT));
      a = new FPTYPE[n_nzs];
      std::memset(a, 0, n_nzs * sizeof(FPTYPE));
      n_nzs = n_nzs;
      tsvd_matrix_reader(filename, n_rows, n_nzs, ia, ja, a);
      for (uint64_t i = 0; i <= n_rows; i++)
        ia2[i] = ia[i] - 1;
      for (uint64_t i = 0; i < n_nzs; i++)
        ja2[i] = ja[i] - 1;
      std::cout << "ia[n_dims] :: " << ia[n_rows] << std::endl;
      std::cout << "nnzs :: " << n_nzs << std::endl;
      assert((uint64_t) ia[n_rows] == (n_nzs + 1));
      n_mms = 0;
      n_mvs = 0;
    }

    // Computes (A^T*(A*m_in)) if trans_first is false
    // Computes (A*(A^T*m_in)) if trans_first is true
    ARMA_FPMAT multiply(ARMA_FPMAT m_in) {
      ARMA_FPMAT m_temp(n_temp_dim, m_in.n_cols);
      ARMA_FPMAT m_out(n_dims, m_in.n_cols);
      m_temp.zeros();
      m_out.zeros();
      // Only start and end elements are useful
      // matdescra[3] = 'F' -> 1-based indexing, dense matrix in column major
      // form
      const char matdescra[] = {'G', 'L', 'N', 'F'};
      MKL_INT    m, n, k;
      m = (MKL_INT) n_rows;
      n = (MKL_INT) m_in.n_cols;
      k = (MKL_INT) n_cols;
      FPTYPE alpha = 1.0, beta = 0.0;
      if (trans_first) {
        // Perform A^T*m_in using MKL
        mkl_scsrmm("T", &m, &n, &k, &alpha, matdescra, a, ja, ia, ia + 1,
                   m_in.memptr(), &m, &beta, m_temp.memptr(), &k);
        // Perform A*m_temp using MKL
        mkl_scsrmm("N", &m, &n, &k, &alpha, matdescra, a, ja, ia, ia + 1,
                   m_temp.memptr(), &k, &beta, m_out.memptr(), &m);

      } else {
        // Perform A*m_in using MKL
        mkl_scsrmm("N", &m, &n, &k, &alpha, matdescra, a, ja, ia, ia + 1,
                   m_in.memptr(), &k, &beta, m_temp.memptr(), &m);
        // Perform A^T*m_temp using MKL
        mkl_scsrmm("T", &m, &n, &k, &alpha, matdescra, a, ja, ia, ia + 1,
                   m_temp.memptr(), &m, &beta, m_out.memptr(), &k);
      }
      n_mms++;
      return m_out;
    }

    // Spectra op function
    // NOTE :: WORKS only when n_rows > n_cols
    // If n_cols > n_rows, ia2 array must be modified to become a n_cols X
    // n_cols matrix.
    void perform_op(FPTYPE *v_in, FPTYPE *v_out) {
      MKL_INT m = (MKL_INT) n_rows;
      // auto destructed after perform_op()
      ARMA_FPVEC v_temp(n_rows);
      ARMA_FPVEC v_in_extended(n_rows);
      memset(v_in_extended.memptr(), 0, n_rows * sizeof(FPTYPE));
      memcpy(v_in_extended.memptr(), v_in, n_dims * sizeof(FPTYPE));
      ARMA_FPVEC v_temp2(n_rows);
      if (trans_first) {
        // Perform A^T*v_temp using MKL
        mkl_cspblas_scsrgemv("T", &m, a, ia2, ja2, v_in_extended.memptr(),
                             v_temp.memptr());
        // Perform A*v using MKL
        mkl_cspblas_scsrgemv("N", &m, a, ia2, ja2, v_temp.memptr(),
                             v_temp2.memptr());
      } else {
        // Perform A*v using MKL
        mkl_cspblas_scsrgemv("N", &m, a, ia2, ja2, v_in_extended.memptr(),
                             v_temp.memptr());
        // Perform A^T*v_temp using MKL
        mkl_cspblas_scsrgemv("T", &m, a, ia2, ja2, v_temp.memptr(),
                             v_temp2.memptr());
      }
      memcpy(v_out, v_temp2.memptr(), n_dims * sizeof(FPTYPE));
      n_mvs++;
    }

    ~TSVDMatrix() {
      delete[] ia;
      delete[] ja;
      delete[] a;
    }

    // Spectra functions
    MKL_INT rows() {
      return (MKL_INT) n_dims;
    }
    MKL_INT cols() {
      return (MKL_INT) n_dims;
    }

    // Status functions
    uint64_t get_num_matprods() {
      return n_mms;
    }
    uint64_t get_num_matvecs() {
      return n_mvs;
    }
  };

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
