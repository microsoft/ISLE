#pragma once
#include "types.h"
#include "parallel.h"
#include "blas-on-flash/include/pointers/pointer.h"
#include "blas-on-flash/include/lib_funcs.h"
#include <cassert>

namespace Kmeans {
  using namespace ISLE;
  template<typename FPTYPE>
  void multiply_with(offset_t *offsets_CSC, flash::flash_ptr<word_id_t> shifted_rows_CSC_fptr,
                    flash::flash_ptr<FPTYPE> shifted_vals_CSC_fptr, const doc_id_t doc_blk_size,
                    const word_id_t vocab_size, const FPTYPE *const in,
                    FPTYPE *const out, const MKL_INT n_cols) {
    GLOG_DEBUG("here");
    // create shifted copy of offsets array
    MKL_INT *shifted_offsets_CSC = new MKL_INT[doc_blk_size + 1];
    for (doc_id_t d = 0; d <= doc_blk_size; ++d) {
      shifted_offsets_CSC[d] = offsets_CSC[d] - offsets_CSC[0];
    }

    // static_assert(sizeof(MKL_INT) == sizeof(offset_t));
    // static_assert(sizeof(word_id_t) == sizeof(MKL_INT));
    // static_assert(sizeof(offset_t) == sizeof(MKL_INT));

    uint64_t nnzs = shifted_offsets_CSC[doc_blk_size];
    word_id_t *shifted_rows_CSC = new word_id_t[nnzs];
    FPTYPE *shifted_vals_CSC = new FPTYPE[nnzs];

    flash::read_sync(shifted_rows_CSC, shifted_rows_CSC_fptr, nnzs);
    flash::read_sync(shifted_vals_CSC, shifted_vals_CSC_fptr, nnzs);

    const char transa = 'N';
    const MKL_INT m = doc_blk_size;
    const MKL_INT n = n_cols;
    const MKL_INT k = vocab_size;
    const char matdescra[6] = {'G', 0, 0, 'C', 0, 0};
    FPTYPE alpha = 1.0;
    FPTYPE beta = 0.0;

    assert(sizeof(MKL_INT) == sizeof(offset_t));
    assert(sizeof(word_id_t) == sizeof(MKL_INT));
    assert(sizeof(offset_t) == sizeof(MKL_INT));

    FPcsrmm(&transa, &m, &n, &k, &alpha, matdescra,
            shifted_vals_CSC, (const MKL_INT *)shifted_rows_CSC,
            (const MKL_INT *)shifted_offsets_CSC, (const MKL_INT *)(shifted_offsets_CSC + 1),
            in, &n, &beta, out, &n);

    delete[] shifted_offsets_CSC;
    delete[] shifted_rows_CSC;
    delete[] shifted_vals_CSC;
  }

  template <typename FPTYPE>
  void distsq_projected_docs_to_projected_centers(
      const FPTYPE *const projected_centers_tr, const FPTYPE *const projected_centers_l2sq,
      offset_t *offsets_CSC, flash::flash_ptr<word_id_t> shifted_rows_CSC_fptr,
      flash::flash_ptr<FPTYPE> shifted_vals_CSC_fptr, const doc_id_t doc_blk_size,
      const FPTYPE *const projected_docs_l2sq, FPTYPE *projected_dist_matrix,
      const word_id_t dim_size, doc_id_t num_centers, FPTYPE *U_rowmajor, uint64_t U_rows, uint64_t U_cols)
  {
    GLOG_DEBUG("here");
    FPTYPE *ones_vec = new FPTYPE[std::max(doc_blk_size, num_centers)];
    std::fill_n(ones_vec, std::max(doc_blk_size, num_centers), (FPTYPE)1.0);

    const char transa = 'N';
    const MKL_INT m = doc_blk_size;
    const MKL_INT n = num_centers;
    const MKL_INT k = U_cols;
    const char matdescra[6] = {'G', 0, 0, 'C', 0, 0};

    FPTYPE *UUTrC = new FPTYPE[U_rows * num_centers];
    FPgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
           U_rows, num_centers, U_cols, (FPTYPE)-2.0,
           U_rowmajor, U_cols, projected_centers_tr, num_centers,
           (FPTYPE)0.0, UUTrC, num_centers);

    multiply_with(offsets_CSC, shifted_rows_CSC_fptr,
                  shifted_vals_CSC_fptr, doc_blk_size,
                  dim_size, UUTrC, projected_dist_matrix, num_centers);
    delete[] UUTrC;

    FPgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
           m, n, 1, (FPTYPE)1.0,
           ones_vec, m, projected_centers_l2sq, n,
           (FPTYPE)1.0, projected_dist_matrix, n);

    FPgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
           m, n, 1, (FPTYPE)1.0,
           projected_docs_l2sq, m, ones_vec, n,
           (FPTYPE)1.0, projected_dist_matrix, n);

    delete[] ones_vec;
  }

  template <class FPTYPE>
  void projected_closest_centers(
      const FPTYPE *const projected_centers_tr, const FPTYPE *const projected_centers_l2sq,
      offset_t *offsets_CSC, flash::flash_ptr<word_id_t> shifted_rows_CSC_fptr,
      flash::flash_ptr<FPTYPE> shifted_vals_CSC_fptr, const doc_id_t doc_blk_size,
      const doc_id_t num_centers, const word_id_t vocab_size, FPTYPE *U_rowmajor, uint64_t U_rows, uint64_t U_cols,
      const FPTYPE *const projected_docs_l2sq, doc_id_t *center_index, FPTYPE *const projected_dist_matrix) {
    GLOG_DEBUG("here");
    // distsq_projected_docs_to_projected_centers
    {
      FPTYPE *ones_vec = new FPTYPE[std::max(doc_blk_size, num_centers)];
      std::fill_n(ones_vec, std::max(doc_blk_size, num_centers), (FPTYPE)1.0);

      const char transa = 'N';
      const MKL_INT m = doc_blk_size;
      const MKL_INT n = num_centers;
      const MKL_INT k = U_cols;
      const char matdescra[6] = {'G', 0, 0, 'C', 0, 0};

      FPTYPE *UUTrC = new FPTYPE[U_rows * num_centers];
      FPgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
             U_rows, num_centers, U_cols, (FPTYPE)-2.0,
             U_rowmajor, U_cols, projected_centers_tr, num_centers,
             (FPTYPE)0.0, UUTrC, num_centers);

      // multiply_with
      {
        // create shifted copy of offsets array
        MKL_INT *shifted_offsets_CSC = new MKL_INT[doc_blk_size + 1];
        for (doc_id_t d = 0; d <= doc_blk_size; ++d) {
          shifted_offsets_CSC[d] = offsets_CSC[d] - offsets_CSC[0];
        }

        uint64_t nnzs = shifted_offsets_CSC[doc_blk_size];
        word_id_t *shifted_rows_CSC = new word_id_t[nnzs];
        FPTYPE *shifted_vals_CSC = new FPTYPE[nnzs];

        flash::read_sync(shifted_rows_CSC, shifted_rows_CSC_fptr, nnzs);
        flash::read_sync(shifted_vals_CSC, shifted_vals_CSC_fptr, nnzs);

        const char transa = 'N';
        const MKL_INT m = doc_blk_size;
        const MKL_INT n = num_centers;
        const MKL_INT k = vocab_size;
        const char matdescra[6] = {'G', 0, 0, 'C', 0, 0};
        FPTYPE alpha = 1.0;
        FPTYPE beta = 0.0;

        assert(sizeof(MKL_INT) == sizeof(offset_t));
        assert(sizeof(word_id_t) == sizeof(MKL_INT));
        assert(sizeof(offset_t) == sizeof(MKL_INT));

        FPcsrmm(&transa, &m, &n, &k, &alpha, matdescra,
                shifted_vals_CSC, (const MKL_INT *)shifted_rows_CSC,
                (const MKL_INT *)shifted_offsets_CSC, (const MKL_INT *)(shifted_offsets_CSC + 1),
                UUTrC, &n, &beta, projected_dist_matrix, &n);

        delete[] shifted_offsets_CSC;
        delete[] shifted_rows_CSC;
        delete[] shifted_vals_CSC;
      }

      delete[] UUTrC;

      FPgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
             m, n, 1, (FPTYPE)1.0,
             ones_vec, m, projected_centers_l2sq, n,
             (FPTYPE)1.0, projected_dist_matrix, n);

      FPgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
             m, n, 1, (FPTYPE)1.0,
             projected_docs_l2sq, m, ones_vec, n,
             (FPTYPE)1.0, projected_dist_matrix, n);

      delete[] ones_vec;
    }

    pfor_static_131072(int64_t d = 0; d < doc_blk_size; ++d)
        center_index[d] = (doc_id_t)FPimin(num_centers, projected_dist_matrix + (size_t)d * (size_t)num_centers, 1);
  }

  template <class FPTYPE>
  void update_min_distsq_to_projected_centers(
      const FPTYPE *const projected_centers,
      const FPTYPE *const projected_docs_l2sq,
      offset_t *offsets_CSC, flash::flash_ptr<word_id_t> shifted_rows_CSC_fptr,
      flash::flash_ptr<FPTYPE> shifted_vals_CSC_fptr,
      FPTYPE *U_rowmajor, uint64_t U_rows, uint64_t U_cols,
      FPTYPE *min_dist, FPTYPE *projected_dist, // pre-allocated for `doc_blk_size` or NULL
      const uint64_t dim, const uint64_t doc_blk_size, const uint64_t num_centers) {
    GLOG_DEBUG("here");
    bool dist_alloc = false;
    if (projected_dist == NULL) {
      projected_dist = new FPTYPE[doc_blk_size * num_centers];
      dist_alloc = true;
    }

    FPTYPE *projected_center_l2sq = new FPTYPE[num_centers];
    for (uint64_t c = 0; c < num_centers; ++c){
      projected_center_l2sq[c] = FPdot(dim,
                                       projected_centers + (c * dim), 1,
                                       projected_centers + (c * dim), 1);
    }

    // compute transpose of `projected_centers`
    FPTYPE *projected_centers_tr = new FPTYPE[num_centers * U_cols];
    FPomatcopy('C', 'T', U_cols, num_centers, 1.0f, projected_centers,
               U_cols, projected_centers_tr, num_centers);

    Kmeans::distsq_projected_docs_to_projected_centers(projected_centers_tr, projected_center_l2sq,
                                                       offsets_CSC, shifted_rows_CSC_fptr, shifted_vals_CSC_fptr,
                                                       doc_blk_size, projected_docs_l2sq, projected_dist,
                                                       dim, num_centers, U_rowmajor, U_rows, U_cols);

    pfor_static_131072(uint64_t d = 0; d < doc_blk_size; ++d) {
      if (num_centers == 1) {
        // Round about for small negative distances
        projected_dist[d] = std::max(projected_dist[d], (FPTYPE)0.0);
        min_dist[d] = std::min(min_dist[d], projected_dist[d]);
      } else {
        for (uint64_t c = 0; c < num_centers; ++c) {
          uint64_t pos = (uint64_t)c + (uint64_t)d * (uint64_t)num_centers;
          projected_dist[pos] = std::max(projected_dist[pos], (FPTYPE)0.0);
          min_dist[d] = std::min(min_dist[d], projected_dist[pos]);
        }
      }
    }

    delete[] projected_center_l2sq;
    delete[] projected_centers_tr;

    if (dist_alloc) {
      delete[] projected_dist;
    }
  }

  // compute L2^2 of each col in CSC matrix
  void compute_col_l2sq(MKL_INT *offsets_CSC, flash::flash_ptr<FPTYPE> vals_CSC_fptr,
                        FPTYPE *col_l2sq, MKL_INT n_cols, MKL_INT col_blk_size);

  void projected_closest_centers_full(
      const FPTYPE *const projected_centers_tr, const FPTYPE *const projected_centers_l2sq,
      offset_t *offsets_CSC, flash::flash_ptr<word_id_t> rows_CSC_fptr,
      flash::flash_ptr<FPTYPE> vals_CSC_fptr, const doc_id_t num_docs,
      const doc_id_t doc_blk_size, const doc_id_t num_centers,
      const word_id_t vocab_size, FPTYPE *U_rowmajor, uint64_t U_rows, uint64_t U_cols,
      const FPTYPE *const projected_docs_l2sq, doc_id_t *center_index);

  void projected_compute_new_centers_full(
      offset_t *offsets_CSC, flash::flash_ptr<word_id_t> rows_CSC_fptr,
      flash::flash_ptr<FPTYPE> vals_CSC_fptr, const doc_id_t num_docs,
      const doc_id_t doc_blk_size, const doc_id_t num_centers,
      const word_id_t vocab_size, FPTYPE *U_rowmajor, uint64_t U_rows, uint64_t U_cols,
      doc_id_t *center_index, std::vector<doc_id_t> *closest_docs,
      FPTYPE *projected_centers, std::vector<size_t> &cluster_sizes);

} // namespace kmeans