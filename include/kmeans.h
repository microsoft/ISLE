#pragma once
#include "types.h"
#include "blas-on-flash/include/pointers/pointer.h"
#include "blas-on-flash/include/lib_funcs.h"

namespace Kmeans {
  using namespace ISLE;
  template<typename FPTYPE>
  void multiply_with(offset_t *offsets_CSC, flash::flash_ptr<word_id_t> shifted_rows_CSC_fptr,
                    flash::flash_ptr<FPTYPE> shifted_vals_CSC_fptr, const doc_id_t doc_blk_size,
                    const word_id_t vocab_size, const FPTYPE *const in,
                    FPTYPE *const out, const MKL_INT n_cols) {
    
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
      const doc_id_t num_centers, const word_id_t vocab_size, FPTYPE* U_rowmajor, uint64_t U_rows, uint64_t U_cols,
      const FPTYPE *const projected_docs_l2sq, doc_id_t *center_index, FPTYPE *const projected_dist_matrix) {
    Kmeans::distsq_projected_docs_to_projected_centers(projected_centers_tr, projected_centers_l2sq, offsets_CSC,
                                               shifted_rows_CSC_fptr, shifted_vals_CSC_fptr, doc_blk_size,
                                               projected_docs_l2sq, projected_dist_matrix, vocab_size, num_centers,
                                               U_rowmajor, U_rows, U_cols);
    pfor_static_131072(int64_t d = 0; d < doc_blk_size; ++d)
        center_index[d] = (doc_id_t)FPimin(num_centers, projected_dist_matrix + (size_t)d * (size_t)num_centers, 1);
  }

  class ProjClosestCentersTask : public BaseTask{
    FPTYPE *projected_centers_tr = nullptr;
    FPTYPE *projected_centers_l2sq = nullptr;
    offset_t *shifted_offsets_CSC = nullptr;
    const FPTYPE *const projected_docs_l2sq = nullptr;
    doc_id_t *center_index = nullptr;

    flash::flash_ptr<word_id_t> shifted_rows_CSC_fptr;
    flash::flash_ptr<FPTYPE> shifted_vals_CSC_fptr;
    
    const doc_id_t doc_blk_size;
    const word_id_t vocab_size;
    doc_id_t num_centers;
    const FPTYPE *UUTrC = nullptr;
    uint64_t U_rows, U_cols;

  public:
    ProjClosestCentersTask(const FPTYPE *const projected_docs_l2sq, doc_id_t *center_index,
                           offset_t *offsets_CSC,
                           flash::flash_ptr<word_id_t> shifted_rows_CSC_fptr,
                           flash::flash_ptr<FPTYPE> shifted_vals_CSC_fptr,
                           const doc_id_t doc_blk_size,
                           const word_id_t vocab_size,
                           uint64_t U_rows, uint64_t U_cols) : projected_docs_l2sq(projected_docs_l2sq),
                                                               center_index(center_index),
                                                               shifted_rows_CSC_fptr(shifted_rows_CSC_fptr),
                                                               shifted_vals_CSC_fptr(shifted_vals_CSC_fptr),
                                                               doc_blk_size(doc_blk_size),
                                                               vocab_size(vocab_size),
                                                               U_rows(U_rows),
                                                               U_cols(U_cols)
    {
      this->shifted_offsets_CSC = new MKL_INT[doc_blk_size + 1];
      for (doc_id_t d = 0; d <= doc_blk_size; ++d)
      {
        shifted_offsets_CSC[d] = offsets_CSC[d] - offsets_CSC[0];
      }
      uint64_t nnzs = shifted_offsets_CSC[doc_blk_size];

      flash::StrideInfo sinfo = {1, 1, 1};
      sinfo.len_per_stride = nnzs * sizeof(word_id_t);
      this->add_read(this->shifted_rows_CSC_fptr, sinfo);
      sinfo.len_per_stride = nnzs * sizeof(FPTYPE);
      this->add_read(this->shifted_vals_CSC_fptr, sinfo);
    }

    ~ProjClosestCentersTask(){
      // release mem for local offsets copy
      delete[] this->shifted_offsets_CSC;
    }

    void execute(){
      GLOG_ASSERT(this->UUTrC != nullptr, "UUTrC is nullptr");
      // Init
      word_id_t *shifted_rows_CSC = (word_id_t *) this->in_mem_ptrs[this->shifted_rows_CSC_fptr];
      FPTYPE *shifted_vals_CSC = (FPTYPE *) this->in_mem_ptrs[this->shifted_vals_CSC_fptr];
      FPTYPE *projected_dist_matrix = new FPTYPE[this->doc_blk_size * this->num_centers];

      /* projected_closest_centers -- BEGIN  */
      {
        /* distsq_projected_docs_to_projected_centers -- BEGIN  */
        {
          const uint64_t ones_vec_size = std::max(this->doc_blk_size, this->num_centers);
          FPTYPE *ones_vec = new FPTYPE[ones_vec_size];
          std::fill_n(ones_vec, ones_vec_size, (FPTYPE)1.0);

          const char transa = 'N';
          const MKL_INT m = doc_blk_size;
          const MKL_INT n = num_centers;
          const MKL_INT k = U_cols;
          const char matdescra[6] = {'G', 0, 0, 'C', 0, 0};

          /* `multiply_with` -- BEGIN  */
          {
            assert(sizeof(MKL_INT) == sizeof(offset_t));
            assert(sizeof(word_id_t) == sizeof(MKL_INT));
            assert(sizeof(offset_t) == sizeof(MKL_INT));

            const char transa = 'N';
            const MKL_INT m = doc_blk_size;
            const MKL_INT n = num_centers;
            const MKL_INT k = vocab_size;
            const char matdescra[6] = {'G', 0, 0, 'C', 0, 0};
            FPTYPE alpha = 1.0;
            FPTYPE beta = 0.0;

            FPcsrmm(&transa, &m, &n, &k, &alpha, matdescra,
                    shifted_vals_CSC, (const MKL_INT *)shifted_rows_CSC,
                    (const MKL_INT *)shifted_offsets_CSC, (const MKL_INT *)(shifted_offsets_CSC + 1),
                    UUTrC, &n, &beta, projected_dist_matrix, &n);
          }
          /* `multiply_with` -- END  */
          this->UUTrC = nullptr; // needs to be re-set for next iteration

          FPgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                 m, n, 1, (FPTYPE)1.0,
                 ones_vec, m, projected_centers_l2sq, n,
                 (FPTYPE)1.0, projected_dist_matrix, n);

          FPgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                 m, n, 1, (FPTYPE)1.0,
                 projected_docs_l2sq, m, ones_vec, n,
                 (FPTYPE)1.0, projected_dist_matrix, n);

          delete[] ones_vec;
          /* distsq_projected_docs_to_projected_centers -- END  */
        }

        // compute new center assignment
        pfor_static_131072(int64_t d = 0; d < this->doc_blk_size; ++d){
            center_index[d] = (doc_id_t)FPimin(num_centers, projected_dist_matrix + (size_t)d * (size_t)num_centers, 1);
        }
      }
      /* projected_closest_centers -- END  */
      
      // Cleanup
      delete[] projected_dist_matrix;
    }

    // reset with new center info
    void reset(FPTYPE *const projected_centers_tr, FPTYPE *const projected_centers_l2sq,
               FPTYPE *const UUTrC, uint64_t num_centers) {
      this->projected_centers_tr = projected_centers_tr;
      this->projected_centers_l2sq = projected_centers_l2sq;
      this->UUTrC = UUTrC;
      this->num_centers = num_centers;
    }
  };
} // namespace kmeans