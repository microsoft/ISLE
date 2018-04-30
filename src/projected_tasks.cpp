#include "types.h"
#include "kmeans.h"

#include "blas-on-flash/include/utils.h"
#include "blas-on-flash/include/lib_funcs.h"
#include "blas-on-flash/include/flash_blas.h"
#include "blas-on-flash/include/scheduler/scheduler.h"

//#include "lsan_interface.h"

namespace flash
{
  extern Scheduler sched;
} // namespace flash

namespace {
  using namespace flash;
  using namespace ISLE;
  class BlockL2SqTask : public BaseTask{
    const MKL_INT *blk_offsets_CSC;
    flash_ptr<FPTYPE> blk_vals_CSC_fptr;

    FPTYPE *blk_l2sq;
    MKL_INT blk_size;
  public:
    BlockL2SqTask(const MKL_INT *offsets_CSC,  flash_ptr<FPTYPE> vals_CSC_fptr,
                  FPTYPE *col_l2sq, MKL_INT col_blk_size)
    {
      this->blk_offsets_CSC = offsets_CSC;
      this->blk_vals_CSC_fptr = vals_CSC_fptr;
      this->blk_l2sq = col_l2sq;
      this->blk_size = col_blk_size;
      MKL_INT nnzs = (this->blk_offsets_CSC[this->blk_size] - this->blk_offsets_CSC[0]);

      // add reads
      StrideInfo sinfo = {1, 1, 1};
      sinfo.len_per_stride = (nnzs * sizeof(FPTYPE));
      this->add_read(this->blk_vals_CSC_fptr, sinfo);
    }

    void execute(){
      FPTYPE* blk_vals_CSC = (FPTYPE*) this->in_mem_ptrs[this->blk_vals_CSC_fptr];

      pfor_dynamic_1024(MKL_INT i=0;i<this->blk_size;i++){
        MKL_INT nnzs_i = this->blk_offsets_CSC[i+1] - this->blk_offsets_CSC[i];
        FPTYPE* vec_start = blk_vals_CSC + (this->blk_offsets_CSC[i] - this->blk_offsets_CSC[0]);
        this->blk_l2sq[i] = FPnrm2(nnzs_i, vec_start, 1);
        // scale to get L2^2
        this->blk_l2sq[i] *= this->blk_l2sq[i];
      }
    }

    FBLAS_UINT size(){
      return (1<<20);
    }
  };

  class ProjClosestCentersTask : public BaseTask
  {
    const FPTYPE *projected_centers_tr = nullptr;
    const FPTYPE *projected_centers_l2sq = nullptr;
    offset_t *shifted_offsets_CSC = nullptr;
    const FPTYPE *const projected_docs_l2sq = nullptr;
    doc_id_t *center_index = nullptr;

    flash::flash_ptr<word_id_t> shifted_rows_CSC_fptr;
    flash::flash_ptr<FPTYPE> shifted_vals_CSC_fptr;

    const doc_id_t doc_blk_size = 0;
    const word_id_t vocab_size = 0;
    doc_id_t num_centers = 0;

    const FPTYPE *UUTrC = nullptr;
    uint64_t U_rows = 0, U_cols = 0;

    const FPTYPE *ones_vec = nullptr;

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
                                                               U_rows(U_rows), U_cols(U_cols)
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

    ~ProjClosestCentersTask() {
      // release mem for local offsets copy
      delete[] this->shifted_offsets_CSC;
    }

    void execute()
    {
      GLOG_ASSERT(this->UUTrC != nullptr, "UUTrC is nullptr");
      GLOG_ASSERT(this->ones_vec != nullptr, "ones_vec is nullptr");
      GLOG_ASSERT(this->projected_centers_tr != nullptr, "projected_centers_tr is nullptr");
      GLOG_ASSERT(this->projected_centers_l2sq != nullptr, "projected_centers_l2sq is nullptr");
      GLOG_ASSERT(this->shifted_offsets_CSC != nullptr, "shifted_offsets_CSC is nullptr");
      GLOG_ASSERT(this->projected_docs_l2sq != nullptr, "projected_docs_l2sq is nullptr");
      GLOG_ASSERT(this->center_index != nullptr, "center_index is nullptr");
      GLOG_ASSERT(this->doc_blk_size != 0, "doc_blk_size is 0");
      GLOG_ASSERT(this->vocab_size != 0, "vocab_size is 0");
      GLOG_ASSERT(this->U_rows != 0, "U_rows is 0");
      GLOG_ASSERT(this->U_cols != 0, "U_cols is 0");
      GLOG_ASSERT(this->num_centers != 0, "num_centers is 0");

      // Init
      word_id_t *shifted_rows_CSC = (word_id_t *)this->in_mem_ptrs[this->shifted_rows_CSC_fptr];
      FPTYPE *shifted_vals_CSC = (FPTYPE *)this->in_mem_ptrs[this->shifted_vals_CSC_fptr];

      FPTYPE *projected_dist_matrix = new FPTYPE[this->doc_blk_size * this->num_centers];

      /* projected_closest_centers -- BEGIN  */
      {
        /* distsq_projected_docs_to_projected_centers -- BEGIN  */
        {
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
        }
        /* distsq_projected_docs_to_projected_centers -- END  */

        // compute new center assignment
        pfor_dynamic_1024(int64_t d = 0; d < this->doc_blk_size; ++d)
        {
          center_index[d] = (doc_id_t)FPimin(num_centers, projected_dist_matrix + (size_t)d * (size_t)num_centers, 1);
        }
      }
      /* projected_closest_centers -- END  */

      // Cleanup
      delete[] projected_dist_matrix;
    }

    // reset with new center info
    void reset(const FPTYPE *projected_centers_tr, const FPTYPE * projected_centers_l2sq,
               const FPTYPE *UUTrC, const FPTYPE* ones_vec, uint64_t num_centers) {
      this->projected_centers_tr = projected_centers_tr;
      this->projected_centers_l2sq = projected_centers_l2sq;
      this->UUTrC = UUTrC;
      this->ones_vec = ones_vec;
      this->num_centers = num_centers;
    }

    // DEPRECATED :: to be removed later
    FBLAS_UINT size(){
      return (1<<20);
    }
  };

  class ProjComputeNewCentersTask : public BaseTask
  {
    FPTYPE *projected_centers = nullptr;
    offset_t *shifted_offsets_CSC = nullptr;
    doc_id_t *center_index = nullptr;

    flash::flash_ptr<word_id_t> shifted_rows_CSC_fptr;
    flash::flash_ptr<FPTYPE> shifted_vals_CSC_fptr;

    const doc_id_t doc_blk_size = 0;
    const word_id_t vocab_size = 0;
    doc_id_t num_centers = 0;

    const FPTYPE *U_rowmajor = nullptr;
    uint64_t U_rows = 0, U_cols = 0;

    std::vector<size_t> &cluster_sizes;

  public:
    ProjComputeNewCentersTask(doc_id_t *center_index, FPTYPE* projected_centers,
                              std::vector<size_t> &cluster_sizes, offset_t *offsets_CSC,
                              flash::flash_ptr<word_id_t> shifted_rows_CSC_fptr,
                              flash::flash_ptr<FPTYPE> shifted_vals_CSC_fptr,
                              const doc_id_t doc_blk_size, const word_id_t vocab_sz, const doc_id_t num_centers, FPTYPE *U_rowmajor,
                              uint64_t U_rows, uint64_t U_cols) : center_index(center_index),
                                                                  projected_centers(projected_centers),
                                                                  cluster_sizes(cluster_sizes),
                                                                  shifted_rows_CSC_fptr(shifted_rows_CSC_fptr),
                                                                  shifted_vals_CSC_fptr(shifted_vals_CSC_fptr),
                                                                  doc_blk_size(doc_blk_size),
                                                                  vocab_size(vocab_sz), num_centers(num_centers),
                                                                  U_rowmajor(U_rowmajor),
                                                                  U_rows(U_rows), U_cols(U_cols) {
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

    ~ProjComputeNewCentersTask() {
      // release mem for local offsets copy
      delete[] this->shifted_offsets_CSC;
    }

    void execute()
    {
      GLOG_ASSERT(this->U_rowmajor != nullptr, "U_rowmajor is nullptr");
      GLOG_ASSERT(this->shifted_offsets_CSC != nullptr, "shifted_offsets_CSC is nullptr");
      GLOG_ASSERT(this->center_index != nullptr, "center_index is nullptr");
      GLOG_ASSERT(this->doc_blk_size != 0, "doc_blk_size is 0");
      GLOG_ASSERT(this->vocab_size != 0, "vocab_size is 0");
      GLOG_ASSERT(this->U_rows != 0, "U_rows is 0");
      GLOG_ASSERT(this->U_cols != 0, "U_cols is 0");
      GLOG_ASSERT(this->num_centers != 0, "num_centers is 0");

      // Init
      word_id_t *shifted_rows_CSC = (word_id_t *)this->in_mem_ptrs[this->shifted_rows_CSC_fptr];
      FPTYPE *shifted_vals_CSC = (FPTYPE *)this->in_mem_ptrs[this->shifted_vals_CSC_fptr];
      FPTYPE *projected_docs = new FPTYPE[doc_blk_size * num_centers];

      std::vector<std::vector<doc_id_t>> closest_docs(num_centers, std::vector<doc_id_t>());

      for (doc_id_t d = 0; d < doc_blk_size; ++d)
        closest_docs[center_index[d]].push_back(d);

      // cluster sizes
      for (size_t c = 0; c < num_centers; ++c)
        cluster_sizes[c] += closest_docs[c].size();

      // UT_times_docs(block * doc_blk_size,
      //               block * doc_blk_size + num_docs_in_block,
      //               projected_docs);
      /* `multiply_with` -- BEGIN  */
      {
        assert(sizeof(MKL_INT) == sizeof(offset_t));
        assert(sizeof(word_id_t) == sizeof(MKL_INT));
        assert(sizeof(offset_t) == sizeof(MKL_INT));

        const char transa = 'N';
        const MKL_INT m = doc_blk_size;
        const MKL_INT n = U_cols;
        const MKL_INT k = vocab_size;
        const char matdescra[6] = {'G', 0, 0, 'C', 0, 0};
        FPTYPE alpha = 1.0;
        FPTYPE beta = 0.0;

        FPcsrmm(&transa, &m, &n, &k, &alpha, matdescra,
                shifted_vals_CSC, (const MKL_INT *)shifted_rows_CSC,
                (const MKL_INT *)shifted_offsets_CSC, (const MKL_INT *)(shifted_offsets_CSC + 1),
                U_rowmajor, &n, &beta, projected_docs, &n);
      }
      /* `multiply_with` -- END  */

      pfor_dynamic_1(uint64_t c = 0; c < num_centers; ++c) {
        FPTYPE *center = projected_centers + c * num_centers;
        for (auto diter = closest_docs[c].begin(); diter != closest_docs[c].end(); ++diter)
          FPaxpy(num_centers, 1.0, projected_docs + (*diter) * num_centers, 1, center, 1);
      }
      delete[] projected_docs;
    }

    // DEPRECATED :: to be removed later
    FBLAS_UINT size(){
      return (1<<20);
    }
  };
} // namespace

namespace Kmeans{
  void compute_col_l2sq(const MKL_INT *offsets_CSC, flash_ptr<FPTYPE> vals_CSC_fptr,
                        FPTYPE *col_l2sq, MKL_INT n_cols, MKL_INT col_blk_size) {
    memset(col_l2sq, 0, n_cols * sizeof(FPTYPE));
    uint64_t n_col_blks = ROUND_UP(n_cols, col_blk_size) / col_blk_size;
    BlockL2SqTask **l2sq_tsks = new BlockL2SqTask *[n_col_blks];
    for (MKL_INT i = 0; i < n_col_blks; i++) {
      MKL_INT col_start = col_blk_size * i;
      MKL_INT col_block_size = std::min(n_cols - col_start, col_blk_size);
      l2sq_tsks[i] = new BlockL2SqTask(offsets_CSC + col_start,
                                       vals_CSC_fptr + offsets_CSC[col_start],
                                       col_l2sq + col_start,
                                       col_block_size);
      flash::sched.add_task(l2sq_tsks[i]);
    }
    
    flash::sleep_wait_for_complete(l2sq_tsks, n_col_blks);
    for (MKL_INT i = 0; i < n_col_blks; i++) {
      delete l2sq_tsks[i];
    }
    delete[] l2sq_tsks;

    return;
  }

  void projected_closest_centers_full(
      const FPTYPE *const projected_centers_tr, const FPTYPE *const projected_centers_l2sq,
      offset_t *offsets_CSC, flash::flash_ptr<word_id_t> rows_CSC_fptr,
      flash::flash_ptr<FPTYPE> vals_CSC_fptr, const doc_id_t num_docs,
      const doc_id_t doc_block_size, const doc_id_t num_centers,
      const word_id_t vocab_size, FPTYPE *U_rowmajor, uint64_t U_rows, uint64_t U_cols,
      const FPTYPE *const projected_docs_l2sq, doc_id_t *center_index){
    
    // compute UUTrC
    FPTYPE *UUTrC = new FPTYPE[U_rows * num_centers];
    FPgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
           U_rows, num_centers, U_cols, (FPTYPE)-2.0,
           U_rowmajor, U_cols, projected_centers_tr, num_centers,
           (FPTYPE)0.0, UUTrC, num_centers);
    
    // create a common ones_vec
    const uint64_t ones_vec_size = std::max(doc_block_size, num_centers);
    FPTYPE* ones_vec = new FPTYPE[ones_vec_size];
    std::fill_n(ones_vec, ones_vec_size, (FPTYPE)1.0f);

    // construct and issue tasks
    uint64_t n_doc_blks = ROUND_UP(num_docs, doc_block_size) / doc_block_size;
    ProjClosestCentersTask **proj_tsks = new ProjClosestCentersTask *[n_doc_blks];
    for (MKL_INT i = 0; i < n_doc_blks; i++) {
      MKL_INT doc_blk_start = doc_block_size * i;
      MKL_INT doc_blk_size = std::min(num_docs - doc_blk_start, (doc_id_t)doc_block_size);
      MKL_INT doc_blk_offset = offsets_CSC[doc_blk_start];
      proj_tsks[i] = new ProjClosestCentersTask(projected_docs_l2sq + doc_blk_start,
                                                center_index + doc_blk_start,
                                                offsets_CSC + doc_blk_start,
                                                rows_CSC_fptr + doc_blk_offset,
                                                vals_CSC_fptr + doc_blk_offset,
                                                doc_blk_size, vocab_size,
                                                U_rows, U_cols);

      proj_tsks[i]->reset(projected_centers_tr, projected_centers_l2sq, UUTrC, ones_vec, num_centers);
      
      flash::sched.add_task(proj_tsks[i]);
    }

    // wait for completion
    flash::sleep_wait_for_complete(proj_tsks, n_doc_blks);
    for (MKL_INT i = 0; i < n_doc_blks; i++) {
      delete proj_tsks[i];
    }
    delete[] proj_tsks;

    // cleanup
    delete[] UUTrC;
    delete[] ones_vec;
    // // leak-check code
    // __lsan_do_recoverable_leak_check();
    return;
  }

  void projected_compute_new_centers_full(
      offset_t *offsets_CSC, flash::flash_ptr<word_id_t> rows_CSC_fptr,
      flash::flash_ptr<FPTYPE> vals_CSC_fptr, const doc_id_t num_docs,
      const doc_id_t doc_block_size, const doc_id_t num_centers,
      const word_id_t vocab_size, FPTYPE *U_rowmajor, uint64_t U_rows, uint64_t U_cols,
      doc_id_t *center_index, FPTYPE* projected_centers, std::vector<size_t> &cluster_sizes)
  {

    // construct and issue tasks
    uint64_t n_doc_blks = ROUND_UP(num_docs, doc_block_size) / doc_block_size;
    ProjComputeNewCentersTask **proj_tsks = new ProjComputeNewCentersTask *[n_doc_blks];
    for (MKL_INT i = 0; i < n_doc_blks; i++) {
      MKL_INT doc_blk_start = doc_block_size * i;
      MKL_INT doc_blk_size = std::min(num_docs - doc_blk_start, (doc_id_t)doc_block_size);
      MKL_INT doc_blk_offset = offsets_CSC[doc_blk_start];
      proj_tsks[i] = new ProjComputeNewCentersTask(center_index + doc_blk_start,
                                                   projected_centers,
                                                   cluster_sizes, offsets_CSC + doc_blk_start,
                                                   rows_CSC_fptr + doc_blk_offset,
                                                   vals_CSC_fptr + doc_blk_offset,
                                                   doc_blk_size, vocab_size,
                                                   num_centers, U_rowmajor, U_rows, U_cols);

      if(i > 0){
        proj_tsks[i]->add_parent(proj_tsks[i-1]->get_id());
      }

      flash::sched.add_task(proj_tsks[i]);
    }

    // wait for completion
    flash::sleep_wait_for_complete(proj_tsks, n_doc_blks);
    for (MKL_INT i = 0; i < n_doc_blks; i++) {
      delete proj_tsks[i];
    }
    delete[] proj_tsks;
    // // leak-check code
    // __lsan_do_recoverable_leak_check();
    return;
  }

} // namespace Kmeans
