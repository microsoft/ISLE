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
  using namespace ISLE;
  using namespace flash;
class ClosestCentersTask : public BaseTask
{
  const FPTYPE *centers_tr = nullptr;
  const FPTYPE *centers_l2sq = nullptr;
  offset_t *shifted_offsets_CSC = nullptr;
  const FPTYPE *const docs_l2sq = nullptr;
  doc_id_t *center_index = nullptr;

  flash::flash_ptr<word_id_t> shifted_rows_CSC_fptr;
  flash::flash_ptr<FPTYPE> shifted_vals_CSC_fptr;

  const doc_id_t doc_blk_size = 0;
  const word_id_t vocab_size = 0;
  doc_id_t num_centers = 0;

  const FPTYPE *ones_vec = nullptr;

public:
  ClosestCentersTask(const FPTYPE *const docs_l2sq,
                     doc_id_t *center_index,
                     offset_t *offsets_CSC,
                     flash::flash_ptr<word_id_t> shifted_rows_CSC_fptr,
                     flash::flash_ptr<FPTYPE> shifted_vals_CSC_fptr,
                     const doc_id_t doc_blk_size,
                     const word_id_t vocab_size) : docs_l2sq(docs_l2sq),
                                                   center_index(center_index),
                                                   shifted_rows_CSC_fptr(shifted_rows_CSC_fptr),
                                                   shifted_vals_CSC_fptr(shifted_vals_CSC_fptr),
                                                   doc_blk_size(doc_blk_size),
                                                   vocab_size(vocab_size)
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

  ~ClosestCentersTask()
  {
    // release mem for local offsets copy
    delete[] this->shifted_offsets_CSC;
  }

  void execute()
  {
    GLOG_ASSERT(this->ones_vec != nullptr, "ones_vec is nullptr");
    GLOG_ASSERT(this->centers_tr != nullptr, "centers_tr is nullptr");
    GLOG_ASSERT(this->centers_l2sq != nullptr, "centers_l2sq is nullptr");
    GLOG_ASSERT(this->shifted_offsets_CSC != nullptr, "shifted_offsets_CSC is nullptr");
    GLOG_ASSERT(this->docs_l2sq != nullptr, "docs_l2sq is nullptr");
    GLOG_ASSERT(this->center_index != nullptr, "center_index is nullptr");
    GLOG_ASSERT(this->doc_blk_size != 0, "doc_blk_size is 0");
    GLOG_ASSERT(this->vocab_size != 0, "vocab_size is 0");
    GLOG_ASSERT(this->num_centers != 0, "num_centers is 0");

    // Init
		mkl_set_num_threads_local(0);
    word_id_t *shifted_rows_CSC = (word_id_t *)this->in_mem_ptrs[this->shifted_rows_CSC_fptr];
    FPTYPE *shifted_vals_CSC = (FPTYPE *)this->in_mem_ptrs[this->shifted_vals_CSC_fptr];

    FPTYPE *dist_matrix = new FPTYPE[this->doc_blk_size * this->num_centers];

    /* closest_centers -- BEGIN  */
    {
      /* distsq_docs_to_centers -- BEGIN  */
      {
        assert(sizeof(MKL_INT) == sizeof(offset_t));

        const char transa = 'N';
        const MKL_INT m = doc_blk_size;
        const MKL_INT n = num_centers;
        const MKL_INT k = vocab_size;
        const char matdescra[6] = {'G', 0, 0, 'C', 0, 0};
        FPTYPE alpha = -2.0;
        FPTYPE beta = 0.0;

        FPcsrmm(&transa, &m, &n, &k, &alpha, matdescra,
                shifted_vals_CSC, (const MKL_INT *)shifted_rows_CSC,
                (const MKL_INT *)(shifted_offsets_CSC), (const MKL_INT *)(shifted_offsets_CSC + 1),
                centers_tr, &n, &beta, dist_matrix, &n);

        FPgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
               doc_blk_size, num_centers, 1,
               (FPTYPE)1.0, ones_vec, doc_blk_size, centers_l2sq, num_centers,
               (FPTYPE)1.0, dist_matrix, num_centers);
        FPgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
               doc_blk_size, num_centers, 1,
               (FPTYPE)1.0, docs_l2sq, doc_blk_size, ones_vec, num_centers,
               (FPTYPE)1.0, dist_matrix, num_centers);
      }
      /* distsq_docs_to_centers -- END  */

      // compute new center assignment
			#pragma omp parallel for schedule(static, 8192) num_threads(MAX_THREADS)
      for(int64_t d = 0; d < this->doc_blk_size; ++d)
      {
        center_index[d] = (doc_id_t)FPimin(num_centers, dist_matrix + (size_t)d * (size_t)num_centers, 1);
      }
    }
    /* closest_centers -- END  */

    // Cleanup
    delete[] dist_matrix;
  }

  // reset with new center info
  void reset(const FPTYPE *centers_tr, const FPTYPE *centers_l2sq, 
             const FPTYPE *ones_vec, uint64_t num_centers)
  {
    this->centers_tr = centers_tr;
    this->centers_l2sq = centers_l2sq;
    this->ones_vec = ones_vec;
    this->num_centers = num_centers;
  }

  // DEPRECATED :: to be removed later
  FBLAS_UINT size()
  {
    return (1 << 20);
  }
};

class ComputeNewCentersTask : public BaseTask
{
  FPTYPE *centers = nullptr;
  offset_t *shifted_offsets_CSC = nullptr;
  doc_id_t *center_index = nullptr;

  flash::flash_ptr<word_id_t> shifted_rows_CSC_fptr;
  flash::flash_ptr<FPTYPE> shifted_vals_CSC_fptr;

  const doc_id_t doc_blk_size = 0;
  const word_id_t vocab_size = 0;
  doc_id_t num_centers = 0;

  std::vector<size_t> &cluster_sizes;

public:
  ComputeNewCentersTask(doc_id_t *center_index,
                        FPTYPE *centers,
                        std::vector<size_t> &cluster_sizes,
                        offset_t *offsets_CSC,
                        flash::flash_ptr<word_id_t> shifted_rows_CSC_fptr,
                        flash::flash_ptr<FPTYPE> shifted_vals_CSC_fptr,
                        const doc_id_t doc_blk_size,
                        const word_id_t vocab_sz,
                        const doc_id_t num_centers) : center_index(center_index),
                                                      centers(centers),
                                                      cluster_sizes(cluster_sizes),
                                                      shifted_rows_CSC_fptr(shifted_rows_CSC_fptr),
                                                      shifted_vals_CSC_fptr(shifted_vals_CSC_fptr),
                                                      doc_blk_size(doc_blk_size),
                                                      vocab_size(vocab_sz),
                                                      num_centers(num_centers) {
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

  ~ComputeNewCentersTask()
  {
    // release mem for local offsets copy
    delete[] this->shifted_offsets_CSC;
  }

  void execute()
  {
    GLOG_ASSERT(this->shifted_offsets_CSC != nullptr, "shifted_offsets_CSC is nullptr");
    GLOG_ASSERT(this->center_index != nullptr, "center_index is nullptr");
    GLOG_ASSERT(this->doc_blk_size != 0, "doc_blk_size is 0");
    GLOG_ASSERT(this->vocab_size != 0, "vocab_size is 0");
    GLOG_ASSERT(this->num_centers != 0, "num_centers is 0");

    // Init
		mkl_set_num_threads_local(0);
    std::vector<std::vector<doc_id_t>> closest_docs(num_centers);
    word_id_t *shifted_rows_CSC = (word_id_t *)this->in_mem_ptrs[this->shifted_rows_CSC_fptr];
    FPTYPE *shifted_vals_CSC = (FPTYPE *)this->in_mem_ptrs[this->shifted_vals_CSC_fptr];

    for (doc_id_t d = 0; d < doc_blk_size; ++d){
      closest_docs[center_index[d]].push_back(d);
    }

    // update section - serialize access using mutex if running in parallel
    {
      for (uint64_t c = 0; c < num_centers; ++c)
        cluster_sizes[c] += closest_docs[c].size();

#pragma omp parallel for schedule(dynamic, 1) num_threads(MAX_THREADS)
      for(uint64_t c = 0; c < num_centers; ++c)
      {
        auto center = centers + (c * vocab_size);
        for (auto diter = closest_docs[c].begin(); diter != closest_docs[c].end(); ++diter)
        {
          for (auto witer = shifted_offsets_CSC[*diter]; witer < shifted_offsets_CSC[*diter + 1]; ++witer)
          {
            *(center + shifted_rows_CSC[witer]) += shifted_vals_CSC[witer];
          }
        }
      }
    }

    closest_docs.clear();
  }

  // DEPRECATED :: to be removed later
  FBLAS_UINT size()
  {
    return (1 << 20);
  }
};
} // namespace


namespace Kmeans{
void closest_centers_full(
    const FPTYPE *const centers,
    const FPTYPE *const centers_l2sq,
    offset_t *offsets_CSC,
    flash::flash_ptr<word_id_t> rows_CSC_fptr,
    flash::flash_ptr<FPTYPE> vals_CSC_fptr,
    const doc_id_t num_docs,
    const doc_id_t doc_block_size,
    const doc_id_t num_centers,
    const word_id_t vocab_size,
    const FPTYPE *const docs_l2sq,
    doc_id_t* const center_index) {
  // transpose centers
  FPTYPE* centers_tr = new FPTYPE[num_centers * vocab_size];
  FPomatcopy('C', 'T', vocab_size, num_centers, 1.0f, centers,
             vocab_size, centers_tr, num_centers);

  // create a common ones_vec
  const uint64_t ones_vec_size = std::max(doc_block_size, num_centers);
  FPTYPE *ones_vec = new FPTYPE[ones_vec_size];
  std::fill_n(ones_vec, ones_vec_size, (FPTYPE)1.0f);

  // construct and issue tasks
  uint64_t n_doc_blks = ROUND_UP(num_docs, doc_block_size) / doc_block_size;
  ClosestCentersTask **cc_tsks = new ClosestCentersTask *[n_doc_blks];
  for (MKL_INT i = 0; i < n_doc_blks; i++) {
    MKL_INT doc_blk_start = doc_block_size * i;
    MKL_INT doc_blk_size = std::min(num_docs - doc_blk_start, (doc_id_t)doc_block_size);
    MKL_INT doc_blk_offset = offsets_CSC[doc_blk_start];
    cc_tsks[i] = new ClosestCentersTask(docs_l2sq + doc_blk_start,
                                        center_index + doc_blk_start,
                                        offsets_CSC + doc_blk_start,
                                        rows_CSC_fptr + doc_blk_offset,
                                        vals_CSC_fptr + doc_blk_offset,
                                        doc_blk_size,
                                        vocab_size);

    cc_tsks[i]->reset(centers_tr, centers_l2sq, ones_vec, num_centers);

    flash::sched.add_task(cc_tsks[i]);
  }

  // wait for completion
  flash::sleep_wait_for_complete(cc_tsks, n_doc_blks);
  for (MKL_INT i = 0; i < n_doc_blks; i++) {
    delete cc_tsks[i];
  }
  delete[] cc_tsks;

  // cleanup
  delete[] centers_tr;
  delete[] ones_vec;
  // // leak-check code
  // __lsan_do_recoverable_leak_check();
  return;
}

void compute_new_centers_full(offset_t *offsets_CSC,
                              flash::flash_ptr<word_id_t> rows_CSC_fptr,
                              flash::flash_ptr<FPTYPE> vals_CSC_fptr,
                              const doc_id_t num_docs,
                              const doc_id_t doc_block_size,
                              const doc_id_t num_centers,
                              const word_id_t vocab_size,
                              doc_id_t *center_index,
                              FPTYPE *centers,
                              std::vector<size_t> &cluster_sizes)
{
  // construct and issue tasks
  uint64_t n_doc_blks = ROUND_UP(num_docs, doc_block_size) / doc_block_size;
  ComputeNewCentersTask **cnc_tsks = new ComputeNewCentersTask *[n_doc_blks];
  for (MKL_INT i = 0; i < n_doc_blks; i++) {
    MKL_INT doc_blk_start = doc_block_size * i;
    MKL_INT doc_blk_size = std::min(num_docs - doc_blk_start, (doc_id_t)doc_block_size);
    MKL_INT doc_blk_offset = offsets_CSC[doc_blk_start];
    cnc_tsks[i] = new ComputeNewCentersTask(center_index + doc_blk_start,
                                            centers,
                                            cluster_sizes,
                                            offsets_CSC + doc_blk_start,
                                            rows_CSC_fptr + doc_blk_offset,
                                            vals_CSC_fptr + doc_blk_offset,
                                            doc_blk_size,
                                            vocab_size,
                                            num_centers);

    if (i > 0) {
      cnc_tsks[i]->add_parent(cnc_tsks[i - 1]->get_id());
    }

    flash::sched.add_task(cnc_tsks[i]);
  }

  // wait for completion
  flash::sleep_wait_for_complete(cnc_tsks, n_doc_blks);
  for (MKL_INT i = 0; i < n_doc_blks; i++)
  {
    delete cnc_tsks[i];
  }
  delete[] cnc_tsks;
  // // leak-check code
  // __lsan_do_recoverable_leak_check();
  return;
}
} // namespace Kmeans
