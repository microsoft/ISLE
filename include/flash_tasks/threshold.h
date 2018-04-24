#pragma once

#include "blas-on-flash/include/pointers/pointer.h"
#include "blas-on-flash/include/pointers/allocator.h"
#include "blas-on-flash/include/tasks/task.h"

namespace ISLE
{
class ThresholdTask : public flash::BaseTask
{
  // flash pointers
  offset_t *local_csr_offs;
  flash::flash_ptr<FPTYPE> local_csr_vals;

  // A_sp
  SparseMatrix<A_TYPE> *A_sp;

  // problem parameters
  uint64_t start_row;
  uint64_t blk_size;
  uint64_t num_topics;
  FPTYPE avg_doc_size;

  // output
  // NOTE :: start writing output at 'thresholds[start_row]'
  std::vector<FPTYPE> &thresholds;
  std::vector<A_TYPE> *freqs;
  offset_t *nnzs_store;

public:
  ThresholdTask(SparseMatrix<A_TYPE> *A_sp, offset_t *csr_offs,
                flash::flash_ptr<FPTYPE> base_csr_vals,
                uint64_t start_row, uint64_t blk_size,
                std::vector<FPTYPE> &thresholds, std::vector<A_TYPE> *freqs,
                uint64_t num_topics, offset_t *nnzs_store,
                FPTYPE avg_doc_size) : 
                A_sp(A_sp), thresholds(thresholds), freqs(freqs), num_topics(num_topics), nnzs_store(nnzs_store)
  {
    this->local_csr_offs = csr_offs + start_row;
    this->start_row = start_row;
    this->blk_size = blk_size;
    this->local_csr_vals = base_csr_vals + csr_offs[start_row];
    this->avg_doc_size = avg_doc_size;

    // add reads and writes
    flash::StrideInfo sinfo = {1, 1, 1};
    sinfo.len_per_stride = (this->local_csr_offs[this->blk_size] - this->local_csr_offs[0]) * sizeof(FPTYPE);
    this->add_read(this->local_csr_vals, sinfo);
  }

  void execute()
  {
    FPTYPE *local_csr_vals_ptr = (FPTYPE *)this->in_mem_ptrs[this->local_csr_vals];
    FPscal(local_csr_offs[blk_size] - local_csr_offs[0], avg_doc_size, local_csr_vals_ptr, 1);

    // list word freqs from CSR
    this->A_sp->list_word_freqs_from_CSR(start_row, start_row + blk_size, local_csr_vals_ptr, local_csr_offs, freqs);

    // compute thresholds
    *this->nnzs_store = this->A_sp->compute_thresholds(start_row, start_row + blk_size, freqs, thresholds, num_topics);

    // clear out freqs
    for (word_id_t word = start_row; word < start_row + blk_size; ++word)
      freqs[word].clear();
  }

  // DEPRECATED; to be removed in future versions
  FBLAS_UINT size()
  {
    return (1 << 20);
  }
};
}
