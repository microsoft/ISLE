#pragma once

#include "blas-on-flash/include/pointers/pointer.h"
#include "blas-on-flash/include/pointers/allocator.h"
#include "blas-on-flash/include/tasks/task.h"

namespace ISLE
{
class RthHighest : public flash::BaseTask
{
  // flash pointers
  offset_t *local_csr_offs;
  flash::flash_ptr<FPTYPE> local_csr_vals;
  flash::flash_ptr<doc_id_t> local_csr_cols;

  // A_sp
  SparseMatrix<A_TYPE> *A_sp;

  // problem parameters
  uint64_t start_row;
  uint64_t blk_size;
  uint64_t num_topics;
  uint64_t r;
  FPTYPE avg_doc_size;

  // other arrays
  int* cluster_ids;
  FPTYPE* threshold_matrix_tr;
  std::vector<doc_id_t> *closest_docs;

public:
  RthHighest(SparseMatrix<A_TYPE> *A_sp, offset_t *csr_offs,
             flash::flash_ptr<doc_id_t> base_csr_cols,
             flash::flash_ptr<FPTYPE> base_csr_vals,
             uint64_t start_row, uint64_t blk_size,
             uint64_t num_topics, uint64_t r,
             std::vector<doc_id_t> *closest_docs, int *cluster_ids,
             FPTYPE *threshold_matrix_tr, FPTYPE avg_doc_size) : A_sp(A_sp)
  {
    this->local_csr_offs = csr_offs + start_row;
    this->local_csr_cols = base_csr_cols + csr_offs[start_row];
    this->local_csr_vals = base_csr_vals + csr_offs[start_row];
    this->start_row = start_row;
    this->blk_size = blk_size;

    this->avg_doc_size = avg_doc_size;
    this->num_topics = num_topics;
    this->r = r;
    this->cluster_ids = cluster_ids;
    this->closest_docs = closest_docs;
    this->threshold_matrix_tr = threshold_matrix_tr;

    // add reads and writes
    flash::StrideInfo sinfo = {1, 1, 1};
    sinfo.len_per_stride = (this->local_csr_offs[this->blk_size] - this->local_csr_offs[0]) * sizeof(doc_id_t);
    this->add_read(this->local_csr_cols, sinfo);
    sinfo.len_per_stride = (this->local_csr_offs[this->blk_size] - this->local_csr_offs[0]) * sizeof(FPTYPE);
    this->add_read(this->local_csr_vals, sinfo);
  }

  void execute()
  {
    doc_id_t *local_csr_cols_ptr = (doc_id_t *)this->in_mem_ptrs[this->local_csr_cols];
    FPTYPE *local_csr_vals_ptr = (FPTYPE *)this->in_mem_ptrs[this->local_csr_vals];
    // FPscal(local_csr_offs[blk_size] - local_csr_offs[0], avg_doc_size, local_csr_vals_ptr, 1);

    // list word freqs from CSR
    this->A_sp->rth_highest_element_using_CSR(start_row, start_row + blk_size,
                                              num_topics, r, closest_docs, local_csr_vals_ptr,
                                              local_csr_cols_ptr, local_csr_offs, cluster_ids, threshold_matrix_tr);
  }

  // DEPRECATED; to be removed in future versions
  FBLAS_UINT size()
  {
    return (1 << 20);
  }
};
}
