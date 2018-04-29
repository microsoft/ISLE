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
    uint64_t nnzs = (this->local_csr_offs[this->blk_size] - this->local_csr_offs[0]);

    // add reads and writes
    flash::StrideInfo sinfo = {1, 1, 1};
    sinfo.len_per_stride = nnzs * sizeof(doc_id_t);
    this->add_read(this->local_csr_cols, sinfo);
    sinfo.len_per_stride = nnzs * sizeof(FPTYPE);
    this->add_read(this->local_csr_vals, sinfo);
  }

  void execute()
  {
    doc_id_t *local_csr_cols_ptr = (doc_id_t *)this->in_mem_ptrs[this->local_csr_cols];
    FPTYPE *local_csr_vals_ptr = (FPTYPE *)this->in_mem_ptrs[this->local_csr_vals];
    // FPscal(local_csr_offs[blk_size] - local_csr_offs[0], avg_doc_size, local_csr_vals_ptr, 1);

    // list word freqs from CSR
    // 32K words / chunk
    uint64_t chunk_size = (1 << 14); 
    uint64_t num_chunks = ROUND_UP(blk_size, chunk_size) / chunk_size;
    pfor(uint64_t i=0;i<num_chunks;i++){
      uint64_t chunk_start = start_row + (chunk_size * i);
      uint64_t chunk_end = std::min(blk_size, chunk_start + chunk_size);
      uint64_t chunk_offset = local_csr_offs[chunk_start] - *local_csr_offs;
      
      offset_t* chunk_csr_offs = local_csr_offs + chunk_start;
      doc_id_t *chunk_csr_cols_ptr = local_csr_cols_ptr + chunk_offset;
      FPTYPE *chunk_csr_vals_ptr = local_csr_vals_ptr + chunk_offset;
      
      this->A_sp->rth_highest_element_using_CSR(chunk_start, chunk_end,
                                                num_topics, r, closest_docs,
                                                chunk_csr_vals_ptr, chunk_csr_cols_ptr,
                                                chunk_csr_offs, cluster_ids,
                                                threshold_matrix_tr);
    }
  }

  // DEPRECATED; to be removed in future versions
  FBLAS_UINT size()
  {
    return (1 << 20);
  }
};
}
