#pragma once

#include "BLAS-on-flash/include/pointers/pointer.h"
#include "BLAS-on-flash/include/pointers/allocator.h"
#include "BLAS-on-flash/include/tasks/task.h"

namespace ISLE
{
  template<typename T>
  class L2ThresholdedTask : public flash::BaseTask
  {
    // flash pointers
    offset_t *local_csc_offs;
    flash::flash_ptr<word_id_t> local_csc_rows;
    flash::flash_ptr<FPTYPE> local_csc_vals;

    // problem parameters
    uint64_t start_col;
    uint64_t blk_size;

    // output
    // NOTE :: start writing output at 'thresholds[start_col]'
    const std::vector<T> &thresholds;
    FPTYPE* local_norms;

  public:
    L2ThresholdedTask(offset_t *doc_offs, flash::flash_ptr<word_id_t> doc_word_idx,
                  flash::flash_ptr<T> doc_word_val, const std::vector<T> &thresholds,
                  FPTYPE *norms, offset_t start_col, offset_t blk_size) : thresholds(thresholds){
      this->local_csc_offs = doc_offs + start_col;
      this->start_col = start_col;
      this->blk_size = blk_size;
      this->local_csc_rows = doc_word_idx + doc_offs[start_col];
      this->local_csc_vals = doc_word_val + doc_offs[start_col];
      this->local_norms = norms + start_col;

      // add reads and writes
      flash::StrideInfo sinfo = {1, 1, 1};
      sinfo.len_per_stride = (this->local_csc_offs[this->blk_size] - this->local_csc_offs[0]) * sizeof(word_id_t);
      this->add_read(this->local_csc_rows, sinfo);
      sinfo.len_per_stride = (this->local_csc_offs[this->blk_size] - this->local_csc_offs[0]) * sizeof(T);
      this->add_read(this->local_csc_vals, sinfo);
    }

    void execute()
    {
      GLOG_DEBUG("entered execute()");
      word_id_t *local_csc_rows_ptr = (word_id_t *)this->in_mem_ptrs[this->local_csc_rows];
      T *local_csc_vals_ptr = (T *)this->in_mem_ptrs[this->local_csc_vals];

      offset_t pos_base = local_csc_offs[0];
      for(int64_t doc = 0; doc < blk_size; ++doc)
      {
        for (offset_t pos = local_csc_offs[doc] - pos_base; pos < (local_csc_offs[doc + 1] - pos_base); ++pos)
        {
          T val;
          if (std::is_same<T, FPTYPE>::value)
            val = std::round(local_csc_vals_ptr[pos]);
          else if (std::is_same<T, count_t>::value)
            val = local_csc_vals_ptr[pos];
          else
            assert(false);
          if (val >= thresholds[local_csc_rows_ptr[pos]])
            local_norms[doc] += thresholds[local_csc_rows_ptr[pos]];
        }
      }
    }

    // DEPRECATED; to be removed in future versions
    FBLAS_UINT size()
    {
      return (1 << 20);
    }
  };
}
