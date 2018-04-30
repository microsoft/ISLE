#pragma once
#include <stdint.h>
#include <string>
#include "ks_types.h"
#include "ks_utils.h"
#include "blas-on-flash/include/pointers/allocator.h"
#include "timer.h"

using namespace flash;

class ReusableCsrmmTask : public BaseTask {
  // Inputs
  MKL_INT*           ia;
  flash_ptr<MKL_INT> ja;
  flash_ptr<ARMA_FPTYPE>  a;
  ARMA_FPTYPE*            b;
  ARMA_FPTYPE*            c;

  // use aligned-ja, a; then add n_offset
  uint64_t ja_offset;
  uint64_t a_offset;

  // Problem parameters
  uint64_t         a_nrows;
  uint64_t         a_ncols;
  uint64_t         b_ncols;
  const ARMA_FPTYPE alpha = 1.0;
  const ARMA_FPTYPE beta = 0.0;

  // Task-specific parameters
  uint64_t b_offset_bytes;
  uint64_t c_offset_bytes;
  uint64_t nnzs;
  bool use_orig = false;

 public:
  // make a copy of `ia` and zero-normalize it
  ReusableCsrmmTask(const MKL_INT* ia, flash_ptr<MKL_INT> ja,
                    flash_ptr<ARMA_FPTYPE> a, const uint64_t a_rows, const uint64_t a_cols,
                    const uint64_t b_cols);
  ~ReusableCsrmmTask();

  void execute();

  // resets `this->b` and `this->c`
  // also changes state to `Wait`
  void reset(ARMA_FPTYPE* new_b, ARMA_FPTYPE* new_c);

  // DEPRECATED - will be removed
  uint64_t size() {
    return 1 << 20;
  }
};

class FlashProdOp {
  flash_ptr<ARMA_FPTYPE>  a_csr;
  flash_ptr<ARMA_FPTYPE>  a_tr_csr;
  flash_ptr<MKL_INT> a_col;
  flash_ptr<MKL_INT> a_tr_col;
  flash_ptr<MKL_INT> a_off;
  flash_ptr<MKL_INT> a_tr_off;
  MKL_INT*           a_off_ptr;
  MKL_INT*           a_tr_off_ptr;
  ARMA_FPTYPE*            temp_ptr;

  MKL_INT a_nrows;
  MKL_INT a_ncols;
  MKL_INT dim;
  MKL_INT inner_dim;
  MKL_INT blk_size;
  bool    trans_first;

  // Tasks to be used/re-used
  ReusableCsrmmTask **a_tasks = nullptr, **a_tr_tasks = nullptr;
  uint64_t                n_a_tasks, n_a_tr_tasks;
  std::vector<uint64_t>   a_blk_sizes, a_tr_blk_sizes;
  std::vector<uint64_t>   a_blk_offsets, a_tr_blk_offsets;

  ISLE::Timer timer;

  // creates and allocates tasks
  void setup_tasks();

  // destorys tasks and frees resources
  void destroy_tasks();

  // interact with blas-scheduler
  FBLAS_UINT prev_n_compute_thr;

 public:
  uint64_t n_mms = 0;
  FlashProdOp(flash_ptr<ARMA_FPTYPE> a_csr, flash_ptr<MKL_INT> a_col, 
              flash_ptr<ARMA_FPTYPE> a_tr_csr, flash_ptr<MKL_INT> a_tr_col,
              MKL_INT *a_off_ptr, MKL_INT *a_tr_off_ptr, uint64_t a_rows,
              uint64_t a_cols, uint64_t nnzs, uint64_t blk_size);

  ~FlashProdOp();

  // Computes (A^T*(A*m_in)) if trans_first is false
  // Computes (A*(A^T*m_in)) if trans_first is true
  ARMA_FPMAT multiply(ARMA_FPMAT m_in);

  uint64_t rows() {
    return this->dim;
  }

  uint64_t cols() {
    return this->dim;
  }

  uint64_t dims() {
    return this->dim;
  }
};
