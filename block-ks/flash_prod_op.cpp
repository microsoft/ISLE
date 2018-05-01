#include "flash_prod_op.h"
#include "ks_utils.h"
#include "blas-on-flash/include/blas_utils.h"
#include "blas-on-flash/include/flash_blas.h"
#include "blas-on-flash/include/lib_funcs.h"
#include "blas-on-flash/include/scheduler/scheduler.h"

namespace flash {
  extern Scheduler sched;
  extern uint64_t get_next_blk_size(MKL_INT *offs_ptr, MKL_INT nrows,
                                MKL_INT min_size, MKL_INT max_size);
}  // namespace flash

namespace {
  void fill_blocks(MKL_INT *offs, uint64_t n_rows, std::vector<uint64_t> &blk_sizes,
                   std::vector<uint64_t> &offsets) {
    uint64_t cur_start = 0;
    for (; cur_start < n_rows;) {
      uint64_t cblk_size = flash::get_next_blk_size(
          offs + cur_start, n_rows - cur_start, 8, MAX_CSRMM_BLK_SIZE);
      blk_sizes.push_back(cblk_size);
      offsets.push_back(cur_start);
      cur_start += cblk_size;
      GLOG_DEBUG("choosing blk_size=", cblk_size);
    }
  }
}  // namespace

ReusableCsrmmTask::ReusableCsrmmTask(const MKL_INT *ia, flash_ptr<MKL_INT> ja,
                                     flash_ptr<ARMA_FPTYPE> a, const uint64_t a_rows,
                                     const uint64_t a_cols, const uint64_t b_cols) {
  this->a_nrows = a_rows;
  this->a_ncols = a_cols;
  this->b_ncols = b_cols;
  this->ia = new MKL_INT[this->a_nrows + 1];
  for (uint64_t i = 0; i <= this->a_nrows; i++) {
    this->ia[i] = ia[i] - ia[0];
  }
  this->nnzs = this->ia[a_nrows] - this->ia[0];
  GLOG_DEBUG("Using nrows=", this->a_nrows, ", nnzs=", this->nnzs);

  this->ja = ja;
  this->a = a;
  uint64_t ja_start = ROUND_DOWN(this->ja.foffset, SECTOR_LEN);
  uint64_t ja_end = ROUND_UP((this->ja + nnzs).foffset, SECTOR_LEN);
  uint64_t a_start = ROUND_DOWN(this->a.foffset, SECTOR_LEN);
  uint64_t a_end = ROUND_UP((this->a + nnzs).foffset, SECTOR_LEN);
  this->ja_offset = this->ja.foffset - ja_start;
  this->a_offset = this->a.foffset - a_start;
  this->ja.foffset = ja_start;
  this->a.foffset = a_start;

  this->b = nullptr;
  this->c = nullptr;

  StrideInfo sinfo;
  sinfo.n_strides = 1;
  sinfo.len_per_stride = ja_end - ja_start;
  sinfo.stride = 0;
  this->add_read(this->ja, sinfo);
  sinfo.len_per_stride = a_end - a_start;
  this->add_read(this->a, sinfo);

}

ReusableCsrmmTask::~ReusableCsrmmTask() {
  delete[] this->ia;
}

void ReusableCsrmmTask::execute() {
  // GLOG_DEBUG("using a_ptr=", this->in_mem_ptrs[this->a],
  //            ", ja_ptr=", this->in_mem_ptrs[this->ja]);
  MKL_INT old_nthreads = mkl_set_num_threads_local(FLASH_CSRMM_MKL_NTHREADS);
	// GLOG_INFO("old_nthreads=", old_nthreads);
  ARMA_FPTYPE *a_ptr =
      (ARMA_FPTYPE *) offset_buf(this->in_mem_ptrs[this->a], this->a_offset);
  MKL_INT *ja_ptr =
      (MKL_INT *) offset_buf(this->in_mem_ptrs[this->ja], this->ja_offset);

  // prepare csrmm parameters
  CHAR    trans_a = 'N';
  MKL_INT m = (MKL_INT) this->a_nrows;
  MKL_INT n = (MKL_INT) this->b_ncols;
  MKL_INT k = (MKL_INT) this->a_ncols;
  CHAR    matdescra[5] = {'G', 'X', 'X', 'C', 'X'};

  // GLOG_DEBUG("mkl_in_params:m=", m, ", n=", n, ", k=", k,
  //            ", nnzs=", this->nnzs);

  // execute csrmm
  mkl_csrmm(&trans_a, &m, &n, &k, &this->alpha, &matdescra[0], a_ptr, ja_ptr,
            this->ia, this->ia + 1, this->b, &n, &this->beta, this->c, &n);
	// restore MKL num threads for this thread
  mkl_set_num_threads_local(old_nthreads);
}

void ReusableCsrmmTask::reset(ARMA_FPTYPE *new_b, ARMA_FPTYPE *new_c) {
  this->in_mem_ptrs.clear();
  this->set_status(flash::Wait);
  this->b = new_b;
  this->c = new_c;
}

FlashProdOp::FlashProdOp(flash_ptr<ARMA_FPTYPE> a_csr, flash_ptr<MKL_INT> a_col,
                         flash_ptr<ARMA_FPTYPE> a_tr_csr, flash_ptr<MKL_INT> a_tr_col,
                         MKL_INT *a_off_ptr, MKL_INT *a_tr_off_ptr, uint64_t a_rows,
                         uint64_t a_cols, uint64_t nnzs, uint64_t blk_size){
  this->a_nrows = a_rows;
  this->a_ncols = a_cols;
  this->dim = std::min(a_nrows, a_ncols);
  this->inner_dim = std::max(a_nrows, a_ncols);
  this->trans_first = (this->a_nrows == this->dim);

  this->a_csr = a_csr;
  this->a_col = a_col;
  this->a_tr_csr =a_tr_csr;
  this->a_tr_col = a_tr_col;

  // read in offsets array during startup
  this->a_off_ptr = a_off_ptr;
  GLOG_ASSERT(this->a_off_ptr[a_rows] == nnzs,
              " nnzs mismatch; expected=", nnzs,
              ", found=", this->a_off_ptr[a_rows] == nnzs);
  this->a_tr_off_ptr = a_tr_off_ptr;
  GLOG_ASSERT(this->a_tr_off_ptr[a_cols] == nnzs,
              " nnzs mismatch; expected=", nnzs,
              ", found=", this->a_tr_off_ptr[a_cols] == nnzs);
  
  this->blk_size = blk_size;
  this->temp_ptr = new ARMA_FPTYPE[this->inner_dim * this->blk_size];

  this->prev_n_compute_thr = sched.get_num_compute_threads();
  sched.set_num_compute_threads(BLOCK_KS_COMPUTE_THR);

  SchedulerOptions opts;
  opts.enable_prioritizer = false;
  opts.enable_overlap_check = false;
  opts.single_use_discard = true;
	sched.flush_cache();
  sched.set_options(opts);

  // setup tasks
  setup_tasks();
}

FlashProdOp::~FlashProdOp() {
  destroy_tasks();
  delete[] this->temp_ptr;

  // restore num compute threads
  sched.set_num_compute_threads(this->prev_n_compute_thr);

  SchedulerOptions opts;
  opts.enable_prioritizer = true;
  opts.enable_overlap_check = true;
  opts.single_use_discard = false;
  sched.set_options(opts);
  mkl_set_num_threads_local(0);
	sched.flush_cache();
}

void FlashProdOp::destroy_tasks() {
  for (uint64_t i = 0; i < this->n_a_tasks; i++) {
    delete this->a_tasks[i];
  }
  delete[] this->a_tasks;

  for (uint64_t i = 0; i < this->n_a_tr_tasks; i++) {
    delete this->a_tr_tasks[i];
  }
  delete[] this->a_tr_tasks;
}

ARMA_FPMAT FlashProdOp::multiply(ARMA_FPMAT m_in) {
  timer.start();
  ARMA_FPMAT rm_in = arma::trans(m_in).eval();
  ARMA_FPMAT m_out(m_in.n_cols, this->dim);
  n_mms++;
  if ((MKL_INT) m_in.n_cols == this->blk_size) {
    // dispatch setup tasks
    ARMA_FPTYPE *new_b = rm_in.memptr();
    ARMA_FPTYPE *new_c = this->temp_ptr;
    if (this->trans_first) {
      for (uint64_t i = 0; i < this->n_a_tr_tasks; i++) {
        auto blk_c = new_c + (this->a_tr_blk_offsets[i] * this->blk_size);
        this->a_tr_tasks[i]->reset(new_b, blk_c);
        ::sched.add_task(this->a_tr_tasks[i]);
      }
      sleep_wait_for_complete(this->a_tr_tasks, this->n_a_tr_tasks);
      GLOG_DEBUG("finished A^T * B");
      new_b = this->temp_ptr;
      new_c = m_out.memptr();
      for (uint64_t i = 0; i < this->n_a_tasks; i++) {
        auto blk_c = new_c + (this->a_blk_offsets[i] * this->blk_size);
        this->a_tasks[i]->reset(new_b, blk_c);
        ::sched.add_task(this->a_tasks[i]);
      }
      sleep_wait_for_complete(this->a_tasks, this->n_a_tasks);
      GLOG_DEBUG("finished A * A^T");
    } else {
      for (uint64_t i = 0; i < this->n_a_tasks; i++) {
        auto blk_c = new_c + (this->a_blk_offsets[i] * this->blk_size);
        this->a_tasks[i]->reset(new_b, blk_c);
        ::sched.add_task(this->a_tasks[i]);
      }
      sleep_wait_for_complete(this->a_tasks, this->n_a_tasks);
      new_b = this->temp_ptr;
      new_c = m_out.memptr();
      for (uint64_t i = 0; i < this->n_a_tr_tasks; i++) {
        auto blk_c = new_c + (this->a_tr_blk_offsets[i] * this->blk_size);
        this->a_tr_tasks[i]->reset(new_b, blk_c);
        ::sched.add_task(this->a_tr_tasks[i]);
      }
      sleep_wait_for_complete(this->a_tr_tasks, this->n_a_tr_tasks);
    }
  } else {
    // call into flash::csrmm
    ARMA_FPMAT m_temp(m_in.n_cols, this->inner_dim);
    if (trans_first) {
      // Perform A*A^T*m_in using flash blas
      flash::csrmm('N', this->a_ncols, this->a_nrows, m_in.n_cols, 1.0f, 0.0f,
                   this->a_tr_csr, this->a_tr_off, this->a_tr_col, 'R',
                   rm_in.memptr(), m_temp.memptr());
      flash::csrmm('N', this->a_nrows, this->a_ncols, m_in.n_cols, 1.0f, 0.0f,
                   this->a_csr, this->a_off, this->a_col, 'R', m_temp.memptr(),
                   m_out.memptr());
    } else {
      // Perform A^T*A*m_in using flash blas
      flash::csrmm('N', this->a_nrows, this->a_ncols, m_in.n_cols, 1.0f, 0.0f,
                   this->a_csr, this->a_off, this->a_col, 'R', rm_in.memptr(),
                   m_temp.memptr());
      flash::csrmm('N', this->a_ncols, this->a_nrows, m_in.n_cols, 1.0f, 0.0f,
                   this->a_tr_csr, this->a_tr_off, this->a_tr_col, 'R',
                   m_temp.memptr(), m_out.memptr());
    }
  }
  ARMA_FPMAT ret = arma::trans(m_out).eval();
  timer.next_time_secs("2x flash::csrmm");
  return ret;
}

void FlashProdOp::setup_tasks() {
  // setup A*b tasks
  ::fill_blocks(this->a_off_ptr, this->a_nrows, this->a_blk_sizes,
                this->a_blk_offsets);
  ::fill_blocks(this->a_tr_off_ptr, this->a_ncols, this->a_tr_blk_sizes,
                this->a_tr_blk_offsets);
  this->n_a_tasks = this->a_blk_offsets.size();
  this->n_a_tr_tasks = this->a_tr_blk_offsets.size();
  this->a_tasks = new ReusableCsrmmTask *[n_a_tasks];
  this->a_tr_tasks = new ReusableCsrmmTask *[n_a_tr_tasks];
  for (uint64_t i = 0; i < n_a_tasks; i++) {
    MKL_INT *start_off = a_off_ptr + this->a_blk_offsets[i];
    uint64_t     nnzs = start_off[a_blk_sizes[i]] - *start_off;
    GLOG_DEBUG("nnzs=", nnzs);
    uint64_t vals_offset = *start_off;
    this->a_tasks[i] = new ReusableCsrmmTask(
        start_off, this->a_col + vals_offset, this->a_csr + vals_offset,
        this->a_blk_sizes[i], this->a_ncols, this->blk_size);
  }
  for (uint64_t i = 0; i < n_a_tr_tasks; i++) {
    MKL_INT *start_off = a_tr_off_ptr + this->a_tr_blk_offsets[i];
    uint64_t     nnzs = start_off[a_tr_blk_sizes[i]] - *start_off;
    GLOG_DEBUG("nnzs=", nnzs);
    uint64_t vals_offset = *start_off;
    this->a_tr_tasks[i] = new ReusableCsrmmTask(
        start_off, this->a_tr_col + vals_offset, this->a_tr_csr + vals_offset,
        this->a_tr_blk_sizes[i], this->a_nrows, this->blk_size);
  }
}
