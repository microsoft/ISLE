// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "types.h"
#include "timer.h"

namespace ISLE
{
    template<class FPTYPE>
    class MKL_SpSpTrProd
    {
        FPTYPE		*vals_CSC;
        vocabSz_t	*rows_CSC;
        offset_t	*offsets_CSC;
        bool		 is_offsets_CSC_alloc; // true when nrows>ncols

        const Eigen::Index nrows;
        const Eigen::Index ncols;
        const Eigen::Index max_dim;
        const offset_t	   nnzs;

        bool		split_CSR_by_rows;
        bool		split_CSR_by_cols;

        FPTYPE		*vals_CSR;
        MKL_INT		*cols_CSR;
        MKL_INT		*offsets_CSR;

        FPTYPE		*temp;
        FPTYPE		*y_temp;

        size_t		num_col_blocks;
        FPTYPE		**vals_CSR_arr;
        MKL_INT		**cols_CSR_arr;
        MKL_INT		**offsets_CSR_arr;

        size_t		num_row_blocks;
        size_t		*row_block_offsets;
        FPTYPE		row_block_exp;

        Timer		*op_timer;
        double		op_user_time_sum;
        double		op_sys_time_sum;
        uint64_t	num_op_calls;
    public:
        MKL_SpSpTrProd(
            FPTYPE *vals_CSC_, vocabSz_t *rows_CSC_, offset_t *offsets_CSC_,
            const Eigen::Index nrows_, const Eigen::Index ncols_, const offset_t nnzs_,
            const bool split_CSR_by_cols_ = false,
            const bool split_CSR_by_rows_ = true,
            const bool check = true)
            :
            vals_CSC(vals_CSC_), rows_CSC(rows_CSC_),
            nrows(nrows_), ncols(ncols_), nnzs(nnzs_),
            max_dim(ncols_ > nrows_ ? ncols_ : nrows_),
            split_CSR_by_rows(split_CSR_by_rows_),
            split_CSR_by_cols(split_CSR_by_cols_)
        {
            assert(sizeof(vocabSz_t) == sizeof(MKL_INT));
            assert(sizeof(offset_t) == sizeof(MKL_INT));
            assert(!(split_CSR_by_rows && split_CSR_by_cols));

            if (max_dim > ncols) { // Pad offsets to make matrix square
                is_offsets_CSC_alloc = true;
                offsets_CSC = new offset_t[max_dim + 1];
                memcpy(offsets_CSC, offsets_CSC_, sizeof(offset_t) * (ncols + 1));
                for (auto col = ncols + 1; col <= max_dim; ++col)
                    offsets_CSC[col] = nnzs;
            }
            else {
                is_offsets_CSC_alloc = false;
                offsets_CSC = offsets_CSC_;
            }

            temp = new FPTYPE[max_dim + nrows];
            y_temp = new FPTYPE[max_dim];

            // Convert CSC to one big CSR
            vals_CSR = new FPTYPE[nnzs];
            cols_CSR = new MKL_INT[nnzs];
            offsets_CSR = new MKL_INT[max_dim + 1];

            const MKL_INT job[6] = { 1,0,0,0,0,1 }; // First 1: CSC->CSR, last 1: fill acsr, ja,ia
            const MKL_INT m = max_dim;
            MKL_INT info = 0;

            FPcsrcsc(job, &m,
                vals_CSR, cols_CSR, offsets_CSR,
                vals_CSC, (MKL_INT*)rows_CSC, (MKL_INT*)offsets_CSC,
                &info); // info is useless

            if (check) {
                for (auto i = nrows; i <= max_dim; ++i)
                    assert(offsets_CSR[i] == nnzs);
                for (auto r = 0; r < nrows; ++r)
                    for (auto pos = offsets_CSR[r]; pos < offsets_CSR[r + 1]; ++pos) {
                        assert(vals_CSR[pos] > 0);
                        if (pos < offsets_CSR[r + 1] - 1)
                            assert(cols_CSR[pos] < cols_CSR[pos + 1]);
                        assert(cols_CSR[pos] < ncols && cols_CSR[pos] >= 0);
                    }
            }

            if (nrows >= ncols) {
                if (split_CSR_by_cols)
                    std::cout << "\n === WARNING: turning off split CSR by cols\n" << std::endl;
                split_CSR_by_cols = false;
            }

            if (split_CSR_by_cols) { // Split CSR into square pieces
                assert(ncols > nrows);
                num_col_blocks = ncols % nrows == 0 ? ncols / nrows : ncols / nrows + 1;
                vals_CSR_arr = new FPTYPE*[num_col_blocks];
                cols_CSR_arr = new MKL_INT*[num_col_blocks];
                offsets_CSR_arr = new MKL_INT*[num_col_blocks];

                auto nnzs_arr = new MKL_INT[num_col_blocks];
                auto pos_arr = new size_t[num_col_blocks];
                for (auto block = 0; block < num_col_blocks; ++block) {
                    nnzs_arr[block] = 0;
                    pos_arr[block] = 0;
                    offsets_CSR_arr[block] = new MKL_INT[nrows + 1];
                    offsets_CSR_arr[block][0] = 0;
                }

                for (auto pos = 0; pos < offsets_CSR[nrows]; ++pos)
                    ++nnzs_arr[cols_CSR[pos] / nrows];
                size_t nnzs_sum = 0;
                for (auto block = 0; block < num_col_blocks; ++block) nnzs_sum += nnzs_arr[block];
                assert(nnzs_sum == offsets_CSR[nrows]);

                for (auto block = 0; block < num_col_blocks; ++block) {
                    vals_CSR_arr[block] = new FPTYPE[nnzs_arr[block]];
                    cols_CSR_arr[block] = new MKL_INT[nnzs_arr[block]];
                }

                for (auto row = 0; row < nrows; ++row) {
                    size_t prev_block = 0;
                    size_t prev_pos = offsets_CSR[row];
                    size_t pos = offsets_CSR[row];
                    for (pos = offsets_CSR[row]; pos < offsets_CSR[row + 1]; ++pos) {
                        auto block = cols_CSR[pos] / nrows;
                        if (prev_block < block) {
                            offsets_CSR_arr[prev_block][row + 1] = offsets_CSR_arr[prev_block][row] + (pos - prev_pos);
                            prev_block++;

                            while (prev_block < block) {
                                offsets_CSR_arr[prev_block][row + 1] = offsets_CSR_arr[prev_block][row];
                                ++prev_block;
                            }

                            prev_pos = pos;
                            prev_block = block;
                        }

                        assert(pos_arr[block] < nnzs_arr[block]);
                        vals_CSR_arr[block][pos_arr[block]] = vals_CSR[pos];
                        cols_CSR_arr[block][pos_arr[block]] = cols_CSR[pos] - block*nrows;
                        ++pos_arr[block];
                    }
                    assert(prev_block < num_col_blocks);
                    {
                        offsets_CSR_arr[prev_block][row + 1] = offsets_CSR_arr[prev_block][row] + (pos - prev_pos);
                        prev_block++;

                        while (prev_block < num_col_blocks) {
                            offsets_CSR_arr[prev_block][row + 1] = offsets_CSR_arr[prev_block][row];
                            ++prev_block;
                        }
                    }
                }
                for (auto block = 0; block < num_col_blocks; ++block) {
                    assert(pos_arr[block] == nnzs_arr[block]);
                    assert(offsets_CSR_arr[block][nrows] == nnzs_arr[block]);
                }

                delete[] pos_arr;
                delete[] nnzs_arr;
            }

            if (split_CSR_by_rows) {
                size_t row_block_size = 32;
                row_block_exp = 1.25;
                num_row_blocks = (std::log2(1 + (nrows / row_block_size)) / std::log2(row_block_exp));
                row_block_offsets = new size_t[num_row_blocks + 20];
                row_block_offsets[0] = 0;
                int i = 0;
                while (1) {
                    row_block_offsets[i + 1] = row_block_offsets[i] + std::pow(row_block_exp, i)*row_block_size;
                    if (row_block_offsets[i + 1] >= nrows) {
                        row_block_offsets[i + 1] = nrows;
                        num_row_blocks = i + 1;
                        break;
                    }
                    i++;
                }
                assert(row_block_offsets[num_row_blocks - 1] < nrows);
                assert(row_block_offsets[num_row_blocks] == nrows);
            }

            op_timer = new Timer;
            op_user_time_sum = op_sys_time_sum = 0.0;
            num_op_calls = 0;
        }

        ~MKL_SpSpTrProd()
        {
            if (is_offsets_CSC_alloc)
                delete[] offsets_CSC;

            delete[] temp;
            delete[] y_temp;

            delete[] vals_CSR;
            delete[] cols_CSR;
            delete[] offsets_CSR;

            if (split_CSR_by_cols) {
                for (auto block = 0; block < num_col_blocks; ++block) {
                    delete[] offsets_CSR_arr[block];
                    delete[] vals_CSR_arr[block];
                    delete[] cols_CSR_arr[block];
                }
            }
            if (split_CSR_by_rows) {
                assert(row_block_offsets != NULL);
                delete[] row_block_offsets;
            }

            std::cout << "Time spent in matvecs: "
                << op_user_time_sum << "s(user) "
                << op_sys_time_sum << "s(sys)" << std::endl;
            std::cout << "#Calls to matvecs: " << num_op_calls << std::endl;
            delete op_timer;
        }

        Eigen::Index rows() const { return nrows; }
        Eigen::Index cols() const { return ncols; }

        void perform_op(
            FPTYPE *x_in,
            FPTYPE *y_out)
        {
            op_timer->next_time_secs_silent();
            ++num_op_calls;
            const char no_trans = 'N';
            FPcsrgemv(	// Pretend CSC_transpose is CSR.
                &no_trans, (MKL_INT*)&max_dim,
                vals_CSC, (MKL_INT*)offsets_CSC, (MKL_INT*)rows_CSC,
                x_in, temp);

            if (!split_CSR_by_cols && !split_CSR_by_rows) {
                FPcsrgemv(
                    &no_trans, (MKL_INT*)&max_dim,
                    vals_CSR, (MKL_INT*)offsets_CSR, (MKL_INT*)cols_CSR,
                    temp, y_temp);
                //FPblascopy(nrows, y_temp, 1, y_out, 1);
                memcpy(y_out, y_temp, sizeof(FPTYPE) * nrows);
            }
            else if (split_CSR_by_rows) {
                FPscal(nrows, 0.0, y_out, 1);
                {
                    int block_size = 32;
                    size_t num_blocks = nrows % block_size == 0
                        ? nrows / block_size : nrows / block_size + 1;
                    pfor_dynamic_1(int i = 0; i <= num_row_blocks; ++i) {
                        int begin = row_block_offsets[i];
                        int end = row_block_offsets[i + 1];
                        if (end > nrows) end = nrows;
                        for (auto row = begin; row < end; ++row)
                            for (auto pos = offsets_CSR[row]; pos < offsets_CSR[row + 1]; ++pos)
                                y_out[row] += temp[cols_CSR[pos]] * vals_CSR[pos];
                    }
                }
            }
            else if (split_CSR_by_cols) {
                FPscal(nrows, 0.0, y_out, 1);

                for (auto block = 0; block < num_col_blocks; ++block) {
                    FPcsrgemv(
                        &no_trans, (MKL_INT*)&nrows,
                        vals_CSR_arr[block], (MKL_INT*)offsets_CSR_arr[block], (MKL_INT*)cols_CSR_arr[block],
                        temp + block*nrows, y_temp);
                    FPaxpy(nrows, 1.0, y_temp, 1, y_out, 1);

                }
            }
            auto next_time = op_timer->next_time_secs_silent();
            op_user_time_sum += next_time.first;
            op_sys_time_sum += next_time.second;
        }
    };


    template<class FPTYPE>
    class MKL_DenseGenMatProd
    {
        const FPTYPE *data;
        const Eigen::Index nrows, ncols;
        const bool IsRowMajor;

    public:
        MKL_DenseGenMatProd(const Eigen::MatrixX& mat) :
            data(mat.data()),
            nrows(mat.rows()),
            ncols(mat.cols()),
            IsRowMajor(mat.IsRowMajor)
        {}

        Eigen::Index rows() const { return nrows; }
        Eigen::Index cols() const { return ncols; }

        void perform_op(FPTYPE *x_in, FPTYPE *y_out) const
        {
            FPgemv(IsRowMajor ? CblasRowMajor : CblasColMajor, CblasNoTrans,
                (MKL_INT)nrows, (MKL_INT)ncols, 1.0, data, (MKL_INT)(IsRowMajor ? ncols : nrows),
                x_in, 1, 0.0, y_out, 1);
        }
    };


    template<class FPTYPE>
    class MKL_DenseSymMatProd
    {
        const FPTYPE *data;
        const Eigen::Index nrows; // Same as ncols
        const bool IsRowMajor;

    public:
        MKL_DenseSymMatProd(const Eigen::MatrixX& mat) :
            data(mat.data()),
            nrows(mat.rows()),
            IsRowMajor(mat.IsRowMajor)
        {
            assert(mat.rows() == mat.cols());
        }
        Eigen::Index rows() const { return nrows; }

        void perform_op(FPTYPE *x_in, FPTYPE *y_out) const
        {
            FPsymv(IsRowMajor ? CblasRowMajor : CblasColMajor, CblasUpper,
                (MKL_INT)nrows, 1.0, data, (MKL_INT)nrows, x_in, 1, 0.0, y_out, 1);
        }
    };

    inline double rand_fraction()
    {
        const double normalizer = (double)(((uint64_t)RAND_MAX + 1) * ((uint64_t)RAND_MAX + 1));
        return ((double)rand() + (double)rand()*(((double)RAND_MAX + 1.0))) / normalizer;
    }
}

//void bit_outer_prod_seq(
//	FPTYPE *const out,
//	const uint64_t *const mat,
//	const MKL_INT nrows, const MKL_INT ncols,
//	const MKL_INT row1_b, const MKL_INT row1_e,
//	const MKL_INT row2_b, const MKL_INT row2_e,
//	const MKL_INT col_b, const MKL_INT col_e)
//{
//#ifdef OPENMP
//#pragma omp parallel
//#pragma omp single nowait
//#endif
//	for (auto row1 = row1_b; row1 < row1_e; ++row1) {
//		for (auto row2 = row2_b; row2 < row2_e && row2 <= row1; ++row2) {
//			MKL_INT count = 0;
//			size_t row_offset1 = row1 * (ncols / 64);
//			size_t row_offset2 = row2 * (ncols / 64);
//
//			for (size_t blk = col_b / 64; blk < col_e / 64; ++blk)
//				count += __popcnt64(mat[row_offset1 + blk] & mat[row_offset2 + blk]);
//
//			out[row1*nrows + row2] += (FPTYPE)count;
//			if (row1 != row2)
//				out[row2*nrows + row1] += (FPTYPE)count;
//		}
//	}
//}
//
//void bit_outer_prod_r(FPTYPE *const out,
//	const uint64_t *const mat,
//	const MKL_INT nrows, const MKL_INT ncols,
//	const MKL_INT row1_b, const MKL_INT row1_e,
//	const MKL_INT row2_b, const MKL_INT row2_e,
//	const MKL_INT col_b, const MKL_INT col_e) 
//{
//
//	/*std::cout << "r1b:" << row1_b << "  r1e:" << row1_e
//	<< "  r2b:" << row2_b << "  r2e:" << row2_e
//	<< "  cb:" << col_b << "  ce:" << col_e << std::endl;*/
//
//	assert(col_b % 256 == 0 && col_e % 256 == 0);
//	auto row1_diff = row1_e - row1_b;
//	auto row2_diff = row2_e - row2_b;
//	auto col_diff = col_e - col_b;
//
//	const MKL_INT row_blk = 64;
//	const MKL_INT col_blk = (1 << 14);
//
//	if (row1_diff <= row_blk && row2_diff <= row_blk && col_diff <= col_blk) {
//		bit_outer_prod_seq(out, mat, nrows, ncols, row1_b, row1_e, row2_b, row2_e, col_b, col_e);
//	}
//	else if (row1_diff >= row2_diff && row1_diff > row_blk) {
//		bit_outer_prod_r(out, mat, nrows, ncols,
//			row1_b, (row1_b + row1_e) / 2, row2_b, row2_e, col_b, col_e);
//		bit_outer_prod_r(out, mat, nrows, ncols,
//			(row1_b + row1_e) / 2, row1_e, row2_b, row2_e, col_b, col_e);
//	}
//	else if (row2_diff > row_blk) {
//		bit_outer_prod_r(out, mat, nrows, ncols,
//			row1_b, row1_e, row2_b, (row2_b + row2_e) / 2, col_b, col_e);
//		bit_outer_prod_r(out, mat, nrows, ncols,
//			row1_b, row1_e, (row2_b + row2_e) / 2, row2_e, col_b, col_e);
//	}
//	else if (col_diff > col_blk) {
//		auto col_mid = col_b + 256 * (col_diff / (512));
//		assert(col_mid > col_b); assert(col_mid < col_e);
//		bit_outer_prod_r(out, mat, nrows, ncols,
//			row1_b, row1_e, row2_b, row2_e, col_b, col_mid);
//		bit_outer_prod_r(out, mat, nrows, ncols,
//			row1_b, row1_e, row2_b, row2_e, col_mid, col_e);
//	}
//	else {
//		assert(false);
//	}
//}
//
//// Compute out = mat * tr(mat)
//// mat in row-major format, and of size nrows X ncols
//// out is of size nrows X nrows
//void bit_outer_prod(FPTYPE *out,
//	const uint64_t* mat,
//	const MKL_INT nrows,
//	const MKL_INT ncols) 
//{
//	for (size_t i = 0; i < nrows*nrows; ++i)
//		out[i] = 0.0;
//	bit_outer_prod_r(out, mat, nrows, ncols, 0, nrows, 0, nrows, 0, ncols);
//}

//#include <random>
//std::random_device rd; 
//std::mt19937 gen(rd());
//std::uniform_real_distribution<> dis(0, 1);

// Return random number in [0,1)
