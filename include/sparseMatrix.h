// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <mkl.h>

#include <set>
#include <limits>

#include "types.h"
// types.h is included before influences the behavior of Eigen and, therefore, Spectra
#include "spectra-master/include/SymEigsSolver.h"

#include "parallel.h"
#include "utils.h"
#include "matUtils.h"
#include "hyperparams.h"
#include "denseMatrix.h"

namespace ISLE
{
    template<class T>
    class SparseMatrix // : public Matrix<T> 
    {
    protected:
        word_id_t	_vocab_size;	// Size of the vocabulary
        doc_id_t	_num_docs;		// Number of documents
        offset_t    _nnzs;			// Number of non-zero entries
                                    //  3-row Compressed Sparse Column format, 0-index-based
        T			*vals_CSC;		// vals_CSC array 
        word_id_t	*rows_CSC;		// Row array for non-zero word count
        offset_t	*offsets_CSC;		// offsets_CSC array

        T			*normalized_vals_CSC;	// normalized values

        T			avg_doc_sz;	// Average size of a doc

    public:
        SparseMatrix(
            const word_id_t d,
            const doc_id_t s,
            const offset_t nnzs = 0);

        ~SparseMatrix();

        inline offset_t  get_nnzs()		const { return _nnzs; }
        inline doc_id_t  num_docs()		const { return _num_docs; }
        inline word_id_t vocab_size()	const { return _vocab_size; }

        inline offset_t	 offset_CSC(doc_id_t doc)		const { return offsets_CSC[doc]; }
        inline word_id_t row_CSC(offset_t pos)			const { return rows_CSC[pos]; }
        inline T		 val_CSC(offset_t pos)			const { return vals_CSC[pos]; }
        inline T		 normalized_val_CSC(offset_t pos)	const { return normalized_vals_CSC[pos]; }

        inline void set_row_CSC(const offset_t& pos, const word_id_t& word) { rows_CSC[pos] = word; }
        inline void set_val_CSC(const offset_t& pos, const T& val) { vals_CSC[pos] = val; }
        inline void set_offset_CSC(const doc_id_t& doc, const offset_t& off) { offsets_CSC[doc] = off; }

        inline T normalized(word_id_t word, doc_id_t doc) const
        {
            assert(normalized_vals_CSC && rows_CSC && offsets_CSC);
            auto location = std::find(rows_CSC + offsets_CSC[doc],
                rows_CSC + offsets_CSC[doc + 1], word);
            return (location == rows_CSC + offsets_CSC[doc + 1])
                ? (T)0.0 : normalized_vals_CSC[location - rows_CSC];
        }

        void allocate(const offset_t nnzs_);

        inline T elem(
            const word_id_t& word,
            const doc_id_t& doc) const
        {
            assert(vals_CSC && rows_CSC && offsets_CSC);
            auto location = std::find(rows_CSC + offsets_CSC[doc],
                rows_CSC + offsets_CSC[doc + 1], word);
            return (location == rows_CSC + offsets_CSC[doc + 1])
                ? (T)0.0 : vals_CSC[location - rows_CSC];
        }

        void populate_CSC(const std::vector<DocWordEntry<count_t> >& entries);

        template<class FPTYPE>
        FPTYPE doc_norm(
            doc_id_t doc,
            std::function<FPTYPE(const FPTYPE&, const FPTYPE&)> norm_fn);

        void normalize_docs(
            bool delete_unnormalized = false,
            bool normalize_to_one = false);

        size_t count_distint_top_five_words(int min_distinct);

        void list_word_freqs_r(
            std::vector<T>* freqs,
            const doc_id_t d_b,
            const doc_id_t d_e,
            const word_id_t v_b,
            const word_id_t v_e);

        void list_word_freqs(std::vector<T>* freqs);

        void list_word_freqs_by_sorting(std::vector<A_TYPE>* freqs);

        // Input: @num_topics, @freqs: one vector for each word listing its non-zero freqs in docs
        // Output: @zetas: Reference to cutoffs frequencies for each word
        // Return: Total number of entries that are above threshold
        offset_t compute_thresholds(
            std::vector<T>& zetas,
            const doc_id_t num_topics);

        // TODO: Optimize this
        // Input: @r,  @doc_partition: list of docs in this partition
        // Ouput: @thresholds: the @k-th highest count of each word in @doc_partition, length = vocab_size()
        void rth_highest_element(
            const MKL_UINT r,
            const std::vector<doc_id_t>& doc_partition,
            T *thresholds);

        // Input: @num_topics, @thresholds: Threshold for (words,topic)
        // Output: @catchwords: list of catchwords for each topic
        void find_catchwords(
            const doc_id_t num_topics,
            const T *const thresholds,
            std::vector<word_id_t> *catchwords);

        void construct_topic_model(
            DenseMatrix<FPTYPE>& Model,
            const doc_id_t num_topics,
            const std::vector<doc_id_t> *const closest_docs,
            const std::vector<word_id_t> *const catchwords,
            bool avg_null_topics,
            std::vector<std::tuple<int, int, doc_id_t> >* top_topic_pairs = NULL,
            std::vector<std::pair<word_id_t, int> >* catchword_topics = NULL,
            std::vector<std::tuple<doc_id_t, doc_id_t, FPTYPE> >*  doc_topic_sum = NULL);

        // Input: @topic, @M: how many dominant words from topic to pick,  @model: model
        // Output: @top_words: output the top words used to compute coherence
        // Return topic coherence of @test w.r.t. this model
        void topic_coherence(
            const doc_id_t num_topics,
            const word_id_t& M,
            const DenseMatrix<FPTYPE>& model,
            std::vector<std::pair<word_id_t, FPTYPE> >* top_words,
            std::vector<FPTYPE>& coherences,
            const FPTYPE coherence_eps = DEFAULT_COHERENCE_EPS);

        // Output: @joint_counts: joint_counts[i][j] contains joint freq for j<i
        void compute_joint_doc_frequency(
            const int num_topics,
            const std::vector<std::pair<word_id_t, FPTYPE> >* top_words,
            std::vector<std::vector<std::vector<size_t> > >& joint_counts)
            const;

        // Convert to algebra. Equivalent to "this*column(1)"
        void compute_doc_frequency(
            const int num_topics,
            const std::vector<std::pair<word_id_t, FPTYPE> >* top_words,
            std::vector<std::vector<size_t> >& doc_frequencies)
            const;

        void compute_log_combinatorial(std::vector<FPTYPE>& docs_log_fact);

        friend class FPSparseMatrix<FPTYPE>;
    };

    template<class FPTYPE>
    class FPSparseMatrix : public SparseMatrix<FPTYPE>
    {
        Eigen::MatrixX BBT;				// For Spectra EigenSolve
        //Eigen::MatrixX U_Spectra;		// first few eig vectors of this*this^T
        FPTYPE *U_colmajor;             // First num_topics eigenvectors of this*this^T
        FPTYPE *U_rowmajor;             // U_colmajor in row-major form
        MKL_INT U_rows;
        MKL_INT U_cols;
        FPTYPE *SigmaVT;			    // Sigma * VT obtained from eigensolvers 
        MKL_INT num_singular_vals;		// Number of Singular values

    public:
        friend class FPDenseMatrix<FPTYPE>;

        using SparseMatrix<FPTYPE>::allocate;
        using SparseMatrix<FPTYPE>::num_docs;
        using SparseMatrix<FPTYPE>::vocab_size;
        using SparseMatrix<FPTYPE>::get_nnzs;
        using SparseMatrix<FPTYPE>::row_CSC;
        using SparseMatrix<FPTYPE>::val_CSC;
        using SparseMatrix<FPTYPE>::offset_CSC;
        using SparseMatrix<FPTYPE>::normalized_val_CSC;
        using SparseMatrix<FPTYPE>::rows_CSC;
        using SparseMatrix<FPTYPE>::vals_CSC;
        using SparseMatrix<FPTYPE>::offsets_CSC;
        using SparseMatrix<FPTYPE>::normalized_vals_CSC;
        using SparseMatrix<FPTYPE>::_nnzs;
        using SparseMatrix<FPTYPE>::_num_docs;

        FPSparseMatrix(const word_id_t d, const doc_id_t s);

        ~FPSparseMatrix();

        FPSparseMatrix(
            const SparseMatrix<FPTYPE>& from,
            const bool copy_normalized = false);

        FPTYPE frobenius() const;

        FPTYPE normalized_frobenius() const;

        struct WordDocPair
        {
            word_id_t word;
            doc_id_t doc;
            WordDocPair(const word_id_t& word_, const doc_id_t& doc_);
            WordDocPair(const WordDocPair& from);
        };

        void get_word_major_list(
            std::vector<WordDocPair>& entries,
            std::vector<offset_t>& word_offsets);

        void initialize_for_eigensolver(const doc_id_t num_topics);

        void cleanup_after_eigensolver();

        void compute_Spectra(
            const doc_id_t num_topics,
            std::vector<FPTYPE>& evalues);

        void compute_block_ks(
            const doc_id_t num_topics,
            std::vector<FPTYPE>& evalues);

        void compute_U_rowmajor();
        void compute_sigmaVT(const doc_id_t num_topics);

        // Input: @from: Copy from here
        // Input: @zetas: zetas[word] indicates the threshold for each word
        // Input: @nnzs: Number of nnz elements that would remain after thresholding, pre-calculated
        // Input: @drop_empty: Drop all-zero cols while converting?
        // Output: @original_cols: For remaining cols, id to original cols, if drop_empty==true
        template <class fromT>
        void threshold_and_copy(
            const SparseMatrix<fromT>& from,
            const std::vector<fromT>& zetas,
            const offset_t nnzs,
            std::vector<doc_id_t>& original_cols);

        template <class fromT>
        void sampled_threshold_and_copy(
            const SparseMatrix<fromT>& from,
            const std::vector<fromT>& zetas,
            const offset_t nnzs,
            std::vector<doc_id_t>& original_cols,
            const FPTYPE sample_rate);

        // Input: @in: assumed col-major, must be initialized to size >= @ncols * @k 
        // Input: @ld_in: leading dimension of input columns
        // Input: @ncols: number of cols in input and output
        // Output: @out: must be init'd to size>= @ncols * this->vocab_size. Col_major.
        void left_multiply_by_U_Spectra(
            FPTYPE *const out,
            const FPTYPE *in,
            const doc_id_t ld_in,
            const doc_id_t ncols);

        void copy_col_to(
            FPTYPE *const dst,
            const doc_id_t doc) const;

        // pt must have at least vocab_size() entries
        inline FPTYPE distsq_doc_to_pt(
            const doc_id_t doc,
            const FPTYPE *const pt,
            const FPTYPE pt_l2sq = -1.0) const;

        inline FPTYPE distsq_normalized_doc_to_pt(
            const doc_id_t doc,
            const FPTYPE *const pt,
            const FPTYPE pt_l2sq = -1.0) const;

        // Output @dist_matrix: distance of each point to all centers, initialize to num_centers*num_docs() 
        // Input: @dim: the number of dimensions for centers and docs
        // Input: @num_centers: num of centers
        // Input: @centers: column major matrix with coords of centers.
        // Input: @centers_l2sq: l_2^2 of centers
        // Input: @docs_l2sq: l_2^2 of points,
        void distsq_docs_to_centers(
            const word_id_t dim,
            doc_id_t num_centers,
            const FPTYPE *const centers,
            const FPTYPE *const centers_l2sq,
            const doc_id_t doc_begin,
            const doc_id_t doc_end,
            const FPTYPE *const docs_l2sq,
            FPTYPE *dist_matrix);

        void closest_centers(
            const doc_id_t num_centers,
            const FPTYPE *const centers,
            const FPTYPE *const centers_l2sq,
            const doc_id_t doc_begin,
            const doc_id_t doc_end,
            const FPTYPE *const docs_l2sq, // 0-th element corresponds to doc_begin
            doc_id_t *center_index,        // 0-th element corresponds to doc_begin
            FPTYPE *const dist_matrix);    // Initialized to num_centers*(doc_end-doc_begin) size 

        void compute_centers_l2sq(
            FPTYPE * centers,
            FPTYPE * centers_l2sq,
            const doc_id_t num_centers);

        void compute_docs_l2sq(FPTYPE *const docs_l2sq);

        FPTYPE lloyds_iter(
            const doc_id_t num_centers,
            FPTYPE *centers,
            const FPTYPE *const docs_l2sq,
            std::vector<doc_id_t> *closest_docs = NULL,
            bool compute_residual = false);

        FPTYPE run_lloyds(
            const doc_id_t			num_centers,
            FPTYPE					*centers,
            std::vector<doc_id_t>	*closest_docs, // Pass NULL if you dont want closest_docs
            const int				max_reps);

        //
        // U^T x docs in range [doc_begin, doc_end) 
        // Output is projected docs in column-major order
        //
        void UT_times_docs(
            const doc_id_t doc_begin,
            const doc_id_t doc_end,
            FPTYPE* const projected_docs);

        //
        // Let A denote the block of data [doc_begin, doc_end)
        // This function computes out := A^T * in
        // out is row-major with size (doc_end - doc_end) * cols
        // in is row-major with size vocab_size() * cols
        //
        void multiply_with(
            const doc_id_t doc_begin,
            const doc_id_t doc_end,
            const FPTYPE *const in,
            FPTYPE *const out,
            const MKL_INT cols);

        void distsq_projected_docs_to_projected_centers(
            const word_id_t dim,
            doc_id_t num_centers,
            const FPTYPE *const projected_centers_tr,
            const FPTYPE *const projected_centers_l2sq,
            const doc_id_t doc_begin,
            const doc_id_t doc_end,
            const FPTYPE *const projected_docs_l2sq,
            FPTYPE *projected_dist_matrix);

        // same input semantics as `closest_centers`
        void projected_closest_centers(
            const doc_id_t num_centers,
            const FPTYPE *const projected_centers_tr,
            const FPTYPE *const projected_centers_l2sq,
            const doc_id_t doc_begin,
            const doc_id_t doc_end,
            const FPTYPE *const projected_docs_l2sq, 
            doc_id_t *center_index,                  
            FPTYPE *const projected_dist_matrix); 

        void compute_projected_centers_l2sq(
            FPTYPE * projected_centers,
            FPTYPE * projected_centers_l2sq,
            const doc_id_t num_centers);

        void compute_projected_docs_l2sq(
            FPTYPE *const projected_docs_l2sq);

        FPTYPE lloyds_iter_on_projected_space(
            const doc_id_t num_centers,
            FPTYPE *projected_centers,
            const FPTYPE *const projected_docs_l2sq,
            std::vector<doc_id_t> *closest_docs = NULL,
            bool compute_residual = false);

        FPTYPE run_lloyds_on_projected_space(
            const doc_id_t			num_centers,
            FPTYPE					*projected_centers, // centers are already projected to U^T
            std::vector<doc_id_t>	*closest_docs, // Pass NULL if you dont want closest_docs
            const int				max_reps);

        void update_min_distsq_to_projected_centers(
            const word_id_t dim,
            const doc_id_t num_centers,
            const FPTYPE *const projected_centers,
            const doc_id_t doc_begin,
            const doc_id_t doc_end,
            const FPTYPE *const projected_docs_l2sq,
            FPTYPE *min_dist,
            FPTYPE *projected_dist);

        FPTYPE kmeanspp_on_projected_space(
            const doc_id_t k,
            std::vector<doc_id_t>&centers);

        FPTYPE kmeans_init_on_projected_space(
            const int num_centers,
            const int max_reps,
            std::vector<doc_id_t>&	best_seed,   // Wont be initialized if method==KMEANSBB
            FPTYPE *const			best_centers_coords); // Wont be initialized if null

        // Input: @num_centers, @centers: coords of centers to start the iteration, @print_residual
        // Output: @closest_docs: if NULL, nothing is returned; is !NULL, return partition of docs between centers
        FPTYPE run_elkans(
            const doc_id_t			num_centers,
            FPTYPE					*centers,
            std::vector<doc_id_t>	*closest_docs, // Pass NULL if you dont want closest_docs
            const int				max_reps);
    };
}
