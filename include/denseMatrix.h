// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <mkl.h>

#include "Eigen/Dense"

#include "types.h"
#include "parallel.h"
#include "utils.h"
#include "timer.h"
#include "matUtils.h"
#include "hyperparams.h"

namespace ISLE
{
    class WordCountSparseMatrix;
    template<class FPTYPE> class FloatingPointSparseMatrix;

    template<class T>
    class DenseMatrix
    {
    protected:
        vocabSz_t       _vocab_size; // Size of the vocabulary
        docsSz_t        _num_docs;   // Number of documents
        T               *A;          // Storage
        bool            _alloc;      // Allocate storage for A?

    public:

        DenseMatrix(
            const vocabSz_t d_,
            const docsSz_t s_,
            bool alloc = true) // If false, do not allocate A.
            :
            _vocab_size(d_),
            _num_docs(s_),
            _alloc(alloc)
        {
            size_t size = ((size_t)d_ *(size_t)s_);
            if (alloc)
                A = new T[size];
            else
                A = NULL;

            size_t chunk_size = size / 100 > (1 << 22) ? size / 100 : (1 << 22);
            size_t num_chunks = size / chunk_size;
            if (size % chunk_size > 0)
                num_chunks++;
            pfor(int64_t chunk = 0; chunk < num_chunks; ++chunk)
            {
                size_t n = size > (chunk + 1) * chunk_size
                    ? chunk_size
                    : size - chunk*chunk_size;
                std::fill_n(A + chunk*chunk_size, n, (T)0);
            }
        }

        ~DenseMatrix()
        {
            if (_alloc) {
                assert(A != NULL);
                delete[] A;
            }
        }

        inline T*const data() const
        {
            return A;
        }

        // Input: word-id and doc-id, both in 0-based indexing
        // Output: A reference to the entry in the matrix
        inline T& elem_ref(vocabSz_t word, docsSz_t doc)
        {
            return data()[(size_t)word + ((size_t)doc)*(size_t)vocab_size()];
        }

        inline T elem(vocabSz_t word, docsSz_t doc) const
        {
            return data()[(size_t)word + ((size_t)doc)*(size_t)vocab_size()];
        }

        // Copy @n elements starting from (@offset_vocab,@offset_docs) to @dst
        void copy_col_to(T*const dst, const docsSz_t doc) const
        {
            memcpy(dst, data() + (size_t)doc * (size_t)vocab_size(), sizeof(T)*vocab_size());
        }

        void copy_col_from(const T *const src, const docsSz_t doc)
        {
            memcpy(data() + (size_t)doc * (size_t)vocab_size(), src, sizeof(T)*vocab_size());
        }

        inline docsSz_t  num_docs()  const { return _num_docs; }
        inline vocabSz_t vocab_size()const { return _vocab_size; }

        void find_top_words_above_threshold(
            const docsSz_t topic,
            const FPTYPE threshold,
            std::vector<std::pair<vocabSz_t, FPTYPE> >& top_words) const
        {
            assert(top_words.size() == 0);
            for (vocabSz_t word = 0; word < vocab_size(); ++word)
                if (elem(word, topic) > threshold)
                    top_words.push_back(std::make_pair(word, elem(word, topic)));
            std::sort(top_words.begin(), top_words.end(),
                [](auto &left, auto &right) {return left.second > right.second; });
        }

        void print_words_above_threshold(
            const docsSz_t topic,
            const count_t threshold,
            const std::vector<std::string>& vocab_words)
        {
            std::vector<std::pair<vocabSz_t, FPTYPE> > top_words;
            find_top_words_above_threshold(topic, threshold, top_words);
            std::sort(top_words.begin(), top_words.end(),
                [](auto &left, auto &right) {return left.first < right.first; });
            std::cout << "#Dom words: " << top_words.size() << "\n";
            for (auto top_iter = top_words.begin(); top_iter != top_words.end(); ++top_iter)
                std::cout << vocab_words[top_iter->first]
                << ":" << top_iter->first
                << "(" << top_iter->second << ") ";
            std::cout << "\n";
        }

        // Print the @nwords heaviest words for a topic.
        void find_n_top_words(
            const docsSz_t topic,
            const vocabSz_t nwords,
            std::vector<std::pair<vocabSz_t, FPTYPE> >& top_words) const
        {
            assert(top_words.size() == 0);
            for (vocabSz_t word = 0; word < vocab_size(); ++word)
                top_words.push_back(std::make_pair(word, elem(word, topic)));
            std::sort(top_words.begin(), top_words.end(),
                [](auto &left, auto &right) {return left.second > right.second; });
            if (top_words[nwords - 1].second == (FPTYPE)0.0)
                std::cout << "\n ==== WARNING: top words in topic " << topic << " have zero weight\n\n";
            top_words.resize(nwords);
        }

        void print_top_words(
            const std::vector<std::string>& vocab_words,
            const std::vector<std::pair<vocabSz_t, FPTYPE> >& top_words,
            std::ostream& stream = std::cout) const
        {
            stream << "\n#Top words: " << top_words.size() << "\n";
            for (auto top_iter = top_words.begin(); top_iter != top_words.end(); ++top_iter)
                stream << vocab_words[top_iter->first]
                << ":" << top_iter->first
                << "(" << top_iter->second << ") ";
            stream << "\n\n";
        }

        void write_to_file(const std::string& filename) const 
        {
#if FILE_IO_MODE == NAIVE_FILE_IO
            { // Naive File IO
                std::ofstream out_model;
                out_model.open(filename);
                for (docsSz_t topic = 0; topic < num_docs(); ++topic) {
                    for (vocabSz_t word = 0; word < vocab_size(); ++word)
                        out_model << std::setprecision(8) << elem(word, topic) << "\t";
                    out_model << std::endl;
                }
                out_model.close();
            }
#elif FILE_IO_MODE == WIN_MMAP_FILE_IO || FILE_IO_MODE == LINUX_MMAP_FILE_IO
            { // Memory mapped File IO
                MMappedOutput out(filename);
                for (docsSz_t topic = 0; topic < num_docs(); ++topic) {
                    for (vocabSz_t word = 0; word < vocab_size(); ++word) {
                        out.concat_float(elem(word, topic), '\t', 1, 10);
                    }
                    out.add_endline();
                }
                out.flush_and_close();
            }
#else
            assert(false);
#endif
        }

        void write_to_file_as_sparse(
            const std::string& filename,
            const unsigned base = 1) // 0-based or 1-based
            const {
#if FILE_IO_MODE == NAIVE_FILE_IO
            { // Naive File IO
                std::ofstream out_model;
                out_model.open(filename);
                for (docsSz_t topic = 0; topic < num_docs(); ++topic) {
                    for (vocabSz_t word = 0; word < vocab_size(); ++word)
                        if (elem(word, topic) > 0.00000001f)
                            out_model << topic + base << " " << word + base  // Sparse format written in 1-based indexing
                            << " " << std::setprecision(6) << elem(word, topic) << "\n";
                }
                out_model.close();
            }
#elif FILE_IO_MODE == WIN_MMAP_FILE_IO || FILE_IO_MODE == LINUX_MMAP_FILE_IO
            { // Memory mapped File IO
                MMappedOutput out(filename);
                for (docsSz_t topic = 0; topic < num_docs(); ++topic) {
                    for (vocabSz_t word = 0; word < vocab_size(); ++word) {
                        if (elem(word, topic) > 0.00000001f) {
                            out.concat_int(topic + base, '\t');
                            out.concat_int(word + base, '\t');
                            out.concat_float(elem(word, topic), '\n', 1, 10);
                        }
                    }
                }
                out.flush_and_close();
            }
#else
            assert(false);
#endif
        }
    };

    template<class FPTYPE> class FloatingPointDenseMatrix;

    class WordCountDenseMatrix : public DenseMatrix<count_t>
    {
    public:
        WordCountDenseMatrix(vocabSz_t d_, docsSz_t s_) :
            DenseMatrix<count_t>(d_, s_)
        {}

        ~WordCountDenseMatrix()
        { }

        size_t populate(DocWordEntriesReader& data)
        {
            for (auto entry = data.entries.begin(); entry != data.entries.end(); entry++) {
                assert(elem_ref(entry->word, entry->doc) == 0); // No duplicate entries
                elem_ref(entry->word, entry->doc) = entry->count;
            }
            return data.entries.size();
        }
    };

    template<class FPTYPE>
    class FloatingPointDenseMatrix : public DenseMatrix<FPTYPE>
    {
        FPTYPE	*svd_temp;
        FPTYPE	*U;
        FPTYPE	*VT;
        FPTYPE	*Sigma;             // Pointers for full SVD calculation, A=U*Sigma*VT
        FPTYPE	*spectraSigmaVT;	// Sigma * VT obtained from truncatated Spectra solve

        Eigen::MatrixX BBT;
        Eigen::MatrixX U_Spectra;	// first few eig vectors of this*this^T

        MKL_INT num_singular_vals;	// #Singular values


        using DenseMatrix<FPTYPE>::A;
        using DenseMatrix<FPTYPE>::_alloc;
        using DenseMatrix<FPTYPE>::num_docs;
        using DenseMatrix<FPTYPE>::vocab_size;
        using DenseMatrix<FPTYPE>::data;
        using DenseMatrix<FPTYPE>::copy_col_to;

    public:
        FloatingPointDenseMatrix(vocabSz_t d, docsSz_t s) :
            DenseMatrix<FPTYPE>(d, s),
            svd_temp(NULL),
            U(NULL),
            VT(NULL),
            Sigma(NULL),
            spectraSigmaVT(NULL)
        {
            num_singular_vals = num_docs() > vocab_size() ? vocab_size() : num_docs();
        }

        ~FloatingPointDenseMatrix()
        { }


        void populate_from_sparse(const FloatingPointSparseMatrix<FPTYPE>& from)
        {
            assert(vocab_size() == from.vocab_size());
            assert(num_docs() == from.num_docs());

#ifdef MKL_USE_DNSCSR
            assert(sizeof(MKL_INT) == sizeof(vocabSz_t));
            MKL_INT* offsets_MKL = new MKL_INT[num_docs() + 1];

            pfor_static_131072(int doc = 0; doc <= num_docs(); ++doc)
                offsets_MKL[doc] = (MKL_INT)from.offsets_CSC[doc];
            const MKL_INT cols = vocab_size(), rows = num_docs(), job[6] = { 1,0,0,2,0,0 };
            MKL_INT info;
            // Since we are using dnscsr for CSC, we have to think of matrices in transpose
            dnscsr(job, &rows, &cols,
                data(), &cols,
                (FPTYPE*)from.vals_CSC, (MKL_INT*)from.rows_CSC, offsets_MKL,
                &info);
            assert(info == 0);
            delete[] offsets_MKL;
#else
            pfor_dynamic_1024(int doc = 0; doc < num_docs(); ++doc)
                for (offset_t pos = from.offset_CSC(doc); pos < from.offset_CSC(doc + 1); ++pos)
                    elem_ref(from.row_CSC(pos), doc) = (FPTYPE)from.val_CSC(pos);
#endif
        }


        FPTYPE frobenius() const
        {
            return dot((size_t)num_docs() * (size_t)vocab_size(), data(), 1, data(), 1);
        }

        void initialize_for_full_svd()
        {
            // TODO: memory alignment
            svd_temp = new FPTYPE[(size_t)vocab_size()*(size_t)num_docs()];
            memcpy(svd_temp, data(), (size_t)vocab_size() * (size_t)num_docs() * sizeof(FPTYPE));
            U = new FPTYPE[(size_t)vocab_size()*(size_t)num_singular_vals];  // Column major, #rows=vocab_size
            VT = new FPTYPE[(size_t)num_docs()*(size_t)num_singular_vals];
            Sigma = new FPTYPE[num_singular_vals];
            assert(U != NULL); assert(VT != NULL); assert(Sigma != NULL);
        }

        FPTYPE singular_val(int k) const
        {
            assert(Sigma != NULL);
            assert(k < num_singular_vals);
            return Sigma[k];
        }

        void cleanup_full_svd()
        {
            assert(svd_temp != NULL);
            assert(U != NULL);
            assert(VT != NULL);
            assert(Sigma != NULL);
            delete[] svd_temp;
            delete[] U;
            delete[] VT;
            delete[] Sigma;
            svd_temp = U = VT = Sigma = NULL;
        }

        void initialize_for_Spectra(const docsSz_t num_topics)
        {
            spectraSigmaVT = new FPTYPE[(size_t)num_topics*(size_t)num_docs()];

            // Construct BBT = this * (this)^T, defaults to col-major
            BBT.resize(vocab_size(), vocab_size());
            gemm(CblasColMajor, CblasNoTrans, CblasTrans,
                (MKL_INT)vocab_size(), (MKL_INT)vocab_size(), (MKL_INT)num_docs(),
                1.0, data(), (MKL_INT)vocab_size(), data(), (MKL_INT)vocab_size(),
                0.0, BBT.data(), vocab_size());
        }

        void compute_truncated_Spectra(const docsSz_t num_topics)
        {
            // Call truncated Symm Eigensolve on BBT to get squared singular vals and U_trunc
            /*Spectra::DenseSymMatProd<FPTYPE> op(BBT);
            Spectra::SymEigsSolver<FPTYPE, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<FPTYPE> >
                eigs(&op, num_topics, 2 * num_topics + 1);
            MKL_DenseGenMatProd<FPTYPE> op(BBT);
            Spectra::SymEigsSolver<FPTYPE, Spectra::LARGEST_ALGE, MKL_DenseGenMatProd<FPTYPE> >
                eigs(&op, num_topics, 2 * num_topics + 1);*/

            MKL_DenseSymMatProd<FPTYPE> op(BBT);
            Spectra::SymEigsSolver<FPTYPE, Spectra::LARGEST_ALGE, MKL_DenseSymMatProd<FPTYPE> >
                eigs(&op, (MKL_INT)num_topics, 2 * (MKL_INT)num_topics + 1);
            eigs.init();
            int nconv = eigs.compute();
            assert(nconv >= (int)num_topics); // Number of converged eig vals >= #topics
            assert(eigs.info() == Spectra::SUCCESSFUL);

            // Set this->spectraSigmaVT by U^T*this
            auto evalues = eigs.eigenvalues();
            assert(evalues(num_topics - 1) > 0.0);

            U_Spectra = eigs.eigenvectors(num_topics);
            assert(U_Spectra.IsRowMajor == false);
            assert(U_Spectra.rows() == vocab_size() && U_Spectra.cols() == num_topics);

            std::cout << "\nFrob(U_Spectra) in dense: "
                << dot((size_t)vocab_size() * (size_t)num_topics, U_Spectra.data(), 1, U_Spectra.data(), 1) << "\n\n";

            std::cout << "Eigvals:  ";
            for (docsSz_t t = 0; t < num_topics; ++t)
                std::cout << "(" << t << "): " << std::sqrt(evalues(t)) << "\t";
            std::cout << std::endl;

            gemm(CblasColMajor, CblasTrans, CblasNoTrans,
                num_topics, num_docs(), vocab_size(),
                (FPTYPE)1.0, U_Spectra.data(), vocab_size(), this->data(), vocab_size(),
                (FPTYPE)0.0, spectraSigmaVT, num_topics);
        }

        void copy_spectraSigmaVT_from(
            FloatingPointDenseMatrix<FPTYPE>& from,
            const docsSz_t k,
            bool hardCopy = false) // true for memcpy, false for alias
        {
            assert(from.spectraSigmaVT != NULL);
            assert(from.num_docs() == num_docs() && vocab_size() == k);
 
            if (hardCopy) {
                assert(_alloc && data() != NULL);
                memcpy(data(), from.spectraSigmaVT, sizeof(FPTYPE) * (size_t)k * (size_t)num_docs());
            }
            else {
                if (_alloc) {
                    assert(data() != NULL);
                    delete[] A;
                    _alloc = false;
                }
                A = from.spectraSigmaVT;
            }
        }

        void copy_spectraSigmaVT_from(
            FloatingPointSparseMatrix<FPTYPE>& from,
            const docsSz_t k,
            bool hardCopy = false)
        {
            assert(from.spectraSigmaVT != NULL);
            assert(from.num_docs() == num_docs() && vocab_size() == k);

            if (hardCopy) {
                assert(_alloc && data() != NULL);
                memcpy(data(), from.spectraSigmaVT, sizeof(FPTYPE) * (size_t)k * (size_t)num_docs());
            }
            else {
                if (_alloc) {
                    assert(data() != NULL);
                    delete[] A;
                    _alloc = false;
                }
                A = from.spectraSigmaVT;
            }
        }

        void left_multiply_by_U_Spectra(
            FPTYPE *const out,
            const FPTYPE *in,
            const docsSz_t ld_in,
            const docsSz_t ncols)
        {
            assert(!U_Spectra.IsRowMajor);
            assert(U_Spectra.rows() == vocab_size());
            assert(ld_in >= (docsSz_t)U_Spectra.cols());
            gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                (MKL_INT)vocab_size(), (MKL_INT)ncols, (MKL_INT)U_Spectra.cols(),
                (FPTYPE)1.0, U_Spectra.data(), (MKL_INT)vocab_size(), in, (MKL_INT)ld_in,
                (FPTYPE)0.0, out, (MKL_INT)U_Spectra.rows());
        }

        FPTYPE* get_ptr_to_spectraSigmaVT()
        {
            return spectraSigmaVT;
        }

        void compare_LAPACK_Spectra(
            const docsSz_t num_topics,
            const double U_TOLERANCE,
            const double VT_TOLERANCE)
        {
            std::cout << "Comparing LAPACK and Spectra solves with tolerance for"
                << " (U: " << U_TOLERANCE << ")  (VT: " << VT_TOLERANCE << ")" << std::endl;
            assert(U != NULL);
            assert(U_Spectra.data() != NULL);
            assert(Sigma != NULL);	assert(spectraSigmaVT != NULL);
            assert(U_Spectra.IsRowMajor == false);
            for (docsSz_t topic = 0; topic < num_topics; ++topic)
                for (vocabSz_t word = 0; word < vocab_size(); ++word)
                    if (!(std::abs(U_Spectra.data()[word + (size_t)topic * (size_t)vocab_size()]
                        - U[(size_t)word + (size_t)topic * (size_t)vocab_size()]) < U_TOLERANCE
                        || std::abs(U_Spectra.data()[(size_t)word + (size_t)topic * (size_t)vocab_size()]
                            + U[(size_t)word + (size_t)topic * (size_t)vocab_size()]) < U_TOLERANCE))
                        std::cout << "Topic: " << topic << " Word: " << word
                        << std::setw(15) << std::right
                        << "U_sp: " << U_Spectra.data()[(size_t)word + (size_t)topic * (size_t)vocab_size()]
                        << std::setw(15) << std::right
                        << "\tU_MKL: " << U[(size_t)word + (size_t)topic * (size_t)vocab_size()]
                        << std::setw(10) << std::right
                        << "\tdiff: " << U_Spectra.data()[(size_t)word + (size_t)topic*(size_t)vocab_size()]
                        - U[(size_t)word + (size_t)topic * (size_t)vocab_size()]
                        << "\n";

            assert(VT != NULL);
            assert(Sigma != NULL);
            assert(spectraSigmaVT != NULL);
            for (docsSz_t doc = 0; doc < num_docs(); ++doc)
                for (docsSz_t topic = 0; topic < num_topics; ++topic)
                    if (!(std::abs(spectraSigmaVT[doc*num_topics + topic]
                        - VT[(size_t)doc * (size_t)num_singular_vals + (size_t)topic] * Sigma[topic])
                        < VT_TOLERANCE
                        || std::abs(spectraSigmaVT[(size_t)doc*(size_t)num_topics + (size_t)topic]
                            + VT[(size_t)doc*(size_t)num_singular_vals + (size_t)topic] * Sigma[topic])
                        < VT_TOLERANCE))
                        std::cout << "Topic: " << topic << " Doc: " << doc
                        << std::setw(15) << std::right
                        << "Spectra> SigVT: " << spectraSigmaVT[(size_t)doc*(size_t)num_topics + (size_t)topic]
                        << std::setw(10) << std::right
                        << "\tMKL> Sig*VT: " << VT[(size_t)doc*(size_t)num_singular_vals + (size_t)topic] * Sigma[topic]
                        << std::setw(10) << std::right
                        << "\tVT: " << VT[(size_t)doc*(size_t)num_singular_vals + (size_t)topic]
                        << std::setw(10) << std::right
                        << "\tSigma: " << Sigma[topic] << "\n";
        }

        void cleanup_Spectra()
        {
            assert(spectraSigmaVT != NULL);
            delete[] spectraSigmaVT;
            spectraSigmaVT = NULL;
        }


        void populate_with_topk_singulars(
            const docsSz_t k,
            FloatingPointDenseMatrix<FPTYPE> &from)
        {
            assert(from.U != NULL && from.VT != NULL && from.Sigma != NULL);
            assert(this->num_docs() == from.num_docs() && this->vocab_size() == from.vocab_size());
            FPTYPE *SigmaVT = new FPTYPE[(size_t)num_docs()*(size_t)k];
            //  Sigma[1:k] * VT[ k X _ ]
            for (docsSz_t d = 0; d < num_docs(); ++d)
                for (int s = 0; s < k; ++s)
                    SigmaVT[(size_t)d * (size_t)k + (size_t)s]
                    = from.Sigma[s] * from.VT[(size_t)d * (size_t)num_singular_vals + (size_t)s];

            //  U[_ X k] *  (Sigma[1:k]) * VT[k X _])
            gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                vocab_size(), num_docs(), k, 1.0,
                from.U, vocab_size(), SigmaVT, k, 0.0, this->data(), vocab_size());
            delete[] SigmaVT;
        }

        FPTYPE distsq_point_to_center(
            const docsSz_t d,
            const FPTYPE *const center)
        {
            const FPTYPE *const pt = data() + (size_t)d * (size_t)vocab_size();
            return dot(vocab_size(), pt, 1, pt, 1)
                + dot(vocab_size(), center, 1, center, 1)
                - 2 * dot(vocab_size(), pt, 1, center, 1);
        }

        void distsq_alldocs_to_centers(
            const vocabSz_t dim,
            docsSz_t num_centers, const FPTYPE *const centers, const FPTYPE *const centers_l2sq,
            docsSz_t num_docs, const FPTYPE *const docs, const FPTYPE *const docs_l2sq,
            FPTYPE *dist_matrix,
            FPTYPE *ones_vec = NULL) // Scratchspace of num_docs size and init to 1.0
        {
            bool ones_vec_alloc = false;
            if (ones_vec == NULL) {
                ones_vec = new FPTYPE[num_docs > num_centers ? num_docs : num_centers];
                std::fill_n(ones_vec, num_docs > num_centers ? num_docs : num_centers, (FPTYPE)1.0);
                ones_vec_alloc = true;
            }
            gemm(CblasColMajor, CblasTrans, CblasNoTrans,
                num_centers, num_docs, dim,
                (FPTYPE)-2.0, centers, dim, docs, dim,
                (FPTYPE)0.0, dist_matrix, num_centers);
            gemm(CblasColMajor, CblasNoTrans, CblasTrans,
                num_centers, num_docs, 1,
                (FPTYPE)1.0, centers_l2sq, num_centers, ones_vec, num_docs,
                (FPTYPE)1.0, dist_matrix, num_centers);
            gemm(CblasColMajor, CblasNoTrans, CblasTrans,
                num_centers, num_docs, 1,
                (FPTYPE)1.0, ones_vec, num_centers, docs_l2sq, num_docs,
                (FPTYPE)1.0, dist_matrix, num_centers);
            if (ones_vec_alloc) delete[] ones_vec;
        }


        void distsq_to_closest_center(
            const vocabSz_t dim,
            docsSz_t num_centers,
            const FPTYPE *const centers,
            const FPTYPE *const centers_l2sq,
            docsSz_t num_docs,
            const FPTYPE * const docs,
            const FPTYPE *const docs_l2sq,
            FPTYPE *const min_dist,
            FPTYPE *ones_vec = NULL) // Scratchspace of num_docs size and init to 1.0
        {
            FPTYPE *dist_matrix = new FPTYPE[(size_t)num_centers * (size_t)num_docs];
            distsq_alldocs_to_centers(
                dim, num_centers, centers, centers_l2sq,
                num_docs, docs, docs_l2sq,
                dist_matrix, ones_vec);
            pfor_static_131072(int64_t d = 0; d < num_docs; ++d) {
                FPTYPE min = FPTYPE_MAX;
                for (docsSz_t c = 0; c < num_centers; ++c)
                    if (dist_matrix[(size_t)c + (size_t)d * (size_t)num_centers] < min)
                        min = dist_matrix[(size_t)c + (size_t)d * (size_t)num_centers];
                min_dist[d] = min > (FPTYPE)0.0 ? min : (FPTYPE)0.0;
                // TODO: Ugly round about for small distance errors;
            }
            delete[] dist_matrix;
        }


        void update_min_distsq_to_centers(
            const vocabSz_t dim,
            const docsSz_t num_centers,
            const FPTYPE *const centers,
            docsSz_t num_docs,
            const FPTYPE * const docs,
            const FPTYPE *const docs_l2sq,
            FPTYPE *const min_dist,
            FPTYPE *dist = NULL, // preallocated scratch of space num_docs
            FPTYPE *ones_vec = NULL, // Scratchspace of num_docs size and init to 1.
            const bool weighted = false,
            const std::vector<size_t>& weights = std::vector<size_t>())
        {
            assert(!weighted); // Support not added yet

            bool dist_alloc = false;
            if (dist == NULL) {
                dist = new FPTYPE[(size_t)num_docs*(size_t)num_centers];
                dist_alloc = true;
            }
            assert(dist != NULL);
            FPTYPE* center_l2sq = new FPTYPE[num_centers];
            for (auto c = 0; c < num_centers; ++c)
                center_l2sq[c] = dot(dim,
                    centers + (size_t)c * (size_t)dim, 1,
                    centers + (size_t)c * (size_t)dim, 1);
            distsq_alldocs_to_centers(dim,
                num_centers, centers, center_l2sq,
                num_docs, docs, docs_l2sq,
                dist, ones_vec);

            pfor_static_131072(int d = 0; d < num_docs; ++d) {
                dist[d] = dist[d] > (FPTYPE)0.0 ? dist[d] : (FPTYPE)0.0;

                if (num_centers == 1) {
                    min_dist[d] = min_dist[d] > dist[d] ? dist[d] : min_dist[d];
                    // TODO: Fix Ugly round about for small distance errors;
                }
                else {
                    FPTYPE min = FPTYPE_MAX;
                    for (docsSz_t c = 0; c < num_centers; ++c)
                        if (dist[(size_t)c + (size_t)d * (size_t)num_centers] < min)
                            min = dist[(size_t)c + (size_t)d * (size_t)num_centers];
                    min_dist[d] = min > (FPTYPE)0.0 ? min : (FPTYPE)0.0;
                }
            }
            delete[] center_l2sq;
            if (dist_alloc) delete[] dist;
        }

        FPTYPE kmeanspp(
            const docsSz_t k,
            std::vector<docsSz_t>&centers,
            const bool weighted = false,
            const std::vector<size_t>& weights = std::vector<size_t>())
        {
            if (weighted) assert(weights.size() == num_docs());

            FPTYPE *const centers_l2sq = new FPTYPE[k];
            FPTYPE *const docs_l2sq = new FPTYPE[num_docs()];
            FPTYPE *const min_dist = new FPTYPE[num_docs()];
            FPTYPE *const centers_coords = new FPTYPE[(size_t)k * (size_t)vocab_size()];
            std::vector<FPTYPE> dist_cumul(num_docs() + 1);

            scal((size_t)k * (size_t)vocab_size(), 0.0, centers_coords, 1);
            std::fill_n(min_dist, num_docs(), FPTYPE_MAX);
            centers.push_back((docsSz_t)((size_t)rand() * (size_t)84619573 % (size_t)num_docs()));
            centers_l2sq[0] = dot(vocab_size(),
                data() + (size_t)centers[0] * (size_t)vocab_size(), 1,
                data() + (size_t)centers[0] * (size_t)vocab_size(), 1);
            //blascopy (vocab_size(), data() + (size_t)centers[0] * (size_t)vocab_size(), 1, centers_coords, 1);
            memcpy(centers_coords, data() + (size_t)centers[0] * (size_t)vocab_size(), sizeof(FPTYPE)*vocab_size());

            pfor_static_131072(int d = 0; d < num_docs(); ++d)
                docs_l2sq[d] = dot(vocab_size(),
                    data() + (size_t)d * (size_t)vocab_size(), 1,
                    data() + (size_t)d * (size_t)vocab_size(), 1);

            FPTYPE *dist_scratch_space = new FPTYPE[num_docs()];
            FPTYPE *ones_vec = new FPTYPE[num_docs()];
            std::fill_n(ones_vec, num_docs(), (FPTYPE)1.0);
            while (centers.size() < k) {
                update_min_distsq_to_centers(
                    vocab_size(), 1, centers_coords + (size_t)(centers.size() - 1) * (size_t)vocab_size(),
                    num_docs(), data(), docs_l2sq, min_dist, dist_scratch_space, ones_vec,
                    weighted, weights);
                dist_cumul[0] = 0;
                for (docsSz_t doc = 0; doc < num_docs(); ++doc)
                    dist_cumul[doc + 1] = dist_cumul[doc] + min_dist[doc];
                for (auto iter = centers.begin(); iter != centers.end(); ++iter) {
                    // Disance from center to its closest center == 0
                    assert(abs(dist_cumul[(*iter) + 1] - dist_cumul[*iter]) < 1e-4);
                    // Center is not replicated
                    assert(std::find(centers.begin(), centers.end(), *iter) == iter);
                    assert(std::find(iter + 1, centers.end(), *iter) == centers.end());
                }

                auto dice_throw = dist_cumul[num_docs()] * rand_fraction();
                assert(dice_throw < dist_cumul[num_docs()]);
                docsSz_t new_center = (docsSz_t)(std::upper_bound(dist_cumul.begin(), dist_cumul.end(), dice_throw)
                    - 1 - dist_cumul.begin());
                assert(new_center < num_docs());
                centers_l2sq[centers.size()] = dot(vocab_size(),
                    data() + (size_t)new_center * (size_t)vocab_size(), 1,
                    data() + (size_t)new_center * (size_t)vocab_size(), 1);
                //blascopy(vocab_size(), data() + (size_t)new_center * (size_t)vocab_size(), 1,
                //			centers_coords + centers.size() * (size_t)vocab_size(), 1);
                memcpy(centers_coords + centers.size() * (size_t)vocab_size(),
                    data() + (size_t)new_center * (size_t)vocab_size(), sizeof(FPTYPE) * vocab_size());
                centers.push_back(new_center);

            }
            delete[] ones_vec;
            delete[] dist_scratch_space;
            delete[] centers_l2sq;
            delete[] docs_l2sq;
            delete[] min_dist;
            delete[] centers_coords;
            return dist_cumul[num_docs() - 1];
        }

        FPTYPE kmeansbb(
            const docsSz_t k,
            FPTYPE* final_centers_coords)
        {
            FPTYPE  KMEANSBB_L_FACTOR = 0.5;
            MKL_UINT KMEANSBB_L = (count_t)(KMEANSBB_L_FACTOR * (FPTYPE)k);
            MKL_UINT KMEANSBB_R = 10 + 5 * std::log(k);

            MKL_UINT max_centers = KMEANSBB_L * KMEANSBB_R + 1;

            std::vector<docsSz_t> centers;

            FPTYPE *const centers_l2sq = new FPTYPE[max_centers];
            FPTYPE *const docs_l2sq = new FPTYPE[num_docs()];
            FPTYPE *const min_dist = new FPTYPE[num_docs()];
            FPTYPE *const centers_coords = new FPTYPE[(size_t)max_centers*(size_t)vocab_size()];

            scal((size_t)max_centers * (size_t)vocab_size(), 0.0, centers_coords, 1);
            std::fill_n(min_dist, num_docs(), FPTYPE_MAX);
            centers.push_back((docsSz_t)((size_t)rand() * (size_t)84619573 % (size_t)num_docs()));
            centers_l2sq[0] = dot(vocab_size(),
                data() + (size_t)centers[0] * (size_t)vocab_size(), 1,
                data() + (size_t)centers[0] * (size_t)vocab_size(), 1);
            memcpy(centers_coords, data() + (size_t)centers[0] * (size_t)vocab_size(), sizeof(FPTYPE) * vocab_size());

            pfor_static_131072(int d = 0; d < num_docs(); ++d)
                docs_l2sq[d] = dot(vocab_size(),
                    data() + (size_t)d * (size_t)vocab_size(), 1,
                    data() + (size_t)d * (size_t)vocab_size(), 1);

            FPTYPE *ones_vec = new FPTYPE[num_docs()];
            std::fill_n(ones_vec, num_docs(), (FPTYPE)1.0);

            update_min_distsq_to_centers(vocab_size(), 1, centers_coords,
                num_docs(), data(), docs_l2sq, min_dist, NULL, ones_vec);
            docsSz_t old_num_centers = 1;
            for (count_t round = 0; round < KMEANSBB_R && centers.size() < max_centers; ++round) {
                auto total_min_dist = asum(num_docs(), min_dist, 1);
                for (docsSz_t doc = 0; doc < num_docs() && centers.size() < max_centers; ++doc) {
                    if (rand_fraction() < KMEANSBB_L * min_dist[doc] / total_min_dist) {
                        centers_l2sq[centers.size()] = dot(vocab_size(),
                            data() + (size_t)doc * (size_t)vocab_size(), 1,
                            data() + (size_t)doc * (size_t)vocab_size(), 1);
                        memcpy(centers_coords + centers.size() * (size_t)vocab_size(),
                            data() + (size_t)doc * (size_t)vocab_size(), sizeof(FPTYPE) * vocab_size());
                        centers.push_back(doc);
                    }
                }

                std::cout << "New centers added: " << centers.size() - old_num_centers << std::endl;
                update_min_distsq_to_centers(vocab_size(),
                    centers.size() - old_num_centers, centers_coords + (size_t)old_num_centers*(size_t)vocab_size(),
                    num_docs(), data(), docs_l2sq, min_dist,
                    NULL, ones_vec);
                old_num_centers = centers.size();
            }
            delete[] centers_coords;

            std::sort(centers.begin(), centers.end());
            centers.erase(std::unique(centers.begin(), centers.end()), centers.end());
            std::cout << "K-means||, #initial centers: " << centers.size() << std::endl;
            FloatingPointDenseMatrix<FPTYPE> CentersMtx(vocab_size(), centers.size());
            for (size_t c = 0; c < centers.size(); ++c)
                memcpy(CentersMtx.data() + (size_t)c * (size_t)vocab_size(),
                    data() + (size_t)c * (size_t)vocab_size(), sizeof(FPTYPE) * vocab_size());

            for (docsSz_t c = 0; c < centers.size(); ++c)
                centers_l2sq[c] = dot(vocab_size(),
                    CentersMtx.data() + (size_t)c * (size_t)vocab_size(), 1,
                    CentersMtx.data() + (size_t)c * (size_t)vocab_size(), 1);
            docsSz_t doc_batch_size = 8192;
            FPTYPE *dist_matrix = new FPTYPE[centers.size() * (size_t)doc_batch_size];
            docsSz_t *const closest_center = new docsSz_t[num_docs()];
            for (docsSz_t batch = 0; batch*doc_batch_size < num_docs(); ++batch) {
                auto this_batch_size
                    = (batch + 1)*doc_batch_size < num_docs() ? doc_batch_size
                    : num_docs() - batch * doc_batch_size;
                distsq_alldocs_to_centers(vocab_size(), centers.size(), CentersMtx.data(), centers_l2sq,
                    this_batch_size, data() + batch*doc_batch_size*vocab_size(),
                    docs_l2sq + batch*doc_batch_size, dist_matrix);

                for (docsSz_t d = 0; d < this_batch_size; ++d)
                    closest_center[d + batch * doc_batch_size]
                    = (docsSz_t)imin(centers.size(), dist_matrix + (size_t)d*centers.size(), 1);
            }

            std::vector<size_t> center_weights(centers.size(), 0);
            for (docsSz_t doc = 0; doc < num_docs(); ++doc)
                center_weights[closest_center[doc]]++;
            assert(std::accumulate(center_weights.begin(), center_weights.end(), 0) == num_docs());

            auto residual = CentersMtx.run_lloyds(k, final_centers_coords,
                NULL, MAX_KMEANS_LOWD_REPS, true, center_weights);
            std::cout << "Residual in k-means-||: " << residual << std::endl;

            delete[] dist_matrix;
            delete[] ones_vec;
            delete[] centers_l2sq;
            delete[] docs_l2sq;
            delete[] min_dist;
            return residual;
        }

        FPTYPE kmeansmcmc(
            const docsSz_t k,
            std::vector<docsSz_t>&centers)
        {
            auto sample_size = KMEANSMCMC_SAMPLE_SIZE;

            FPTYPE *const centers_l2sq = new FPTYPE[k];
            FPTYPE *const docs_l2sq = new FPTYPE[num_docs()];
            FPTYPE *const init_prob = new FPTYPE[num_docs()];
            FPTYPE *const centers_coords = new FPTYPE[(size_t)k*(size_t)vocab_size()];
            FPTYPE *const ones_vec = new FPTYPE[num_docs()];

            docsSz_t *const sampled_docs = new docsSz_t[sample_size];
            FPTYPE	*const sampled_docs_coords = new FPTYPE[(size_t)sample_size*(size_t)vocab_size()];
            FPTYPE	*const sampled_docs_l2sq = new FPTYPE[sample_size];
            FPTYPE	*const sampled_docs_distsq_to_centers = new FPTYPE[sample_size];

            std::vector<FPTYPE> dist_cumul(num_docs() + 1);

            std::fill_n(ones_vec, num_docs(), (FPTYPE)1.0);
            std::fill_n(init_prob, num_docs(), FPTYPE_MAX);
            scal((size_t)k * (size_t)vocab_size(), 0.0, centers_coords, 1);

            centers.push_back((docsSz_t)((size_t)rand() * (size_t)84619573 % (size_t)num_docs()));
            centers_l2sq[0] = dot(vocab_size(),
                data() + (size_t)centers[0] * (size_t)vocab_size(), 1,
                data() + (size_t)centers[0] * (size_t)vocab_size(), 1);
            memcpy(centers_coords, data() + (size_t)centers[0] * (size_t)vocab_size(), sizeof(FPTYPE) * vocab_size());

            pfor_static_131072(int d = 0; d < num_docs(); ++d)
                docs_l2sq[d] = dot(vocab_size(),
                    data() + (size_t)d * (size_t)vocab_size(), 1,
                    data() + (size_t)d * (size_t)vocab_size(), 1);

            size_t num_centers_processed = 0;
            auto refresh_rate = 1;
            while (centers.size() < k)
            {
                axpy(num_docs(), -1 / (2 * num_docs()), ones_vec, 1, init_prob, 1);
                update_min_distsq_to_centers(vocab_size(),
                    centers.size() - num_centers_processed,
                    centers_coords + (size_t)num_centers_processed * (size_t)vocab_size(),
                    num_docs(), data(), docs_l2sq,
                    init_prob);
                /*distsq_to_closest_center(vocab_size(),
                  centers.size(), centers_coords, centers_l2sq,
                  num_docs(), data(), docs_l2sq,
                  init_prob);*/
                axpy(num_docs(), 1 / (2 * num_docs()), ones_vec, 1, init_prob, 1);
                dist_cumul[0] = 0;
                for (docsSz_t doc = 0; doc < num_docs(); ++doc)
                    dist_cumul[doc + 1] = dist_cumul[doc] + init_prob[doc];
                num_centers_processed = centers.size();

                ++refresh_rate;
                for (int i = 0; i < refresh_rate && centers.size() < k; ++i)
                {
                    pfor_dynamic_512(int s = 0; s < sample_size; ++s) {
                        auto dice_throw = dist_cumul[num_docs()] * rand_fraction();
                        assert(dice_throw < dist_cumul[num_docs()]);
                        auto doc = (docsSz_t)(std::upper_bound(dist_cumul.begin(), dist_cumul.end(), dice_throw)
                            - 1 - dist_cumul.begin());
                        assert(doc < num_docs());
                        sampled_docs[s] = doc;
                        memcpy(sampled_docs_coords + (size_t)s * (size_t)vocab_size(),
                            data() + (size_t)doc * (size_t)vocab_size(), sizeof(FPTYPE) * vocab_size());
                        sampled_docs_l2sq[s] = docs_l2sq[doc];
                    }

                    distsq_to_closest_center(vocab_size(),
                        centers.size(), centers_coords, centers_l2sq,
                        sample_size, sampled_docs_coords, sampled_docs_l2sq,
                        sampled_docs_distsq_to_centers);

                    auto new_center = 0;
                    for (size_t s = 1; s < sample_size; ++s)
                        if ((sampled_docs_distsq_to_centers[s] * init_prob[sampled_docs[new_center]]) /
                            (sampled_docs_distsq_to_centers[new_center] * init_prob[sampled_docs[s]]) > rand_fraction())
                            new_center = s;

                    centers_l2sq[centers.size()] = sampled_docs_l2sq[new_center];
                    memcpy(centers_coords + centers.size() * (size_t)vocab_size(),
                        data() + (size_t)sampled_docs[new_center] * (size_t)vocab_size(), sizeof(FPTYPE) * vocab_size());
                    centers.push_back(sampled_docs[new_center]);
                }
            }
            delete[] ones_vec;
            delete[] centers_l2sq;
            delete[] docs_l2sq;
            delete[] init_prob;
            delete[] centers_coords;

            delete[] sampled_docs;
            delete[] sampled_docs_coords;
            delete[] sampled_docs_distsq_to_centers;
            delete[] sampled_docs_l2sq;
            return dist_cumul[num_docs() - 1];
        }

        FPTYPE kmeans_init(
            const int num_centers,
            const int max_reps,
            const int method,
            std::vector<docsSz_t>&	best_seed,   // Wont be initialized if method==KMEANSBB
            FPTYPE *const			best_centers_coords = NULL) // Wont be initialized if null
        {
            FPTYPE min_total_dist_to_centers = FPTYPE_MAX;
            int best_rep;

            if (method == KMEANSBB) {
                assert(best_centers_coords != NULL);
                auto centers_coords = new FPTYPE[(size_t)num_centers * (size_t)vocab_size()];
                for (int rep = 0; rep < max_reps; ++rep) {
                    FPTYPE dist = kmeansbb(num_centers, centers_coords);
                    if (dist < min_total_dist_to_centers) {
                        min_total_dist_to_centers = dist;
                        memcpy(best_centers_coords, centers_coords,
                            sizeof(FPTYPE) * (size_t)num_centers*(size_t)vocab_size());
                        best_rep = rep;
                    }
                }
                delete[] centers_coords;
            }
            else if (method == KMEANSPP || KMEANS_INIT_METHOD == KMEANSMCMC) {
                auto kmeans_seeds = new std::vector<docsSz_t>[max_reps];
                for (int rep = 0; rep < max_reps; ++rep) {
                    FPTYPE dist;
                    if (method == KMEANSPP)
                        dist = kmeanspp(num_centers, kmeans_seeds[rep]);
                    else if (method == KMEANSMCMC)
                        dist = kmeansmcmc(num_centers, kmeans_seeds[rep]);
                    else;
                    std::cout << "k-means init residual: " << dist << std::endl;
                    if (dist < min_total_dist_to_centers) {
                        min_total_dist_to_centers = dist;
                        best_rep = rep;
                    }
                }
                best_seed = kmeans_seeds[best_rep];
                for (docsSz_t d = 0; d < num_centers; ++d)
                    copy_col_to(best_centers_coords + (size_t)d * (size_t)num_centers, best_seed[d]);
                delete[] kmeans_seeds;
            }
            else assert(false);
            return min_total_dist_to_centers;
        }

#define EIGEN_SOURCE_MKL 1
#define EIGEN_SOURCE_SPECTRA 2
        // Input @k: number of centers, @centers: reference to vector of indices of chosen seeds
        // Input @spectrum_source: MKL or Spectra
        // Output: Sum of distances of all points to chosen seeds
        // All calulations are done on the column space of Sigma*VT of the truncated SVD
        FPTYPE kmeanspp_on_col_space(
            docsSz_t k,
            std::vector<docsSz_t>& centers,
            int spectrum_source)
        {
            FPTYPE *SigmaVT = NULL;
            if (spectrum_source == EIGEN_SOURCE_MKL) {
                assert(U != NULL && VT != NULL && Sigma != NULL);
                SigmaVT = new FPTYPE[(size_t)num_docs() * (size_t)k];
                std::fill_n(SigmaVT, (size_t)num_docs() * (size_t)k, (FPTYPE)0.0);
                //  Sigma[1:k] * VT[ k X _ ]
                // TODO: translate to MKL
                for (docsSz_t doc = 0; doc < num_docs(); ++doc)
                    for (docsSz_t topic = 0; topic < k; ++topic)
                        SigmaVT[(size_t)doc * (size_t)k + topic] = Sigma[topic] * VT[(size_t)doc * (size_t)num_singular_vals + topic];
            }
            else if (spectrum_source == EIGEN_SOURCE_SPECTRA) {
                SigmaVT = spectraSigmaVT;
            }

            FPTYPE *const centers_l2sq = new FPTYPE[k];
            FPTYPE *const docs_l2sq = new FPTYPE[num_docs()];
            FPTYPE *const min_dist = new FPTYPE[num_docs()];
            FPTYPE *const centers_coords = new FPTYPE[(size_t)k * (size_t)k];
            std::vector<FPTYPE> dist_cumul(num_docs() + 1);
            dist_cumul[0] = 0;

            std::fill_n(centers_coords, (size_t)k * (size_t)k, (FPTYPE)0.0);
            std::fill_n(min_dist, num_docs(), FPTYPE_MAX);
            centers.push_back((docsSz_t)((size_t)rand() * (size_t)84619573 % (size_t)num_docs()));
            centers_l2sq[0] = dot((MKL_INT)k,
                SigmaVT + (size_t)centers[0] * (size_t)k, 1,
                SigmaVT + (size_t)centers[0] * (size_t)k, 1);
            memcpy(centers_coords, SigmaVT + (size_t)centers[0] * (size_t)k, sizeof(FPTYPE) * (size_t)k);
            for (docsSz_t d = 0; d < num_docs(); ++d)
                docs_l2sq[d] = dot((MKL_INT)k, SigmaVT + (size_t)d * (size_t)k, 1, SigmaVT + (size_t)d * (size_t)k, 1);

            while (centers.size() < k) {
                update_min_distsq_to_centers((MKL_INT)k,
                    1, centers_coords + (centers.size() - 1) * (size_t)k, num_docs(),
                    SigmaVT, docs_l2sq, min_dist);
                for (docsSz_t i = 0; i < num_docs(); ++i)
                    dist_cumul[i + 1] = dist_cumul[i] + min_dist[i];
                for (auto iter = centers.begin(); iter != centers.end(); ++iter) {
                    // Disance from center to its closest center == 0
                    assert(abs(*iter > 0 ? dist_cumul[*iter] - dist_cumul[(*iter) - 1] : dist_cumul[0]) < 1e-3);
                    // Center is not replicated
                    assert(std::find(centers.begin(), centers.end(), *iter) == iter);
                    assert(std::find(iter + 1, centers.end(), *iter) == centers.end());
                }
                auto dice_throw = dist_cumul[num_docs()] * rand_fraction();
                docsSz_t new_center = (docsSz_t)(std::upper_bound(dist_cumul.begin(), dist_cumul.end(), dice_throw)
                    - 1 - dist_cumul.begin());
                centers_l2sq[centers.size()] = dot((MKL_INT)k,
                    SigmaVT + (size_t)new_center * (size_t)k, 1,
                    SigmaVT + (size_t)new_center * (size_t)k, 1);
                memcpy(centers_coords + centers.size() * (size_t)k,
                    SigmaVT + (size_t)new_center * (size_t)k, sizeof(FPTYPE) * (size_t)k);
                centers.push_back(new_center);
            }
            delete[] centers_l2sq;
            delete[] docs_l2sq;
            delete[] min_dist;
            delete[] centers_coords;
            if (SigmaVT != NULL) delete[] SigmaVT;
            return dist_cumul[num_docs()];
        }

        void closest_centers(
            const docsSz_t num_centers,
            const FPTYPE *const centers,
            const FPTYPE *const docs_l2sq,
            docsSz_t *center_index,
            FPTYPE *const dist_matrix)  // Scratch init to num_centers*num_docs() size
        {
            FPTYPE *const centers_l2sq = new FPTYPE[num_centers];
            for (docsSz_t c = 0; c < num_centers; ++c)
                centers_l2sq[c] = dot(vocab_size(),
                    centers + (size_t)c * (size_t)vocab_size(), 1,
                    centers + (size_t)c * (size_t)vocab_size(), 1);
            distsq_alldocs_to_centers(vocab_size(), num_centers, centers, centers_l2sq,
                num_docs(), data(), docs_l2sq, dist_matrix);
            pfor_static_131072(int d = 0; d < num_docs(); ++d)
                center_index[d] = (docsSz_t)imin(num_centers, dist_matrix + (size_t)d * (size_t)num_centers, 1);
            delete[] centers_l2sq;
        }

        FPTYPE distsq(
            FPTYPE* p1_coords,
            FPTYPE* p2_coords,
            vocabSz_t dim)
        {
            return dot(dim, p1_coords, 1, p1_coords, 1)
                + dot(dim, p2_coords, 1, p2_coords, 1)
                - 2 * dot(dim, p1_coords, 1, p2_coords, 1);
        }



        void compute_docs_l2sq(FPTYPE *const docs_l2sq)
        {
            assert(docs_l2sq != NULL);
            pfor_static_131072(int d = 0; d < num_docs(); ++d)
                docs_l2sq[d] = dot(vocab_size(), data() + d*vocab_size(), 1,
                    data() + d*vocab_size(), 1);
        }


        FPTYPE lloyds_iter(
            const docsSz_t num_centers,
            FPTYPE *centers,
            const FPTYPE *const docs_l2sq,
            std::vector<docsSz_t> *closest_docs = NULL,
            bool weighted = false, // If true, supply weights
            const std::vector<size_t>& weights = std::vector<size_t>())
        {
            if (weighted) assert(weights.size() == num_docs());

            bool return_doc_partition = (closest_docs != NULL);

            Timer timer;

            FPTYPE *const dist_matrix = new FPTYPE[(size_t)num_centers * (size_t)num_docs()];
            docsSz_t *const closest_center = new docsSz_t[num_docs()];
            closest_centers(num_centers, centers, docs_l2sq, closest_center, dist_matrix);
            timer.next_time_secs("lloyds: closest center", 30);

            if (closest_docs == NULL)
                closest_docs = new std::vector<docsSz_t>[num_centers];
            else
                for (docsSz_t c = 0; c < num_centers; ++c)
                    closest_docs[c].clear();
            for (docsSz_t d = 0; d < num_docs(); ++d)
                closest_docs[closest_center[d]].push_back(d);
            scal((size_t)num_centers * (size_t)vocab_size(), 0.0, centers, 1);
            timer.next_time_secs("lloyds: assign pts to centers", 30);

            pfor_dynamic_16(int c = 0; c < num_centers; ++c)
                if (weighted)
                    for (auto iter = closest_docs[c].begin(); iter != closest_docs[c].end(); ++iter)
                        axpy(vocab_size(), (FPTYPE)(weights[*iter]) / closest_docs[c].size(),
                            data() + (*iter)*vocab_size(), 1, centers + (size_t)c * (size_t)vocab_size(), 1);
                else
                    for (auto iter = closest_docs[c].begin(); iter != closest_docs[c].end(); ++iter)
                        axpy(vocab_size(), (FPTYPE)(1.0) / closest_docs[c].size(),
                            data() + (*iter)*vocab_size(), 1, centers + (size_t)c * (size_t)vocab_size(), 1);
            timer.next_time_secs("lloyds: average pts ", 30);


            int BUF_PAD = 32;
            int CHUNK_SIZE = 8196;
            int nchunks = num_docs() / CHUNK_SIZE + (num_docs() % CHUNK_SIZE == 0 ? 0 : 1);
            std::vector<FPTYPE> residuals(nchunks*BUF_PAD, 0.0);

            pfor(int chunk = 0; chunk < nchunks; ++chunk)
                for (docsSz_t d = chunk*CHUNK_SIZE; d < num_docs() && d < (chunk + 1)*CHUNK_SIZE; ++d)
                    residuals[chunk*BUF_PAD] += (weighted ? weights[d] : (FPTYPE)1.0)
                    * distsq(data() + (size_t)d * (size_t)vocab_size(),
                        centers + (size_t)closest_center[d] * (size_t)vocab_size(), vocab_size());

            if (!return_doc_partition)
                delete[] closest_docs;
            delete[] closest_center;
            delete[] dist_matrix;

            FPTYPE residual = 0.0;
            for (int chunk = 0; chunk < nchunks; ++chunk)
                residual += residuals[chunk*BUF_PAD];

            timer.next_time_secs("lloyds: residual ", 30);

            return residual;
        }

        //
        // Return last residual
        //
        FPTYPE run_lloyds(
            const docsSz_t num_centers,
            FPTYPE *centers,
            std::vector<docsSz_t> *closest_docs, // Pass NULL if you dont want closest_docs returned
            const int max_reps,
            bool weighted = false, // If true, supply weights
            const std::vector<size_t>& weights = std::vector<size_t>())
        {
            FPTYPE residual;
            bool return_clusters = (closest_docs != NULL);

            if (return_clusters)
                for (int center = 0; center < num_centers; ++center)
                    assert(closest_docs[center].size() == 0);
            else
                closest_docs = new std::vector<docsSz_t>[num_centers];

            FPTYPE *docs_l2sq = new FPTYPE[num_docs()];
            compute_docs_l2sq(docs_l2sq);

            std::vector<size_t> prev_cl_sizes(num_centers, 0);
            auto prev_closest_docs = new std::vector<docsSz_t>[num_centers];

            Timer timer;
            for (int i = 0; i < max_reps; ++i) {
                residual = lloyds_iter(num_centers, centers, docs_l2sq, closest_docs, weighted, weights);
                std::cout << "Lloyd's iter " << i << ",  dist_sq residual: " << std::sqrt(residual) << "\n";
                timer.next_time_secs("run_lloyds: lloyds", 30);

                bool clusters_changed = false;
                for (int center = 0; center < num_centers; ++center) {
                    if (prev_cl_sizes[center] != closest_docs[center].size())
                        clusters_changed = true;
                    prev_cl_sizes[center] = closest_docs[center].size();
                }

                if (!clusters_changed)
                    for (int center = 0; center < num_centers; ++center) {
                        std::sort(closest_docs[center].begin(), closest_docs[center].end());
                        std::sort(prev_closest_docs[center].begin(), prev_closest_docs[center].end());

                        if (prev_closest_docs[center] != closest_docs[center])
                            clusters_changed = true;
                        prev_closest_docs[center] = closest_docs[center];
                    }

                if (!clusters_changed) {
                    std::cout << "Lloyds converged\n";
                    break;
                }
                timer.next_time_secs("run_lloyds: check conv", 30);
            }
            delete[] docs_l2sq;
            delete[] prev_closest_docs;
            if (!return_clusters)
                delete[] closest_docs;
            return residual;
        }
    };
}
