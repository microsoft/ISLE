// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "Eigen/Core"
#include "Eigen/SparseCore"
#include "spectra-master/include/SymEigsSolver.h"

#include "dense_matrix.h"
#include "sparse_matrix.h"
#include "types.h"
#include "utils.h"

namespace ISLE {
  void load_model_from_file(DenseMatrix<FPTYPE>* model,
                            const std::string&   filename);

  void load_model_from_sparse_file(FPTYPE*            model_by_word,
                                   const doc_id_t     num_topics,
                                   const word_id_t    vocab_size,
                                   const std::string& model_file,
                                   const unsigned     base = 1);

  void load_sparse_model_from_file(SparseMatrix<FPTYPE>* sparse_model,
                                   const std::string&    filename,
                                   const offset_t        max_entries,
                                   const unsigned        base = 1);

  // Transpose a model matrix so its row-major, i.e., word major
  void create_model_by_word(FPTYPE*                          model_by_word,
                            const DenseMatrix<FPTYPE>* const model);

  class ISLEInfer {
    word_id_t vocab_size;
    doc_id_t  num_topics;
    doc_id_t  num_docs;

    const SparseMatrix<FPTYPE>* const infer_data;

    const FPTYPE* const model_by_word;  // the model matrix in word-major order

    Timer* timer;

// Temporary space for intermediate values
#define MAX_NNZS 20000
    FPTYPE* a;
    FPTYPE* M_slice;
    FPTYPE* z;
    FPTYPE* gradw;

   public:
    ISLEInfer(const FPTYPE* const               model_by_word_,
              const SparseMatrix<FPTYPE>* const infer_data_,
              const doc_id_t num_topics_, const word_id_t vocab_size_,
              const doc_id_t num_docs_);
    ~ISLEInfer();

    FPTYPE infer_doc_in_file(doc_id_t doc, FPTYPE* w, const int iters,
                             const FPTYPE Lfguess);

    bool mwu(
        const FPTYPE* const a, const FPTYPE* const M, FPTYPE* w, const int nnzs,
        const int iters,
        FPTYPE    Lf);  // Need to estimate Lipschitz cosntant for this instance

    void grad(FPTYPE* gradw, const FPTYPE* const a, const FPTYPE* const M,
              FPTYPE* w, const int nnzs);

    FPTYPE calculate_llh(const FPTYPE* const a, const FPTYPE* const M,
                         FPTYPE* w, const int nnzs);
  };
}
