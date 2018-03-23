// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "infer.h"

namespace ISLE {
  void read_model(DenseMatrix<FPTYPE>* model, char* buf, uint64_t fileSize) {
    word_id_t vocab_size = model->vocab_size();
    doc_id_t  num_topics = model->num_docs();

    doc_id_t  topic = 0;
    word_id_t word = 0;
    assert(sizeof(size_t) == 8);

    bool   was_whitespace = false;
    bool   before_decimal = true;
    FPTYPE val_before_decimal = 0.0;
    FPTYPE val_after_decimal = 0.0;
    int    pos_after_decimal = 0;

    for (size_t i = 0; i < fileSize; ++i) {
      assert(word < vocab_size);
      switch (buf[i]) {
        case '\r':
          break;
        case '\n':
          assert(word == vocab_size - 1);
        case ' ':
        case '\t':
          model->elem_ref(word, topic) =
              val_before_decimal +
              val_after_decimal * std::pow(0.1, pos_after_decimal);
          was_whitespace = true;
          break;
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
          // Allowing for multiple whitespaces
          if (was_whitespace == true) {
            if (word < vocab_size - 1)
              word++;
            else {
              word = 0;
              topic++;
            }
            was_whitespace = false;
            val_before_decimal = val_after_decimal = 0.0;
            pos_after_decimal = 0;
            assert(!before_decimal);
            before_decimal = true;
          }
          if (before_decimal) {
            val_before_decimal *= 10;
            val_before_decimal += buf[i] - '0';
          } else {
            val_after_decimal *= 10;
            val_after_decimal += buf[i] - '0';
            ++pos_after_decimal;
          }
          break;
        case '.':
          assert(before_decimal);
          before_decimal = false;
          break;
        default:
          std::cerr << "Bad format\n";
      }
    }
    assert(word == 0 || word == vocab_size - 1);
    if (word == vocab_size - 1) {  // Didnt reach "\n" on last line
      model->elem_ref(word, topic) =
          val_before_decimal +
          val_after_decimal * std::pow(0.1, pos_after_decimal);
      topic++;
    }
    assert(topic == num_topics);
  }

  void load_model_from_file(DenseMatrix<FPTYPE>* model,
                            const std::string&   filename) {
    assert(model != NULL);
    auto vocab_size = model->vocab_size();
    auto num_topics = model->num_docs();

#if FILE_IO_MODE == WIN_MMAP_FILE_IO
    {
      HANDLE   hFile, hMapFile;
      char*    pBuf;
      uint64_t fileSize =
          open_win_mmapped_file_handle(filename, hFile, hMapFile, &pBuf);
      read_model(model, pBuf, fileSize);
      close_win_mmapped_file_handle(hFile, hMapFile);
    }
#elif FILE_IO_MODE == LINUX_MMAP_FILE_IO
    {
      int      fd;
      char*    pBuf;
      uint64_t fileSize =
          open_linux_mmapped_file_handle(filename, fd, (void**) &pBuf);
      read_model(model, pBuf, fileSize);
      close_linux_mmapped_file_handle(fd, pBuf, fileSize);
    }
#elif FILE_IO_MODE == NAIVE_FILE_IO
    {
      std::ifstream in(filename);  // , ios_base::in);
      assert(in.is_open());
      for (doc_id_t topic = 0; topic < num_topics; ++topic) {
        for (word_id_t word = 0; word < vocab_size; ++word) {
          in >> model->elem_ref(word, topic);
        }

        auto topic_wt_sum =
            std::accumulate(model->data() + vocab_size * topic,
                            model->data() + vocab_size * (topic + 1), 0.0f);
        if (std::abs(1.0 - topic_wt_sum) > 0.01)
          std::cerr << "Topic " << topic << "  sum of weights: " << topic_wt_sum
                    << std::endl;

        for (word_id_t word = 0; word < vocab_size; ++word)
          model->elem_ref(word, topic) =
              model->elem(word, topic) / topic_wt_sum;
      }
      in.close();
    }
#endif
  }

  void read_sparse_model(FPTYPE* model_by_word, const doc_id_t num_topics,
                         const word_id_t vocab_size, char* buf,
                         uint64_t fileSize, const unsigned base) {
    doc_id_t  topic = 0;
    word_id_t word = 0;
    assert(sizeof(size_t) == 8);

    bool   was_whitespace = false;
    bool   before_decimal = true;
    FPTYPE val_before_decimal = 0.0;
    FPTYPE val_after_decimal = 0.0;
    int    pos_after_decimal = 0;

    int state = 1;

    for (size_t i = 0; i < fileSize; ++i) {
      assert(word < vocab_size);
      switch (buf[i]) {
        case '\r':
          break;
        case '\n':
          assert(state == 3);
          topic -= base;
          word -= base;
          model_by_word[num_topics * word + topic] =
              val_before_decimal +
              val_after_decimal * std::pow(0.1, pos_after_decimal);
          state = 1;
          word = 0;
          topic = 0;
          pos_after_decimal = 0;
          val_after_decimal = val_before_decimal = 0.0;
          assert(!before_decimal);
          before_decimal = true;
          break;
        case ' ':
        case '\t':
          was_whitespace = true;
          break;
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
          // Allowing for multiple whitespaces
          if (was_whitespace == true) {
            state++;
            was_whitespace = false;
            if (state == 3)
              assert(before_decimal);
          }
          switch (state) {
            case 1:
              topic *= 10;
              topic += buf[i] - '0';
              break;
            case 2:
              word *= 10;
              word += buf[i] - '0';
              break;
            case 3:
              if (before_decimal) {
                val_before_decimal *= 10;
                val_before_decimal += buf[i] - '0';
              } else {
                val_after_decimal *= 10;
                val_after_decimal += buf[i] - '0';
                ++pos_after_decimal;
              }
              break;
            default:
              assert(false);
          }
          break;
        case '.':
          assert(before_decimal);
          before_decimal = false;
          break;
        default:
          std::cerr << "Bad format\n";
      }
    }
    assert(state == 1 || state == 3);
    if (state == 3) {  // Didnt reach "\n" on last line
      topic -= base;
      word -= base;
      model_by_word[num_topics * word + topic] =
          val_before_decimal +
          val_after_decimal * std::pow(0.1, pos_after_decimal);
    }
  }

  // Model will be loaded in word-major order
  void load_model_from_sparse_file(FPTYPE*            model_by_word,
                                   const doc_id_t     num_topics,
                                   const word_id_t    vocab_size,
                                   const std::string& model_file,
                                   const unsigned     base) {
#if FILE_IO_MODE == WIN_MMAP_FILE_IO
    {
      HANDLE   hFile, hMapFile;
      char*    pBuf;
      uint64_t fileSize =
          open_win_mmapped_file_handle(model_file, hFile, hMapFile, &pBuf);
      read_sparse_model(model_by_word, num_topics, vocab_size, pBuf, fileSize,
                        base);
      close_win_mmapped_file_handle(hFile, hMapFile);
    }
#elif FILE_IO_MODE == LINUX_MMAP_FILE_IO
    {
      int      fd;
      char*    pBuf;
      uint64_t fileSize =
          open_linux_mmapped_file_handle(model_file, fd, (void**) &pBuf);
      read_sparse_model(model_by_word, num_topics, vocab_size, pBuf, fileSize,
                        base);
      close_linux_mmapped_file_handle(fd, pBuf, fileSize);
    }
#elif FILE_IO_MODE == NAIVE_FILE_IO
    {
      std::ifstream in(model_file);  // , ios_base::in);
      assert(in.is_open());
      doc_id_t  topic;
      word_id_t word;
      FPTYPE    weight;
      while (in.peek() != EOF) {
        in >> topic >> word >> weight;
        topic -= base;
        word -= base;
        model_by_word[num_topics * word + topic] = weight;
      }
      in.close();
    }
#endif
  }

  void load_sparse_model_from_file(SparseMatrix<FPTYPE>* sparse_model,
                                   const std::string&    filename,
                                   const offset_t        max_entries,
                                   const unsigned        base) {
    auto vocab_size = sparse_model->vocab_size();
    auto num_topics = sparse_model->num_docs();
    sparse_model->allocate(max_entries);
    sparse_model->set_offset_CSC(0, 0);
    sparse_model->set_offset_CSC(num_topics, max_entries);

#if FILE_IO_MODE == WIN_MMAP_FILE_IO
    {
      HANDLE   hFile, hMapFile;
      char*    pBuf;
      uint64_t fileSize =
          open_win_mmapped_file_handle(filename, hFile, hMapFile, &pBuf);
      assert(false);  // Not implemented yet
      // read_model(model, pBuf, fileSize, base);
      close_win_mmapped_file_handle(hFile, hMapFile);
    }
#elif FILE_IO_MODE == LINUX_MMAP_FILE_IO
    {
      int      fd;
      char*    pBuf;
      uint64_t fileSize =
          open_linux_mmapped_file_handle(filename, fd, (void**) &pBuf);
      assert(false);  // Not implemented yet
      // read_model(model, pBuf, fileSize, base);
      close_linux_mmapped_file_handle(fd, pBuf, fileSize);
    }
#elif FILE_IO_MODE == NAIVE_FILE_IO
    {
      std::ifstream in(filename);  // , ios_base::in);
      assert(in.is_open());
      offset_t  entries_read = 0;
      doc_id_t  previous_topic = 0, topic = 0;
      word_id_t word;
      FPTYPE    weight;
      while (in.peek() != EOF) {
        in >> topic;
        in >> word >> weight;
        topic -= base;
        word -= base;
        sparse_model->set_row_CSC(entries_read, word);
        sparse_model->set_val_CSC(entries_read, weight);

        if (topic > previous_topic) {
          sparse_model->set_offset_CSC(previous_topic + 1, entries_read);
          while (++previous_topic < topic)
            sparse_model->set_offset_CSC(topic, entries_read);
        }
        entries_read++;
        assert(entries_read <= max_entries);
      }
      assert(entries_read == max_entries);
      in.close();
    }
#endif
  }

  void create_model_by_word(FPTYPE*                          model_by_word,
                            const DenseMatrix<FPTYPE>* const model) {
    assert(model_by_word != NULL);
    auto vocab_size = model->vocab_size();
    auto num_topics = model->num_docs();

    FPomatcopy('C', 'T', vocab_size, num_topics, 1.0, model->data(), vocab_size,
               model_by_word, num_topics);

    /*    for (word_id_t word = 0; word < vocab_size; ++word)
    for (doc_id_t topic = 0; topic < num_topics; ++topic)
    model_by_word[word*num_topics + topic] = model->elem(word, topic);*/
  }

  ISLEInfer::ISLEInfer(const FPTYPE* const               model_by_word_,
                       const SparseMatrix<FPTYPE>* const infer_data_,
                       const doc_id_t num_topics_, const word_id_t vocab_size_,
                       const doc_id_t num_docs_)
      : model_by_word(model_by_word_), infer_data(infer_data_),
        num_topics(num_topics_), vocab_size(vocab_size_), num_docs(num_docs_) {
    timer = new Timer();

    a = new FPTYPE[MAX_NNZS];
    M_slice = new FPTYPE[MAX_NNZS * num_topics];
    gradw = new FPTYPE[num_topics];
    z = new FPTYPE[MAX_NNZS];

    assert(a != NULL);
    assert(M_slice != NULL);
    assert(gradw != NULL);
    assert(z != NULL);
  }

  ISLEInfer::~ISLEInfer() {
    delete timer;
    if (a)
      delete[] a;
    if (M_slice)
      delete[] M_slice;
    if (gradw)
      delete[] gradw;
    if (z)
      delete[] z;
  }

  // Return 0.0 if the calculation did not converge
  FPTYPE ISLEInfer::infer_doc_in_file(const doc_id_t doc, FPTYPE* w,
                                      const int iters, const FPTYPE Lfguess) {
    auto start_pos = infer_data->offset_CSC(doc);
    auto end_pos = infer_data->offset_CSC(doc + 1);
    auto nnzs_in_doc = 0;

    for (auto pos = start_pos; pos < end_pos; ++pos) {
      auto word_id = infer_data->row_CSC(pos);
      if (std::accumulate(model_by_word + word_id * num_topics,
                          model_by_word + (word_id + 1) * num_topics,
                          0.0) > 1.0e-10) {
        a[nnzs_in_doc] = infer_data->normalized_val_CSC(pos);
        memcpy(M_slice + nnzs_in_doc * num_topics,
               model_by_word + word_id * num_topics,
               sizeof(FPTYPE) * num_topics);
        nnzs_in_doc++;
      }
    }
    assert(nnzs_in_doc < MAX_NNZS);

    FPTYPE llh = 0.0;
    if (mwu(a, M_slice, w, nnzs_in_doc, iters, Lfguess) ==
        true)  // MWU converged
      llh = calculate_llh(a, M_slice, w, nnzs_in_doc);
    return llh;
  }

  // Returns llh = \sum_d a[d] * is_log_file_open{(M*w)[d]}
  bool ISLEInfer::mwu(const FPTYPE* const a, const FPTYPE* const M, FPTYPE* w,
                      const int nnzs, const int iters, FPTYPE Lf) {
    bool converged = false;

    for (auto topic = 0; topic < num_topics; ++topic)
      w[topic] = (FPTYPE) 1.0 / ((FPTYPE) num_topics);

    for (int guessLf = 0; guessLf < 10; guessLf++) {
      for (auto topic = 0; topic < num_topics; ++topic)
        w[topic] = (FPTYPE) 1.0 / ((FPTYPE) num_topics);

      for (auto iter = 0; iter < iters; ++iter) {
        // std::cout << calculate_llh(a, M, w, nnzs) << "\t";
        grad(gradw, a, M, w, nnzs);
        auto eta = std::sqrt(2.0 * std::log((FPTYPE) num_topics) /
                             (FPTYPE)(iter + 1)) /
                   Lf;

        for (auto topic = 0; topic < num_topics; ++topic)
          w[topic] *= std::exp(eta * gradw[topic]);

        auto normalizer = std::accumulate(w, w + num_topics, 0.0f);
        for (auto topic = 0; topic < num_topics; ++topic)
          w[topic] /= normalizer;
      }

      auto sumw = std::accumulate(w, w + num_topics, 0.0);
      if (std::isnormal(sumw))
        if (std::abs(1 - sumw) > 0.01)
          std::cout << "sum of W: " << sumw << std::endl;
        else {
          converged = true;
          break;
        }
      else
        Lf *= 2.0;
    }

    return converged;
  }

  void ISLEInfer::grad(FPTYPE* gradw, const FPTYPE* const a,
                       const FPTYPE* const M, FPTYPE* w, const int nnzs) {
    FPgemv(CblasRowMajor, CblasNoTrans, nnzs, num_topics, 1.0, M, num_topics, w,
           1, 0.0, z, 1);
    /* Equivalent to FPgemv above
    for (int r = 0; r < nnzs; ++r) {
        z[r] = 0.0;
        for (doc_id_t c = 0; c < num_topics; ++c) {
            z[r] += M[r*num_topics + c] * w[c];
        }
    } */

    for (int d = 0; d < num_topics; ++d)
      z[d] = a[d] / z[d];

    FPgemv(CblasRowMajor, CblasTrans, nnzs, num_topics, 1.0, M, num_topics, z,
           1, 0.0, gradw, 1);
  }

  FPTYPE ISLEInfer::calculate_llh(const FPTYPE* const a, const FPTYPE* const M,
                                  FPTYPE* w, const int nnzs) {
    auto z = new FPTYPE[nnzs];

    FPgemv(CblasRowMajor, CblasNoTrans, nnzs, num_topics, 1.0, M, num_topics, w,
           1, 0.0, z, 1);

    FPTYPE llh = 0.0;
    for (int d = 0; d < nnzs; ++d)
      llh += a[d] * std::log(z[d]);

    delete[] z;

    return llh;
  }
}
