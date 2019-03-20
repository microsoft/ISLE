#include "types.h"
#include "parallel.h"

#include "BLAS-on-flash/include/utils.h"
#include "BLAS-on-flash/include/lib_funcs.h"
#include "BLAS-on-flash/include/flash_blas.h"
#include "BLAS-on-flash/include/scheduler/scheduler.h"

namespace TopicModel {
  using namespace flash;
  using namespace ISLE;
  void fill_doc_start_index(std::vector<size_t> &doc_start_index,
                            uint64_t num_topics,
                            uint64_t doc_begin,
                            uint64_t doc_blk_size,
                            offset_t *offsets_CSC,
                            flash_ptr<word_id_t> rows_CSC_fptr,
                            flash_ptr<FPTYPE> vals_CSC_fptr,
                            word_id_t *all_catchwords,
                            int* catchword_topic,
                            std::vector<std::tuple<doc_id_t, doc_id_t, FPTYPE> >*  doc_topic_sum,
                            uint64_t num_catchwords) {
    // create shifted copy of offsets array
    MKL_INT *shifted_offsets_CSC = new MKL_INT[doc_blk_size + 1];
    for (doc_id_t d = 0; d <= doc_blk_size; ++d) {
      shifted_offsets_CSC[d] = offsets_CSC[d] - offsets_CSC[0];
    }

    // alloc bufs
    uint64_t nnzs = offsets_CSC[doc_blk_size] - offsets_CSC[0];
    word_id_t *rows_CSC = new word_id_t[nnzs];
    FPTYPE *vals_CSC = new FPTYPE[nnzs];
    FPTYPE *DocTopicSumArray = new FPTYPE[num_topics * doc_blk_size];
    memset(DocTopicSumArray, 0, num_topics * doc_blk_size * sizeof(FPTYPE));

    // read from disk
    flash::read_sync(rows_CSC, rows_CSC_fptr, nnzs);
    flash::read_sync(vals_CSC, vals_CSC_fptr, nnzs);

#pragma omp parallel for schedule(dynamic, 1) num_threads(MAX_THREADS)
    for(uint64_t doc = 0; doc < doc_blk_size; ++doc)
    {
      offset_t c_pos = 0;
      for (offset_t pos = shifted_offsets_CSC[doc]; pos < shifted_offsets_CSC[doc + 1]; ++pos)
      {
        while (c_pos < num_catchwords && all_catchwords[c_pos] < rows_CSC[pos])
          ++c_pos;
        if (c_pos == num_catchwords)
          break;
        if (all_catchwords[c_pos] == rows_CSC[pos])
          DocTopicSumArray[catchword_topic[c_pos] + num_topics * doc] += vals_CSC[pos];
      }
    }

    for (doc_id_t doc = 0; doc < doc_blk_size; ++doc) {
      for (doc_id_t topic = 0; topic < num_topics; ++topic) {
        if (DocTopicSumArray[topic + num_topics * doc]){
          doc_topic_sum->push_back(std::make_tuple(doc + doc_begin, topic,
                                                   DocTopicSumArray[topic + num_topics * doc]));
        }
      }
      doc_start_index.push_back(doc_topic_sum->size());
    }

    // cleanup
    delete[] shifted_offsets_CSC;
    delete[] rows_CSC;
    delete[] vals_CSC;
    delete[] DocTopicSumArray;
  }

  void model_update(DenseMatrix<FPTYPE> &Model,
                    std::vector<std::tuple<doc_id_t, doc_id_t, FPTYPE>> *doc_topic_sum,
                    const std::vector<int> &doc_in_catchless_topic,
                    const std::vector<FPTYPE> &model_threshold,
                    uint64_t doc_begin,
                    uint64_t doc_blk_size,
                    uint64_t &iter_offset,
                    offset_t *offsets_CSC,
                    flash_ptr<word_id_t> rows_CSC_fptr,
                    flash_ptr<FPTYPE> vals_CSC_fptr,
                    uint64_t num_topics)
  {
    // create shifted copy of offsets array
    MKL_INT *shifted_offsets_CSC = new MKL_INT[doc_blk_size + 1];
    for (doc_id_t d = 0; d <= doc_blk_size; ++d) {
      shifted_offsets_CSC[d] = offsets_CSC[d] - offsets_CSC[0];
    }

    // alloc bufs
    uint64_t nnzs = offsets_CSC[doc_blk_size] - offsets_CSC[0];
    word_id_t *rows_CSC = new word_id_t[nnzs];
    FPTYPE *vals_CSC = new FPTYPE[nnzs];

    // read from disk
    flash::read_sync(rows_CSC, rows_CSC_fptr, nnzs);
    flash::read_sync(vals_CSC, vals_CSC_fptr, nnzs);

    auto blk_begin = doc_topic_sum->begin() + iter_offset;
    auto blk_end = doc_topic_sum->end();
    auto begin = blk_begin;
    for (doc_id_t doc = doc_begin; doc < doc_begin + doc_blk_size; ++doc)
    {
      auto end = std::upper_bound(begin, blk_end, doc,
                                  [](const auto &l, const auto &r) { return l < std::get<0>(r); });
      for (auto iter = begin; iter < end; ++iter)
      {
        assert(doc == std::get<0>(*iter));
        auto topic = std::get<1>(*iter);
        if (std::get<2>(*iter) > model_threshold[topic])
        {
          for (offset_t pos = shifted_offsets_CSC[doc - doc_begin];
                        pos < shifted_offsets_CSC[doc + 1 - doc_begin]; ++pos)
            Model.elem_ref(rows_CSC[pos], topic) += (FPTYPE)vals_CSC[pos];
        }
      }

      if (doc_in_catchless_topic[doc] != -1)
        for (offset_t pos = shifted_offsets_CSC[doc - doc_begin];
                      pos < shifted_offsets_CSC[doc - doc_begin + 1]; ++pos)
          Model.elem_ref(rows_CSC[pos], doc_in_catchless_topic[doc]) += (FPTYPE)vals_CSC[pos];
      begin = end;
    }

    // update `iter_offset`
    iter_offset = (begin - doc_topic_sum->begin());

    // cleanup
    delete[] shifted_offsets_CSC;
    delete[] rows_CSC;
    delete[] vals_CSC;
  }
} // namespace TopicModel
