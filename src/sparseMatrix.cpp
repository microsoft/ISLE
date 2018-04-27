#include <algorithm>

#include "sparseMatrix.h"
#include "restarted_block_ks.h"
#include "flash_prod_op.h"
#include "kmeans.h"

#include "blas-on-flash/include/utils.h"
#include "blas-on-flash/include/lib_funcs.h"
#include "blas-on-flash/include/flash_blas.h"
#include "blas-on-flash/include/scheduler/scheduler.h"
#include "flash_tasks/l2_thresh_norm.h"

namespace flash{
    extern Scheduler sched;
} // namespace flash

namespace ISLE
{
    // SparseMatrix

template <class T>
SparseMatrix<T>::SparseMatrix(
    const word_id_t d,
    const doc_id_t s,
    const offset_t nnzs)
    : _vocab_size(d),
      _num_docs(s),
      _nnzs(nnzs),
      vals_CSC(NULL),
      rows_CSC(NULL),
      offsets_CSC(NULL),
      normalized_vals_CSC(NULL),
      offsets_CSC_ptr(NULL),
      rows_CSC_ptr(NULL),
      vals_CSC_ptr(NULL),
      flash_malloc(false),
      avg_doc_sz((T)0.0)
{}

    template<class T>
    SparseMatrix<T>::~SparseMatrix()
    {
        if (vals_CSC) delete[] vals_CSC;
        if (rows_CSC) delete[] rows_CSC;
        if (offsets_CSC) delete[] offsets_CSC;
        if (normalized_vals_CSC) delete[] normalized_vals_CSC;

        if (offsets_CSC_ptr) delete[] offsets_CSC_ptr;
        if (rows_CSC_ptr) delete[] rows_CSC_ptr;
        if (vals_CSC_ptr) delete[] vals_CSC_ptr;
        if(flash_malloc){
            flash::flash_free<offset_t>(this->offsets_CSC_fptr);
            flash::flash_free<word_id_t>(this->rows_CSC_fptr);
            flash::flash_free<T>(this->vals_CSC_fptr);
        }
    }

    template <class T>
    void SparseMatrix<T>::map_flash(const std::string &offs_CSC_fname,
                                    const std::string &rows_CSC_fname,
                                    const std::string &vals_CSC_fname){
        GLOG_DEBUG("Mapping & reading offsets_CSC from ", offs_CSC_fname);
        this->offsets_CSC_fptr = flash::map_file<offset_t>(offs_CSC_fname, flash::Mode::READWRITE);
        this->offsets_CSC_ptr = new offset_t[num_docs() + 1];
        flash::read_sync(this->offsets_CSC_ptr, this->offsets_CSC_fptr, num_docs() + 1);
        _nnzs = this->offsets_CSC_ptr[num_docs()];
        GLOG_DEBUG("Mapping nnzs=", get_nnzs());
        GLOG_DEBUG("Mapping rows_CSC from ", rows_CSC_fname);
        this->rows_CSC_fptr = flash::map_file<word_id_t>(rows_CSC_fname, flash::Mode::READWRITE);
        GLOG_DEBUG("Mapping normalized_vals_CSC_fptr from ", vals_CSC_fname);
        this->normalized_vals_CSC_fptr = flash::map_file<T>(vals_CSC_fname, flash::Mode::READWRITE);
    }
    
    
    template <class T>
    void SparseMatrix<T>::read_flash() {
        if(this->offsets_CSC == nullptr){
            GLOG_DEBUG("Reading offsets_CSC");
            this->offsets_CSC = new offset_t[num_docs() + 1];
            flash::read_sync(this->offsets_CSC, this->offsets_CSC_fptr, num_docs() + 1);
        }

        _nnzs = this->offsets_CSC[num_docs()];
        GLOG_DEBUG("Reading nnzs=", _nnzs);
        GLOG_DEBUG("Reading rows_CSC");
        this->rows_CSC = new word_id_t[_nnzs];
        flash::read_sync(this->rows_CSC, this->rows_CSC_fptr, _nnzs);
        GLOG_DEBUG("Reading vals_CSC");
        this->vals_CSC = new T[_nnzs];
        flash::read_sync(this->vals_CSC, this->vals_CSC_fptr, _nnzs);
    }

    template <class T>
    void SparseMatrix<T>::unmap_flash(){
        GLOG_DEBUG("Unmapping offsets_CSC");
        flash::unmap_file<offset_t>(this->offsets_CSC_fptr);
        GLOG_DEBUG("Unmapping rows_CSC");
        flash::unmap_file<word_id_t>(this->rows_CSC_fptr);
        GLOG_DEBUG("Unmapping normalized_vals_CSC_fptr");
        flash::unmap_file<T>(this->normalized_vals_CSC_fptr);
    }

    template<class T>
    void SparseMatrix<T>::allocate(const offset_t nnzs_)
    {
        assert(_nnzs == 0);
        assert(vals_CSC == NULL && rows_CSC == NULL && offsets_CSC == NULL);

        _nnzs = nnzs_;
        vals_CSC = new T[get_nnzs()];
        rows_CSC = new word_id_t[get_nnzs()];
        offsets_CSC = new offset_t[num_docs() + 1];
    }

    template<class T>
    void SparseMatrix<T>::shrink(const offset_t new_nnzs_)
    {
        assert(_nnzs >= new_nnzs_);
        vals_CSC = (T*)realloc(vals_CSC, sizeof(T)*new_nnzs_);
        rows_CSC = (word_id_t*)realloc(rows_CSC, sizeof(word_id_t)*new_nnzs_);
        assert(offsets_CSC != NULL);

        // flash shrink
        flash::flash_truncate(vals_CSC_fptr, new_nnzs_ * sizeof(T));
        flash::flash_truncate(rows_CSC_fptr, new_nnzs_ * sizeof(word_id_t));
    }
    

    template<class T>
    void SparseMatrix<T>::allocate_flash(const offset_t nnzs_)
    {
        assert(_nnzs == 0);
        assert(flash_malloc == false);

        _nnzs = nnzs_;
        this->offsets_CSC = new offset_t[num_docs() + 1];
        this->offsets_CSC_fptr = flash::flash_malloc<offset_t>((num_docs() + 1) * sizeof(offset_t), "offs_CSC");
        this->rows_CSC_fptr = flash::flash_malloc<word_id_t>(_nnzs * sizeof(word_id_t), "rows_CSC");
        this->vals_CSC_fptr = flash::flash_malloc<T>(_nnzs * sizeof(T), "vals_CSC");

        // set flag to clear flash malloc
        flash_malloc = true;
    }
    

    template<class T>
    void SparseMatrix<T>::populate_CSC(const std::vector<DocWordEntry<count_t> >& entries)
    {
        assert(entries.size() < (1 << 31));
        allocate((offset_t)entries.size());
        doc_id_t doc = 0;
        word_id_t words_in_doc = 0;
        offsets_CSC[0] = 0;

        const auto num_entries = entries.size();
        for (size_t pos = 0; pos < num_entries; ++pos)
        {
            vals_CSC[pos] = (T)entries[pos].count;
            rows_CSC[pos] = entries[pos].word;
            if (entries[pos].doc == doc)
                words_in_doc++;
            else {
                offsets_CSC[doc + 1] = offsets_CSC[doc] + words_in_doc;
                while (++doc < entries[pos].doc)
                    offsets_CSC[doc + 1] = offsets_CSC[doc];
                words_in_doc = 1;
            }
        }

        offsets_CSC[doc + 1] = offsets_CSC[doc] + words_in_doc;
        while (++doc < num_docs())
            offsets_CSC[doc + 1] = offsets_CSC[doc];
        assert(offsets_CSC[num_docs()] == get_nnzs());

        uint64_t total_word_count = 0;
        for (auto iter = entries.begin(); iter != entries.end(); ++iter)
            total_word_count += iter->count;
        avg_doc_sz = (T)(total_word_count / num_docs());

        std::cout << "Entries in sparse matrix: " << get_nnzs() << std::endl;
        std::cout << "Average document size: " << avg_doc_sz << std::endl;
    }

    template<class T>
    void SparseMatrix<T>::populate_preprocessed_CSC(
        offset_t nnzs_,
        FPTYPE avg_doc_sz_,
        FPTYPE* normalized_vals_CSC_,
        word_id_t* rows_CSC_,
        offset_t *offsets_CSC_)
    {
        this->_nnzs = nnzs_;
        this->avg_doc_sz = avg_doc_sz_;
        this->normalized_vals_CSC = normalized_vals_CSC_;
        this->rows_CSC = rows_CSC_;
        this->offsets_CSC = offsets_CSC_;
        GLOG_DEBUG("Adding nnzs=", _nnzs);
    }

    template<class T>
    template<class FPTYPE>
    FPTYPE SparseMatrix<T>::doc_norm(
        doc_id_t doc,
        std::function<FPTYPE(const FPTYPE&, const FPTYPE&)> norm_fn)
    {
        FPTYPE norm = 0.0;
        for (offset_t pos = offsets_CSC[doc]; pos < offsets_CSC[doc + 1]; ++pos)
            norm = norm_fn(norm, vals_CSC[pos]);
        return norm;
    }

    template<class T>
    void SparseMatrix<T>::normalize_docs(
        bool delete_unnormalized,
        bool normalize_to_one)
    {
        normalized_vals_CSC = new T[get_nnzs()];
        uint64_t empty_docs = 0; 

        #ifndef NOPAR
        #pragma omp parallel for schedule(dynamic, 131072) reduction(+:empty_docs)
        #endif
        for(int64_t doc = 0; doc < num_docs(); ++doc) {
            auto doc_sum = std::accumulate(vals_CSC + offsets_CSC[doc], vals_CSC + offsets_CSC[doc + 1],
                (T)0.0, std::plus<T>());
            if (doc_sum == (T)0)
                empty_docs++; 
            for (offset_t pos = offsets_CSC[doc]; pos < offsets_CSC[doc + 1]; ++pos)
                if (std::is_same<T, count_t>::value) {
                    assert(normalize_to_one == false);
                    normalized_vals_CSC[pos]
                        = (count_t)std::ceil((FPTYPE)avg_doc_sz * vals_CSC[pos] / doc_sum);
                }
                else if (std::is_same<T, FPTYPE>::value) {
                    normalized_vals_CSC[pos] = normalize_to_one
                        ? (FPTYPE)vals_CSC[pos] / (FPTYPE)doc_sum
                        : (FPTYPE)avg_doc_sz * ((FPTYPE)vals_CSC[pos] / (FPTYPE)doc_sum);
                }
                else
                    assert(false);
        }
        if (empty_docs > 0)
            std::cout << "\n ==== WARNING:  " << empty_docs
            << " docs are empty\n" << std::endl;
        if (delete_unnormalized) {
            delete[] vals_CSC;
            vals_CSC = NULL;
        }
    }

    template<class T>
    size_t SparseMatrix<T>::count_distint_top_five_words(int min_distinct)
    {
        assert(min_distinct >= 2);
        std::vector<quintuple<T>> top_five;
        T buf[(1 << 15)];
        for (doc_id_t doc = 0; doc < num_docs(); ++doc)
            if (offsets_CSC[doc + 1] - offsets_CSC[doc] >= 5) {
                for (offset_t pos = offsets_CSC[doc]; pos < offsets_CSC[doc + 1]; ++pos)
                    buf[pos - offsets_CSC[doc]] = normalized_vals_CSC[pos];
                std::sort(buf, buf + offsets_CSC[doc + 1] - offsets_CSC[doc],
                    std::greater<>());
                top_five.push_back(
                    quintuple<T>(buf[0], buf[1], buf[2], buf[3], buf[4]));
            }
        std::sort(
            top_five.begin(), top_five.end(),
            [](const quintuple<T>& l, const quintuple<T>& r) {
            return
                l.first < r.first ||
                (l.first == r.first && l.second < r.second) ||
                (l.first == r.first && l.second == r.second && l.third < r.third) ||
                (l.first == r.first && l.second == r.second && l.third == r.third
                    && l.fourth < r.fourth) ||
                    (l.first == r.first && l.second == r.second && l.third == r.third
                        && l.fourth == r.fourth && l.fifth < r.fifth);
        }
        );
        std::cout << "top five vec size: " << top_five.size() << std::endl;
        MKL_UINT num_distinct = 0;
        auto iter = top_five.begin();
        auto prev_iter = top_five.begin();
        while (iter != top_five.end()) {
            if (*iter == *prev_iter) {
                iter++;
                continue;
            }
            if (iter - prev_iter >= min_distinct) {
                prev_iter = iter;
                ++num_distinct;
            }
            ++iter;
        }
        if (prev_iter - iter >= min_distinct) num_distinct++;

        return num_distinct;
    }

    template<class T>
    void SparseMatrix<T>::list_word_freqs_r(
        std::vector<T>* freqs,
        const doc_id_t d_b,
        const doc_id_t d_e,
        const word_id_t v_b,
        const word_id_t v_e)
    {
        if (d_e - d_b <= 16 && v_e - v_b <= 128) {
            for (doc_id_t doc = d_b; doc < d_e; ++doc) {
                if (offsets_CSC[doc] != offsets_CSC[doc + 1]) {
                    auto row_b = std::lower_bound(rows_CSC + offsets_CSC[doc], rows_CSC + offsets_CSC[doc + 1], v_b);
                    auto row_e = std::lower_bound(rows_CSC + offsets_CSC[doc], rows_CSC + offsets_CSC[doc + 1], v_e);
                    for (auto riter = row_b; riter < row_e; ++riter) {
                        freqs[*riter].push_back(normalized(*riter, doc));
                        assert(normalized(*riter, doc) > 0.0);
                        assert(*riter < v_e && *riter >= v_b);
                    }
                }
            }
        }
        else if (8 * (d_e - d_b) <= v_e - v_b) {
            list_word_freqs_r(freqs, d_b, d_e, v_b, (v_b + v_e) / 2);
            list_word_freqs_r(freqs, d_b, d_e, (v_b + v_e) / 2, v_e);
        }
        else if (8 * (d_e - d_b) > v_e - v_b) {
            list_word_freqs_r(freqs, d_b, (d_b + d_e) / 2, v_b, v_e);
            list_word_freqs_r(freqs, (d_b + d_e) / 2, d_e, v_b, v_e);
        }
        else {
            assert(false);
        }
    }

    template<class T>
    void SparseMatrix<T>::list_word_freqs(std::vector<T>* freqs)
    {
        Timer timer;
        assert(freqs != NULL);
        doc_id_t CHUNK_SIZE = 131072;
        long long num_chunks = num_docs() / CHUNK_SIZE;
        if (num_docs() % CHUNK_SIZE > 0)
            num_chunks++;
        auto freqs_comp = new std::vector<T>*[num_chunks];
        for (doc_id_t chunk = 0; chunk < num_chunks; ++chunk)
            freqs_comp[chunk] = new std::vector<T>[vocab_size()];
        timer.next_time_secs("list_freqs: alloc", 30);

        pfor(long long chunk = 0; chunk < num_chunks; chunk++) {
            auto chunk_b = chunk*CHUNK_SIZE;
            auto chunk_e = (chunk + 1) * CHUNK_SIZE > num_docs() ? num_docs() : (chunk + 1) * CHUNK_SIZE;
            list_word_freqs_r(freqs_comp[chunk], chunk_b, chunk_e, 0, vocab_size());
        }
        timer.next_time_secs("list_freqs: chunks", 30);

        //#pragma omp parallel for schedule(dynamic, 128)
        for (word_id_t word = 0; word < vocab_size(); ++word)
            for (doc_id_t chunk = 0; chunk < num_chunks; ++chunk)
                freqs[word].insert(freqs[word].end(), freqs_comp[chunk][word].begin(),
                    freqs_comp[chunk][word].end());
        timer.next_time_secs("list_freqs: append", 30);

        for (doc_id_t chunk = 0; chunk < num_chunks; ++chunk)
            delete[] freqs_comp[chunk];
        timer.next_time_secs("list_freqs: dealloc", 30);

        for (word_id_t word = 0; word < vocab_size(); ++word)
            std::sort(freqs[word].begin(), freqs[word].end(), std::greater<>());
        timer.next_time_secs("list_freqs: sort", 30);
    }

    template<class T>
    void SparseMatrix<T>::list_word_freqs_by_sorting(std::vector<A_TYPE>* freqs)
    {
        Timer timer;
        auto entries = new DocWordEntry<A_TYPE>[get_nnzs()];
        pfor_dynamic_131072 (int64_t doc = 0; doc < (int64_t)num_docs(); ++doc) {
            for (offset_t pos = offsets_CSC[doc]; pos < offsets_CSC[doc + 1]; ++pos) {
                entries[pos].count = normalized_vals_CSC[pos];
                entries[pos].word = rows_CSC[pos];
                entries[pos].doc = doc;
            }
        }
        timer.next_time_secs("list_word_freqs: copy to arr", 30);

        parallel_sort(entries, entries + get_nnzs(),
            [](const auto& l, const auto&r)
        {return l.word < r.word || (l.word == r.word && l.count > r.count); });
        timer.next_time_secs("list_word_freqs: sort", 30);

        size_t *word_offsets = new size_t[vocab_size() + 1];
        word_offsets[0] = 0; word_offsets[vocab_size()] = get_nnzs();
        word_id_t cur_word = 0;
        for (size_t pos = 0; pos < get_nnzs(); ++pos) {
            if (entries[pos].word > cur_word) {
                for (word_id_t w = cur_word + 1; w <= entries[pos].word; ++w)
                    word_offsets[w] = pos;
                cur_word = entries[pos].word;
            }
        }
        while (cur_word < vocab_size())
            word_offsets[++cur_word] = get_nnzs();
        timer.next_time_secs("list_word_freqs: prefix sum", 30);

        pfor_dynamic_256(int64_t word = 0; word < vocab_size(); ++word) {
            if (word_offsets[word + 1] > word_offsets[word]) {
                assert(entries[word_offsets[word]].word == word);
                assert(entries[word_offsets[word + 1] - 1].word == word);
            }
            for (size_t pos = word_offsets[word]; pos < word_offsets[word + 1]; ++pos)
                freqs[word].push_back(entries[pos].count);
        }
        timer.next_time_secs("list_word_freqs: copying output", 30); 

        delete[] word_offsets;
        delete[] entries;
    }


    template<class T>
    void SparseMatrix<T>::list_word_freqs_from_CSR(
        const word_id_t word_begin,
        const word_id_t word_end,
        const FPTYPE *const normalized_vals_CSR,
        const offset_t *const offsets_CSR,
        std::vector<A_TYPE>* freqs)
    {
        for(int64_t word = word_begin; word < word_end; ++word) {
            freqs[word].insert(freqs[word].begin(), 
                normalized_vals_CSR + offsets_CSR[word - word_begin] - offsets_CSR[0],
                normalized_vals_CSR + offsets_CSR[word - word_begin + 1] - offsets_CSR[0]);
            std::sort(freqs[word].begin(), freqs[word].end(), std::greater<>());
        }
    }


    // Input: @num_topics, @freqs: one vector for each word listing its non-zero freqs in docs
    // Output: @zetas: Reference to cutoffs frequencies for each word
    // Return: Total number of entries that are above threshold
    template<class T>
    offset_t SparseMatrix<T>::compute_thresholds(
        word_id_t word_begin,
        word_id_t word_end,
        std::vector<A_TYPE> *const freqs,
        std::vector<T>& zetas,
        const doc_id_t num_topics)
    {
        assert(zetas[word_begin] == 0);

        offset_t new_nnzs = 0;
        word_id_t freq_less_words = 0;

        // Check:  rounding down vs round to nearest
        const doc_id_t count_gr = (doc_id_t)(w0_c * (FPTYPE)num_docs() / (2.0 * (FPTYPE)num_topics));
        const doc_id_t count_eq = (doc_id_t)std::ceil(3.0 * eps1_c * w0_c * (FPTYPE)num_docs() / (FPTYPE)num_topics);

        for (word_id_t word = word_begin; word < word_end; ++word) {
            assert(std::is_sorted(freqs[word].begin(), freqs[word].end(), std::greater<>()));

            if (std::is_same<T, FPTYPE>::value) {
                for (auto iter = freqs[word].begin(); iter != freqs[word].end(); ++iter) {
                    assert(*iter <= avg_doc_sz);
                    *iter = std::round(*iter);
                }
                auto trunc = std::lower_bound(freqs[word].begin(), freqs[word].end(), 0.0, std::greater<>());
                if (trunc != freqs[word].end()) assert(*trunc == 0.0);
                if (trunc != freqs[word].begin()) assert(*(trunc - 1) > 0.0);
                freqs[word].resize(trunc - freqs[word].begin());
            }

            if (freqs[word].size() > 0) {
                if (std::is_same<T, FPTYPE>::value) {
                    assert(freqs[word].size() > 0);
                    assert(freqs[word].back() >= 1.0);
                }
                // Check if there are too few occurrences of the word across documents
                if (count_gr > freqs[word].size()) {
                    assert(*(freqs[word].end() - 1) > 0);
                    if (FEW_SAMPLES_THRESHOLD_DROP)
                        if (std::is_same<T, count_t>::value)
                            zetas[word] = (A_TYPE)(1 << 31);
                        else if (std::is_same<T, FPTYPE>::value)
                            zetas[word] = FP_MAX;
                        else assert(false);
                    else { // Throw everything in
                        assert(freqs[word].size() < (1 << 30));
                        new_nnzs += (offset_t)freqs[word].size();
                        if (std::is_same<T, count_t>::value)
                            zetas[word] = 1;
                        else if (std::is_same<T, FPTYPE>::value)
                            zetas[word] = (FPTYPE)1.0;
                        else assert(false);
                    }
                    continue;
                }

                // Find zeta such that "#(freqs > zeta) > count_gr" & "#(freqs == zeta) <=count_eq" 
                if (std::is_same<T, count_t>::value) {
                    auto zeta = freqs[word][count_gr - 1];
                    while (1) {
                        auto cur_pos = std::lower_bound(freqs[word].begin(), freqs[word].end(), zeta, std::greater<>());
                        auto next_pos = std::upper_bound(freqs[word].begin(), freqs[word].end(), zeta, std::greater<>());
                        assert((cur_pos != freqs[word].end()) && (cur_pos != next_pos));
                        assert(zeta > 0 && zeta == *cur_pos && zeta == *(next_pos - 1));
                        // Found exactly what we are looking for
                        if (next_pos - cur_pos < count_eq) {
                            new_nnzs += next_pos - freqs[word].begin(); // Use freqs==zeta as well??
                            zetas[word] = zeta;
                            break;
                        }
                        // Check if the frequency counts end at zeta, or if zeta can go no lower
                        if (next_pos == freqs[word].end() || zeta == 1) {
                            assert(*(next_pos - 1) != 0);
                            if (BAD_THRESHOLD_DROP)
                                zetas[word] = (A_TYPE)(1 << 31);
                            else {
                                assert(freqs[word].size() < 0x7fffffff);
                                new_nnzs += (offset_t)freqs[word].size();
                                zetas[word] = 1; // Throw everything in??
                            }
                            break;
                        }
                        zeta = *next_pos;
                    }
                }
                else if (std::is_same<T, FPTYPE>::value) {
                    auto zeta = freqs[word][count_gr - 1];
                    while (1) {
                        assert(freqs[word].back() > 0.0);
                        auto cur_pos = std::lower_bound(freqs[word].begin(), freqs[word].end(), zeta, std::greater<>());
                        auto next_pos = std::upper_bound(freqs[word].begin(), freqs[word].end(), zeta, std::greater<>());
                        assert((cur_pos != freqs[word].end()) && (cur_pos != next_pos));
                        assert(zeta > 0); assert(zeta == *cur_pos && zeta == *(next_pos - 1));
                        // Found exactly what we are looking for
                        if (next_pos - cur_pos < count_eq) {
                            new_nnzs += next_pos - freqs[word].begin(); // Use freqs==zeta as well??
                            zetas[word] = zeta;
                            break;
                        }
                        // Check if the frequency counts end at zeta, or if zeta can go no lower
                        if (next_pos == freqs[word].end() || zeta == 1) {
                            assert(*(next_pos - 1) != 0);
                            if (BAD_THRESHOLD_DROP)
                                zetas[word] = FP_MAX;
                            else {
                                assert(freqs[word].size() < 0x7fffffff);
                                new_nnzs += (offset_t)freqs[word].size();
                                zetas[word] = 1.0; // Throw all non-zero rounds everything in??
                            }
                            break;
                        }
                        zeta = *next_pos;
                    }
                }
                else
                    assert(false);
            }
            else {
                freq_less_words++;
                zetas[word] = 1.0;
                //assert(false);
            }
        }
        if (freq_less_words > 0)
            std::cout << "\n ==== WARNING: " << freq_less_words << " words do not occur in the corpus.\n\n";
        return new_nnzs;
    }

    // TODO: Optimize this
    // Input: @r,  @doc_partition: list of docs in this partition
    // Ouput: @thresholds: the @k-th highest count of each word in @doc_partition, length = vocab_size()
    template<class T>
    void SparseMatrix<T>::rth_highest_element(
        const MKL_UINT r,
        const std::vector<doc_id_t>& doc_partition,
        T *thresholds)
    {
        if (doc_partition.size() == 0) {
            for (word_id_t word = 0; word < vocab_size(); ++word)
                thresholds[word] = (T)0.0;
            return;
        }

        auto freqs = new std::vector<T>[vocab_size()];

        for (auto diter = doc_partition.begin(); diter != doc_partition.end(); ++diter)
            for (auto witer = offsets_CSC[*diter]; witer < offsets_CSC[(*diter) + 1]; ++witer)
                freqs[rows_CSC[witer]].push_back(normalized_vals_CSC[witer]);

        for (word_id_t word = 0; word < vocab_size(); ++word) {
            if (freqs[word].size() > r) {
                std::sort(freqs[word].begin(), freqs[word].end(), std::greater<>());
                thresholds[word] = freqs[word][r - 1];
            }
            else {
                thresholds[word] =
                    r >= doc_partition.size()
                    ? freqs[word].size() == doc_partition.size()
                    ? *std::min_element(freqs[word].begin(), freqs[word].end())
                    : (T)0.0
                    : (T)0.0;
            }
        }

        delete[] freqs;
    }

    template<class T>
    void SparseMatrix<T>::rth_highest_element_using_CSR(
        const word_id_t word_begin,
        const word_id_t word_end,
        const doc_id_t num_topics,
        const MKL_UINT r,
        const std::vector<doc_id_t>* closest_docs,
        const FPTYPE *const normalized_vals_CSR,
        const doc_id_t *cols_CSR,
        const offset_t *const offsets_CSR,
        const int *const cluster_ids,
        T *threshold_matrix_tr)
    {
        auto freqs = new std::vector<T>[num_topics];

        for (word_id_t word = word_begin; word < word_end; ++word) {
            for (auto pos = offsets_CSR[word - word_begin] - offsets_CSR[0];
                pos < offsets_CSR[word + 1 - word_begin] - offsets_CSR[0]; ++pos)
                if (cluster_ids[cols_CSR[pos]] != -1)
                    freqs[cluster_ids[cols_CSR[pos]]].push_back(normalized_vals_CSR[pos]);

            for (int topic = 0; topic < num_topics; ++topic)
            {
                auto doc_partition = closest_docs[topic];
                auto thresholds = threshold_matrix_tr + (size_t)word * (size_t)num_topics;

                if (freqs[topic].size() > r) {
                    std::sort(freqs[topic].begin(), freqs[topic].end(), std::greater<>());
                    thresholds[topic] = freqs[topic][r - 1];
                }
                else {
                    thresholds[topic] =
                        r >= doc_partition.size()
                        ? freqs[topic].size() == doc_partition.size()
                        ? *std::min_element(freqs[topic].begin(), freqs[topic].end())
                        : (T)0.0
                        : (T)0.0;
                }
                freqs[topic].clear();
            }
        }
        delete[] freqs;
    }

    // Input: @num_topics, @thresholds: Threshold for (words,topic)
    // Output: @catchwords: list of catchwords for each topic
    template<class T>
    void SparseMatrix<T>::find_catchwords(
        const doc_id_t num_topics,
        const T *const thresholds,
        std::vector<word_id_t> *catchwords)
    {
        for (doc_id_t topic = 0; topic < num_topics; ++topic) {
            for (word_id_t word = 0; word < vocab_size(); ++word) {
                bool is_catchword = false;
                for (doc_id_t other_topic = 0; other_topic < num_topics; ++other_topic) {
                    if (topic != other_topic) {
                        is_catchword =
                            ((T)thresholds[word + (size_t)topic* (size_t)vocab_size()]
                > rho_c * (T)thresholds[word + (size_t)other_topic * (size_t)vocab_size()]);
                        if (!is_catchword)
                            break;
                    }
                }
                if (is_catchword)
                    catchwords[topic].push_back(word);
            }
        }
    }

    template<class T>
    void SparseMatrix<T>::construct_topic_model(
        DenseMatrix<FPTYPE>& Model,
        const doc_id_t num_topics,
        const std::vector<doc_id_t> *const closest_docs,
        const std::vector<word_id_t> *const catchwords,
        bool avg_null_topics,
        std::vector<std::tuple<int, int, doc_id_t> >* top_topic_pairs,
        std::vector<std::pair<word_id_t, int> >* catchword_topics,
        std::vector<std::tuple<doc_id_t, doc_id_t, FPTYPE> >*  doc_topic_sum)
    {
        Timer timer;
        assert(Model.vocab_size() == vocab_size());
        assert(Model.num_docs() == num_topics);
        assert(normalized_vals_CSC != NULL);

        std::fill_n(Model.data(), (size_t)Model.vocab_size() * (size_t)Model.num_docs(), (FPTYPE)0.0);

        pfor_dynamic_1(int topic = 0; topic < num_topics; ++topic) {
            if (catchwords[topic].size() == 0) {
                if (avg_null_topics) {
                    for (auto d_iter = closest_docs[topic].begin();
                        d_iter != closest_docs[topic].end(); ++d_iter)
                        for (offset_t pos = offsets_CSC[*d_iter];
                            pos < offsets_CSC[(*d_iter) + 1]; ++pos)
                            Model.elem_ref(rows_CSC[pos], topic) += (FPTYPE)normalized_vals_CSC[pos];
                    for (word_id_t word = 0; word < vocab_size(); ++word)
                        Model.elem_ref(word, topic) /= closest_docs[topic].size();
                }
            }
        }
        timer.next_time_secs("c TM: catchless", 30);

        
        bool free_catchword_topics = false;
        if (catchword_topics == NULL) {
            catchword_topics = new std::vector<std::pair<word_id_t, int> >;
            free_catchword_topics = true;
        }
        
        for (auto topic = 0; topic < num_topics; ++topic)
            if (catchwords[topic].size() > 0)
                for (auto iter = catchwords[topic].begin(); iter < catchwords[topic].end(); ++iter)
                    catchword_topics->push_back(std::make_pair(*iter, topic));
        std::sort(catchword_topics->begin(), catchword_topics->end(),
            [](const auto& l, const auto& r) {return l.first < r.first; });
        
        word_id_t *all_catchwords = new word_id_t[catchword_topics->size()];
        int *catchword_topic = new int[catchword_topics->size()];
        int num_catchwords = catchword_topics->size();
        for (auto i = 0; i < num_catchwords; ++i) {
            all_catchwords[i] = (*catchword_topics)[i].first;
            catchword_topic[i] = (*catchword_topics)[i].second;
        }
        timer.next_time_secs("c TM: list", 30);


        // Can we use a sparse matrix here?
        size_t doc_block_size = DOC_BLOCK_SIZE;
        size_t doc_blocks = ((size_t)num_docs()) / doc_block_size;
        if (doc_blocks * doc_block_size != (size_t)num_docs()) doc_blocks++;

        bool free_doc_topic_sum = false;
        if (doc_topic_sum == NULL) {
            doc_topic_sum = new std::vector<std::tuple<doc_id_t, doc_id_t, FPTYPE> >;
            free_doc_topic_sum = true;
        }
        std::vector<size_t> doc_start_index;
        doc_start_index.push_back(0);

        size_t num_docs_alloc = (size_t)num_docs() < doc_block_size
            ? (size_t)num_docs() : doc_block_size;
        FPTYPE* DocTopicSumArray = new FPTYPE[(size_t)num_topics * num_docs_alloc];

        for (size_t block = 0; block < doc_blocks; ++block) {
            memset((void*)DocTopicSumArray, 0, (size_t)num_topics * num_docs_alloc * sizeof(FPTYPE));

            size_t doc_begin = block * doc_block_size;
            size_t doc_end = (block + 1) * doc_block_size;
            if ((size_t)num_docs() < doc_end) doc_end = (size_t)num_docs();

            pfor(long long doc = doc_begin; doc < doc_end; ++doc) {
                offset_t c_pos = 0;
                for (offset_t pos = offsets_CSC[doc]; pos < offsets_CSC[doc + 1]; ++pos) {
                    while (c_pos < num_catchwords && all_catchwords[c_pos] < rows_CSC[pos])
                        ++c_pos;
                    if (c_pos == num_catchwords)
                        break;

                    if (all_catchwords[c_pos] == rows_CSC[pos])
                        DocTopicSumArray[catchword_topic[c_pos] + num_topics * (doc - block*doc_block_size)]
                        += normalized_vals_CSC[pos];
                }
            }
            for (doc_id_t doc = block * doc_block_size;
                doc < num_docs() && doc < (block + 1) * doc_block_size; ++doc) {
                for (doc_id_t topic = 0; topic < num_topics; ++topic) {
                    if (DocTopicSumArray[topic + num_topics * (doc - block*doc_block_size)])
                        doc_topic_sum->push_back(std::make_tuple(doc, topic,
                            DocTopicSumArray[topic + num_topics * (doc - block*doc_block_size)]));
                }
                doc_start_index.push_back(doc_topic_sum->size());
            }
        }
        delete[] DocTopicSumArray;
        timer.next_time_secs("c TM: topic wts in doc", 30);


        for (doc_id_t doc = 0; doc < num_docs(); ++doc) {
            if (top_topic_pairs != NULL) {
                FPTYPE max = 0.0, max2 = 0.0; // second max 
                int max_topic = -1, max2_topic = -1;

                for (auto iter = doc_topic_sum->begin() + doc_start_index[doc];
                    iter < doc_topic_sum->begin() + doc_start_index[doc + 1]; ++iter) {

                    if (std::get<2>(*iter) > max) {
                        max2 = max; max2_topic = max_topic;
                        max = std::get<2>(*iter);
                        max_topic = std::get<1>(*iter);
                    }
                    else if (std::get<2>(*iter) > max2) {
                        max2 = std::get<2>(*iter);
                        max2_topic = std::get<1>(*iter);
                    }
                }
                if (max_topic >= 0 && max2_topic >= 0) {
                    assert(max > 0.0 && max2 > 0.0);
                    top_topic_pairs->push_back(std::make_tuple(max_topic, max2_topic, doc));
                }
            }
        }
        delete[] all_catchwords;
        delete[] catchword_topic;
        timer.next_time_secs("c TM: top2 topics/doc", 30);



        // This is expensive for large matrices, parallelize this
        /*FPTYPE *sorted_doc_topic_sums = new FPTYPE[(size_t)num_topics * (size_t)num_docs()];
        std::cout << "contruct topic model: sorted_doc_topic_sums alloc succeeded" << std::endl;
        size_t doc_block_size = 64;
        size_t num_doc_blocks = num_docs() % doc_block_size == 0
            ? num_docs() / doc_block_size : num_docs() / doc_block_size + 1;
        for (auto topic = 0; topic < num_topics; ++topic)
            for (size_t block = 0; block < num_doc_blocks; ++block)
                for (doc_id_t doc = block*doc_block_size;
                    doc < num_docs() && doc < (block + 1)*doc_block_size; ++doc)
                    sorted_doc_topic_sums[(size_t)doc + (size_t)topic * (size_t)num_docs()]
                    = DocTopicSums.elem(topic, doc);*/
        std::cout << "Size of doc_topic_sum array: " << doc_topic_sum->size() << std::endl;
        std::sort(doc_topic_sum->begin(), doc_topic_sum->end(),
            [](const auto& l, const auto& r)
        { return std::get<1>(l) < std::get<1>(r)
            || (std::get<1>(l) == std::get<1>(r) && std::get<2>(l) > std::get<2>(r)); });
        timer.next_time_secs("c TM: order by topic, value", 30);


        const size_t rank_threshold = (doc_id_t)(eps3_c*w0_c*(FPTYPE)num_docs() / ((FPTYPE)num_topics * 2.0));
        assert(rank_threshold > 0);
        //std::vector<T> model_thresholds(num_topics);

        doc_id_t topic_block_size = 10;
        doc_id_t num_topic_blocks = num_topics / topic_block_size;
        if (num_topic_blocks * topic_block_size < num_topics)
            ++num_topic_blocks;

        pfor_dynamic_1(long long topic_block = 0; topic_block < num_topic_blocks; ++topic_block) {
            auto topic_begin = topic_block * topic_block_size;
            auto topic_end = (topic_block + 1) * topic_block_size;
            if (topic_end > num_topics) topic_end = num_topics;

            auto iter = std::lower_bound(doc_topic_sum->begin(), doc_topic_sum->end(), topic_begin,
                [](const auto& l, const auto& r) {return std::get<1>(l) < r; });
            assert(iter != doc_topic_sum->end());

            for (auto topic = topic_begin; topic < topic_end; ++topic) {
                if (catchwords[topic].size() > 0) {
                    
                    auto end = std::upper_bound(iter, doc_topic_sum->end(), topic,
                        [](const auto& l, const auto& r)  {return l < std::get<1>(r); });

                    FPTYPE model_threshold = 0.0;
                    if (end - iter < rank_threshold) {
                        std::cout << "\n==== Warning: Topic " << topic << " threshold is 0.\n";
                        model_threshold = 0.0;
                    }
                    else {
                        model_threshold = std::get<2>(*(iter + (size_t)rank_threshold - 1));
                        assert(std::get<1>(*(iter + (size_t)rank_threshold - 1)) == topic);
                    }

                    int num_cols_used = 0;
                    while (iter < end && std::get<2>(*iter) > model_threshold) {
                        ++num_cols_used;
                        doc_id_t doc = std::get<0>(*iter);
                        for (offset_t pos = offsets_CSC[doc]; pos < offsets_CSC[doc + 1]; ++pos)
                            Model.elem_ref(rows_CSC[pos], topic) += (FPTYPE)normalized_vals_CSC[pos];
                        ++iter;
                    }

                    for (word_id_t word = 0; word < vocab_size(); ++word)
                        Model.elem_ref(word, topic) /= num_cols_used;

                    iter = end;
                }
                else {
                    //assert(std::get<1>(*iter) > topic);
                }
            }
        }
        timer.next_time_secs("c TM: threshold and add", 30);

        /*pfor_dynamic_1(int topic = 0; topic < num_topics; ++topic) {
            if (catchwords[topic].size() > 0) {
                int num_cols_used = 0;
                for (doc_id_t doc = 0; doc < num_docs(); ++doc) {
                    if (DocTopicSums.elem(topic, doc) >= model_thresholds[topic]) {
                        num_cols_used++;
                        for (offset_t pos = offsets_CSC[doc]; pos < offsets_CSC[doc + 1]; ++pos)
                            Model.elem_ref(rows_CSC[pos], topic) += (FPTYPE)normalized_vals_CSC[pos];
                    }
                }

                for (word_id_t word = 0; word < vocab_size(); ++word)
                    Model.elem_ref(word, topic) /= num_cols_used;
            }
        }*/

        pfor_dynamic_1(int t = 0; t < num_topics; ++t) {
            auto topic_vector_sum = FPasum(Model.vocab_size(), Model.data() + (size_t)Model.vocab_size() * (size_t)t, 1);
            FPscal(Model.vocab_size(), 1.0 / topic_vector_sum, Model.data() + (size_t)Model.vocab_size() * (size_t)t, 1);
        }
        timer.next_time_secs("c TM: scaling", 30);

        if (free_catchword_topics)
            delete catchword_topics;
        if (free_doc_topic_sum)
            delete doc_topic_sum;
    }

    template<class T>
    void SparseMatrix<T>::topic_coherence(
        const doc_id_t num_topics,
        const word_id_t& M,
        const DenseMatrix<FPTYPE>& model,
        std::vector<std::pair<word_id_t, FPTYPE> >* top_words,
        std::vector<FPTYPE>& coherences,
        const FPTYPE coherence_eps)
    {
        assert(model.vocab_size() == vocab_size());
        for (auto topic = 0; topic < num_topics; ++topic)
            model.find_n_top_words(topic, M, top_words[topic]);

        // joint_counts[i][j] will hold joint freq for jth and ith dom words for j<i
        std::vector<std::vector<std::vector<size_t> > > joint_doc_freqs;
        compute_joint_doc_frequency(num_topics, top_words, joint_doc_freqs);
        std::vector<std::vector<size_t> > doc_freqs;
        compute_doc_frequency(num_topics, top_words, doc_freqs);

        coherences.resize(num_topics, 0.0);
        for (auto topic = 0; topic < num_topics; ++topic) {
            if (top_words[topic].size() > 1)
                pfor_dynamic_8192(long long word = 0; word < M; ++word)
                for (word_id_t word2 = 0; word2 < word; ++word2) {
                    assert(doc_freqs[topic][word2] > 0);
                    coherences[topic]
                        += (FPTYPE)std::log(joint_doc_freqs[topic][word][word2] + coherence_eps)
                        - std::log((FPTYPE)doc_freqs[topic][word2]);
                }
        }
    }

    // Output: @joint_counts: joint_counts[i][j] contains joint freq for j<i
    template<class T>
    void SparseMatrix<T>::compute_joint_doc_frequency(
        const int num_topics,
        const std::vector<std::pair<word_id_t, FPTYPE> >* top_words,
        std::vector<std::vector<std::vector<size_t> > >& joint_counts)
        const
    {
        bool use_parallel_version = true;

        Timer timer;

        joint_counts.resize(num_topics);
        for (auto topic = 0; topic < num_topics; ++topic) {
            joint_counts[topic].resize(top_words[topic].size());

            for (word_id_t word = 0; word < top_words[topic].size(); ++word) {
                assert(joint_counts[topic][word].size() == 0);
                joint_counts[topic][word].resize(word, 0);
            }
        }

        if (!use_parallel_version)
        { // Serial version
            for (doc_id_t doc = 0; doc < num_docs(); ++doc)
                for (auto topic = 0; topic < num_topics; ++topic)
                    for (word_id_t w1 = 0; w1 < top_words[topic].size(); ++w1)
                        for (word_id_t w2 = 0; w2 < w1; ++w2)
                            if (normalized(top_words[topic][w1].first, doc) > 0
                                && normalized(top_words[topic][w2].first, doc) > 0)
                                ++joint_counts[topic][w1][w2];
        }
        else
        { // Parallel version
            const size_t chunk_size = (1 << 16);
            const size_t num_chunks = (num_docs() % chunk_size == 0)
                ? num_docs() / chunk_size
                : (num_docs() / chunk_size) + 1;
            std::vector<std::vector<std::vector<std::vector<size_t> > > >joint_counts_chunks;

            joint_counts_chunks.resize(num_chunks);
            for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
                joint_counts_chunks[chunk].resize(num_topics);
                for (auto topic = 0; topic < num_topics; ++topic) {
                    joint_counts_chunks[chunk][topic].resize(top_words[topic].size());

                    for (word_id_t word = 0; word < top_words[topic].size(); ++word) {
                        assert(joint_counts_chunks[chunk][topic][word].size() == 0);
                        joint_counts_chunks[chunk][topic][word].resize(word, 0);
                    }
                }
            }

            pfor_dynamic_1(int64_t chunk = 0; chunk < num_chunks; ++chunk)
                for (int64_t doc = chunk*chunk_size;
                    doc < num_docs() && doc < (chunk + 1)*chunk_size; ++doc)
                    for (auto topic = 0; topic < num_topics; ++topic)
                        for (word_id_t w1 = 0; w1 < top_words[topic].size(); ++w1)
                            for (word_id_t w2 = 0; w2 < w1; ++w2)
                                if (normalized(top_words[topic][w1].first, doc) > 0
                                    && normalized(top_words[topic][w2].first, doc) > 0)
                                    ++joint_counts_chunks[chunk][topic][w1][w2];

            for (size_t chunk = 0; chunk < num_chunks; ++chunk)
                for (auto topic = 0; topic < num_topics; ++topic)
                    for (word_id_t w1 = 0; w1 < top_words[topic].size(); ++w1)
                        for (word_id_t w2 = 0; w2 < w1; ++w2)
                            joint_counts[topic][w1][w2]
                            += joint_counts_chunks[chunk][topic][w1][w2];
        }

        for (auto topic = 0; topic < num_topics; ++topic)
            for (word_id_t word = 0; word < top_words[topic].size(); ++word)
                assert(joint_counts[topic][word].size() == word);

        /*for (auto pos = offsets_CSC[doc]; pos != offsets_CSC[doc + 1]; ++pos)
        for (auto pos2 = offsets_CSC[doc]; pos2 != offsets_CSC[doc + 1]; ++pos2)
        joint_counts[rows_CSC[pos]][rows_CSC[pos2]]++;*/
    }

    // Convert to algebra. Equivalent to "this*column(1)"
    template<class T>
    void SparseMatrix<T>::compute_doc_frequency(
        const int num_topics,
        const std::vector<std::pair<word_id_t, FPTYPE> >* top_words,
        std::vector<std::vector<size_t> >& doc_frequencies)
        const
    {
        bool use_parallel_version = true;

        Timer timer;
        doc_frequencies.resize(num_topics);
        for (auto topic = 0; topic < num_topics; ++topic) {
            assert(top_words[topic].size() > 0);
            doc_frequencies[topic].resize(top_words[topic].size(), 0);
        }

        if (!use_parallel_version)
        { // Use Serial version
            for (int64_t doc = 0; doc < num_docs(); ++doc)
                for (auto topic = 0; topic < num_topics; ++topic)
                    for (word_id_t word = 0; word < top_words[topic].size(); ++word)
                        if (normalized(top_words[topic][word].first, doc) > (T)0.0)
                            doc_frequencies[topic][word]++;
        }
        else
        { // Use parallel version
            const size_t chunk_size = (1 << 16);
            const size_t num_chunks = (num_docs() % chunk_size == 0)
                ? num_docs() / chunk_size
                : (num_docs() / chunk_size) + 1;
            std::vector<std::vector<std::vector<size_t> > > doc_frequencies_chunks;
            doc_frequencies_chunks.resize(num_chunks);

            for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
                doc_frequencies_chunks[chunk].resize(num_topics);
                for (auto topic = 0; topic < num_topics; ++topic) {
                    assert(top_words[topic].size() > 0);
                    doc_frequencies_chunks[chunk][topic].resize(top_words[topic].size(), 0);
                }
            }

            pfor_dynamic_1(int64_t chunk = 0; chunk < num_chunks; ++chunk)
                for (int64_t doc = chunk*chunk_size;
                    doc < num_docs() && doc < (chunk + 1)*chunk_size; ++doc)
                    for (auto topic = 0; topic < num_topics; ++topic)
                        for (word_id_t word = 0; word < top_words[topic].size(); ++word)
                            if (normalized(top_words[topic][word].first, doc) > (T)0.0)
                                doc_frequencies_chunks[chunk][topic][word]++;

            for (size_t chunk = 0; chunk < num_chunks; ++chunk)
                for (auto topic = 0; topic < num_topics; ++topic)
                    for (word_id_t word = 0; word < top_words[topic].size(); ++word)
                        doc_frequencies[topic][word] +=
                        doc_frequencies_chunks[chunk][topic][word];
        }

        for (auto topic = 0; topic < num_topics; ++topic) {
            assert(doc_frequencies[topic].size() == top_words[topic].size());
            for (auto iter = doc_frequencies[topic].begin();
                iter < doc_frequencies[topic].end(); ++iter)
                assert(*iter > 0);
        }
    }

    template<class T>
    void SparseMatrix<T>::compute_log_combinatorial(std::vector<FPTYPE>& docs_log_fact)
    {
        auto doc_word_count = new count_t[num_docs()];
        pfor_dynamic_131072(int64_t doc = 0; doc < num_docs(); ++doc) {
            doc_word_count[doc] = 0;
            for (offset_t pos = offset_CSC(doc); pos < offset_CSC(doc + 1); ++pos)
                doc_word_count[doc] += (int)val_CSC(pos);
        }
        int max_words_per_doc = 0;
        for (doc_id_t doc = 0; doc < num_docs(); ++doc) {
            if (max_words_per_doc < doc_word_count[doc])
                max_words_per_doc = doc_word_count[doc];
        }

        auto log_fact = new FPTYPE[max_words_per_doc + 1];
        log_fact[0] = 0;
        for (int i = 0; i < max_words_per_doc; ++i)
            log_fact[i + 1] = log_fact[i] + std::log(i + 1);

        assert(docs_log_fact.size() == 0);
        docs_log_fact.resize(num_docs());
        pfor_dynamic_131072(int64_t doc = 0; doc < num_docs(); ++doc) {
            FPTYPE doclogfact = 0;
            for (offset_t pos = offset_CSC(doc); pos < offset_CSC(doc + 1); ++pos)
                doclogfact -= log_fact[(int)val_CSC(pos)];
            doclogfact += log_fact[doc_word_count[doc]];
            docs_log_fact[doc] = doclogfact;
        }
        delete[] doc_word_count;
        delete[] log_fact;
    }

    // FPSparseMatrix

    template<class FPTYPE>
    FPSparseMatrix<FPTYPE>::FPSparseMatrix(const word_id_t d, const doc_id_t s)
        :
        SparseMatrix<FPTYPE>(d, s),
        U_colmajor(NULL),
        U_rowmajor(NULL),
        U_rows(0),
        U_cols(0),
        SigmaVT(NULL)
    {}

    template<class FPTYPE>
    FPSparseMatrix<FPTYPE>::~FPSparseMatrix()
    {
        if (SigmaVT) delete[] SigmaVT;
        if (U_colmajor) delete[] U_colmajor;
        if (U_rowmajor) delete[] U_rowmajor;
    }

    template<class FPTYPE>
    FPSparseMatrix<FPTYPE>::FPSparseMatrix(
        const SparseMatrix<FPTYPE>& from,
        const bool copy_normalized)
        : SparseMatrix<FPTYPE>(from.vocab_size(), from.num_docs()),
        SigmaVT(NULL)
    {
        allocate(from.get_nnzs());
        if (from.normalized_vals_CSC != NULL)
            normalized_vals_CSC = new FPTYPE[_nnzs];
        if (copy_normalized) {
            assert(from.normalized_vals_CSC != NULL);
            memcpy(vals_CSC, from.normalized_vals_CSC, (size_t)_nnzs * sizeof(FPTYPE));
            memcpy(normalized_vals_CSC, from.normalized_vals_CSC, (size_t)_nnzs * sizeof(FPTYPE));
        }
        else {
            memcpy(vals_CSC, from.vals_CSC, (size_t)_nnzs * sizeof(FPTYPE));
            memcpy(normalized_vals_CSC, from.normalized_vals_CSC, (size_t)_nnzs * sizeof(FPTYPE));
        }
        memcpy(rows_CSC, from.rows_CSC, (size_t)_nnzs * sizeof(word_id_t));
        memcpy(offsets_CSC, from.offsets_CSC, ((size_t)num_docs() + 1) * sizeof(offset_t));
    }

    template<class FPTYPE>
    FPTYPE FPSparseMatrix<FPTYPE>::frobenius() const
    {
        assert(offsets_CSC[0] == 0);
        return FPdot(get_nnzs(), vals_CSC, 1, vals_CSC, 1);
    }

    template<class FPTYPE>
    FPTYPE FPSparseMatrix<FPTYPE>::normalized_frobenius() const
    {
        assert(normalized_vals_CSC != NULL);
        return FPdot(get_nnzs(), normalized_vals_CSC, 1,
            normalized_vals_CSC, 1);
    }

    template<class FPTYPE>
    FPSparseMatrix<FPTYPE>::WordDocPair::WordDocPair(const word_id_t& word_, const doc_id_t& doc_)
        : word(word_), doc(doc_)
    {}

    template<class FPTYPE>
    FPSparseMatrix<FPTYPE>::WordDocPair::WordDocPair(const WordDocPair& from)
    {
        word = from.word;
        doc = from.doc;
    }

    template<class FPTYPE>
    void FPSparseMatrix<FPTYPE>::get_word_major_list(
        std::vector<WordDocPair>& entries,
        std::vector<offset_t>& word_offsets)
    {
        assert(entries.size() == 0);
        assert(word_offsets.size() == 0);

        entries.reserve(get_nnzs());
        for (doc_id_t doc = 0; doc < num_docs(); ++doc)
            for (offset_t pos = offsets_CSC[doc]; pos < offsets_CSC[doc + 1]; ++pos)
                entries.push_back(WordDocPair(rows_CSC[pos], doc));
        assert(entries.size() == get_nnzs());

        // Sort by words
        std::sort(entries.begin(), entries.end(), [](const auto& l, const auto& r)
        {return (l.word < r.word) || (l.word == r.word && l.doc < r.doc); });

        // Find word offsets
        for (word_id_t word = 0; word < vocab_size(); ++word)
            word_offsets.push_back(
                std::lower_bound(entries.begin(), entries.end(), WordDocPair(word, 0),
                    [](const auto& l, const auto& r) {return l.word < r.word; })
                - entries.begin());
        word_offsets.push_back(get_nnzs());
    }

    template<class FPTYPE>
    void FPSparseMatrix<FPTYPE>::initialize_for_eigensolver(const doc_id_t num_topics)
    {
        U_rows = vocab_size();
        U_cols = num_topics;
        U_colmajor = new FPTYPE[(size_t)num_topics * (size_t)vocab_size()];
        SigmaVT = new FPTYPE[(size_t)num_topics * (size_t)num_docs()];
    }

    template<class FPTYPE>
    void FPSparseMatrix<FPTYPE>::compute_Spectra(
        const doc_id_t num_topics,
        std::vector<FPTYPE>& evalues)
    {

        MKL_SpSpTrProd<FPTYPE> op(vals_CSC, rows_CSC, offsets_CSC,
            vocab_size(), num_docs(), get_nnzs());
        Spectra::SymEigsSolver<FPTYPE, Spectra::LARGEST_ALGE, MKL_SpSpTrProd<FPTYPE> >
            eigs(&op, (MKL_INT)num_topics, 2 * (MKL_INT)num_topics + 1);

        Eigen::Matrix<FPTYPE, Eigen::Dynamic, 1> evalues_mat;

        eigs.init();
        int nconv = eigs.compute();
        assert(nconv >= (int)num_topics); // Number of converged eig vals >= #topics
        assert(eigs.info() == Spectra::SUCCESSFUL);

        evalues_mat = eigs.eigenvalues();
        assert(evalues_mat(num_topics - 1) > 0.0);
        for (auto i = 0; i < num_topics; ++i)
            evalues.push_back(evalues_mat(i));


        // this->SigmaVT  = U^T*this
        auto U_Spectra = eigs.eigenvectors(num_topics);
        assert(U_Spectra.IsRowMajor == false);
        assert(U_Spectra.rows() == vocab_size() && U_Spectra.cols() == num_topics);
        memcpy(U_colmajor, U_Spectra.data(), U_rows * U_cols * sizeof(FPTYPE));
        compute_sigmaVT(num_topics);
    }

    template<class FPTYPE>
    void FPSparseMatrix<FPTYPE>::compute_block_ks(
        const doc_id_t num_topics,
        std::vector<FPTYPE>& evalues)
    {
        // MKL_SpSpTrProd<FPTYPE> op(vals_CSC, rows_CSC, offsets_CSC,
        //     vocab_size(), num_docs(), get_nnzs());
        // BlockKs<MKL_SpSpTrProd<FPTYPE>> eigensolver(&op, num_topics, 2 * num_topics + BLOCK_KS_BLOCK_SIZE,
        //                                             BLOCK_KS_MAX_ITERS, BLOCK_KS_BLOCK_SIZE, BLOCK_KS_TOLERANCE);

        // enumerate params for FlashProdOp    
        uint64_t n_rows = vocab_size();
        uint64_t n_cols = num_docs();
        std::swap(n_rows, n_cols);
        uint64_t nnzs = get_nnzs();

        MKL_INT* a_off_ptr = this->offsets_CSC;
        flash_ptr<MKL_INT> a_off = this->offsets_CSC_fptr;
        flash_ptr<MKL_INT> a_col = this->rows_CSC_fptr;
        flash_ptr<FPTYPE> a_csr = this->vals_CSC_fptr;
        
        // convert CSC to CSR
        flash_ptr<MKL_INT> a_tr_off = flash::flash_malloc<MKL_INT>((n_cols + 1) * sizeof(MKL_INT), "a_tr_off");
        flash_ptr<MKL_INT> a_tr_col = flash::flash_malloc<MKL_INT>(nnzs * sizeof(MKL_INT), "a_tr_col");
        flash_ptr<FPTYPE> a_tr_csr = flash::flash_malloc<FPTYPE>(nnzs * sizeof(FPTYPE), "a_tr_csr");
        flash::csrcsc(n_rows, n_cols, a_off, a_col, a_csr, a_tr_off, a_tr_col, a_tr_csr);
        MKL_INT* a_tr_off_ptr = new MKL_INT[n_cols + 1];
        flash::read_sync(a_tr_off_ptr, a_tr_off, n_cols + 1);
        GLOG_ASSERT(a_tr_off_ptr[n_cols] == a_off_ptr[n_rows],
                    "expected nnzs=", nnzs, ", got=", a_tr_off_ptr[n_cols]);
        
        // create FlashProdOp
        FlashProdOp op(a_csr, a_col, a_tr_csr, a_tr_col, a_off_ptr, a_tr_off_ptr,
                        n_rows, n_cols, nnzs, BLOCK_KS_BLOCK_SIZE);

        std::cout << "Op init done" << std::endl;

        BlockKs<FlashProdOp> eigensolver(&op,num_topics, 2 * num_topics + BLOCK_KS_BLOCK_SIZE,
                                         BLOCK_KS_MAX_ITERS, BLOCK_KS_BLOCK_SIZE, BLOCK_KS_TOLERANCE);
        eigensolver.init();
        eigensolver.compute();
        assert(eigensolver.num_converged() == num_topics);

        // cleanup from FlashProdOp
        flash::flash_free(a_tr_off);
        flash::flash_free(a_tr_col);
        flash::flash_free(a_tr_csr);

        ARMA_FPMAT sevecs = eigensolver.eigenvectors();
        ARMA_FPVEC sevs = eigensolver.eigenvalues();

        for (int i = 0; i < num_topics; ++i)
            evalues.push_back(sevs[i]);
        memcpy(U_colmajor, sevecs.memptr(), U_rows * U_cols * sizeof(FPTYPE));
        compute_sigmaVT(num_topics);
    }
    
    template<class FPTYPE>
    void FPSparseMatrix<FPTYPE>::compute_U_rowmajor() {
        assert(U_colmajor != NULL);
        assert(U_rowmajor == NULL);
        U_rowmajor = new FPTYPE[(size_t)U_rows*(size_t)U_cols];

        for (word_id_t r = 0; r < U_rows; ++r)
            for (auto c = 0; c < U_cols; ++c)
                U_rowmajor[(size_t)c + (size_t)r * (size_t)U_cols]
                = U_colmajor[(size_t)r + (size_t)c * (size_t)U_rows];

        auto tr_diff = std::abs(FPdot((size_t)U_rows * (size_t)U_cols,
            U_colmajor, 1, U_colmajor, 1))
            - FPdot((size_t)U_rows*(size_t)U_cols, U_rowmajor, 1, U_rowmajor, 1);

        if (tr_diff > 0.01)
            std::cout << "\n === WARNING : Diff between marix and transpose is "
            << tr_diff << "\n" << std::endl;
    }
    
    template<class FPTYPE>
    void FPSparseMatrix<FPTYPE>::compute_sigmaVT(const doc_id_t num_topics)
    {
        if (!U_rowmajor)
            compute_U_rowmajor();

        const char transa = 'N';
        const MKL_INT m = num_docs();
        const MKL_INT n = num_topics;
        const MKL_INT k = vocab_size();
        const char matdescra[6] = { 'G',0,0,'C',0,0 };
        FPTYPE alpha = 1.0; FPTYPE beta = 0.0;

        assert(sizeof(MKL_INT) == sizeof(offset_t));
        assert(sizeof(word_id_t) == sizeof(MKL_INT));
        assert(sizeof(offset_t) == sizeof(MKL_INT));

        FPcsrmm(&transa, &m, &n, &k, &alpha, matdescra,
            vals_CSC, (const MKL_INT*)rows_CSC, (const MKL_INT*)offsets_CSC, (const MKL_INT*)(offsets_CSC + 1),
            U_rowmajor, &n,
            &beta, SigmaVT, &n);
    }

    template<class FPTYPE>
    void FPSparseMatrix<FPTYPE>::cleanup_after_eigensolver()
    {
        assert(U_colmajor != NULL);
        delete[] U_colmajor;
        U_colmajor = NULL;

        assert(SigmaVT != NULL);
        delete[] SigmaVT;
        SigmaVT = NULL;
    }


    // Input: @from: Copy from here
    // Input: @zetas: zetas[word] indicates the threshold for each word
    // Input: @nnzs: Number of nnz elements that would remain after thresholding, pre-calculated
    // Input: @drop_empty: Drop all-zero cols while converting?
    // Output: @original_cols: For remaining cols, id to original cols, if drop_empty==true
    template<class FPTYPE>
    template <class fromT>
    void FPSparseMatrix<FPTYPE>::threshold_and_copy(
        const SparseMatrix<fromT>& from,
        const std::vector<fromT>& zetas,
        const offset_t nnzs,
        std::vector<doc_id_t>& original_cols)
    {
        assert(vocab_size() == from.vocab_size() && num_docs() == from.num_docs());
        assert(original_cols.size() == 0); assert(zetas.size() == vocab_size());

        // In case filtration has small error
        size_t extra = 1000;
        
        this->allocate_flash(nnzs + extra);
        
        offsets_CSC[0] = 0;
        offset_t this_pos = 0;
        doc_id_t nz_docs = 0;

        doc_id_t num_doc_blocks = divide_round_up(num_docs(), (doc_id_t)DOC_BLOCK_SIZE);
        // Do not parallelize this loop.
        for (doc_id_t block = 0; block < num_doc_blocks; ++block)
            threshold_and_copy_doc_block(block * DOC_BLOCK_SIZE,
                std::min((block + 1) * DOC_BLOCK_SIZE, num_docs()),
                this_pos, nz_docs, NULL, from, zetas, nnzs, original_cols);

        _num_docs = nz_docs;
        assert(original_cols.size() == nz_docs);
        std::cout << "Columns remaining after thresholding: " << nz_docs << "\n";

        if (offsets_CSC[nz_docs] != get_nnzs() - extra) {
            std::cout << "\n Estimated nnzs: " << get_nnzs() - extra
                << "  Last offset: " << offsets_CSC[nz_docs] << "\n";
            std::cout << " ************ WARNING: last offset != allocation ************* \n"
                << " ************ Resetting nnzs ********************************* \n\n";
            assert(get_nnzs() >= offsets_CSC[nz_docs]);
        }
        _nnzs = offsets_CSC[nz_docs];
        // write to disk
        flash::write_sync(this->offsets_CSC_fptr, this->offsets_CSC, this->num_docs() + 1);
        //avg_doc_sz = (count_t)get_nnzs() / num_docs();

        // for continuity
        this->read_flash();
    }

    //
    // Do not call this function in parallel. IT must be called by document block left to right
    //
    template<class FPTYPE>
    template <class fromT>
    void FPSparseMatrix<FPTYPE>::threshold_and_copy_doc_block(
        const doc_id_t doc_begin,
        const doc_id_t doc_end,
        offset_t& this_pos,
        doc_id_t& nz_docs,
        const bool* select_docs,             // pass NULL to pick all docs
        const SparseMatrix<fromT>& from,
        const std::vector<fromT>& zetas,
        const offset_t nnzs,
        std::vector<doc_id_t>& original_cols)
    {
        offset_t  from_blk_offset = from.offsets_CSC[doc_begin];
        offset_t  from_blk_nnzs = from.offsets_CSC[doc_end] - from.offsets_CSC[doc_begin];
        word_id_t *from_blk_rows_CSC = new word_id_t[from_blk_nnzs];
        fromT     *from_blk_vals_CSC = new fromT[from_blk_nnzs];

        offset_t  to_blk_offset = this_pos;
        offset_t  to_blk_nnzs = 0;
        word_id_t *to_blk_rows_CSC = new word_id_t[from_blk_nnzs];
        FPTYPE    *to_blk_vals_CSC = new FPTYPE[from_blk_nnzs];

        // read from_block_*_CSC
        flash::read_sync(from_blk_rows_CSC, from.rows_CSC_fptr + from_blk_offset, from_blk_nnzs);
        flash::read_sync(from_blk_vals_CSC, from.normalized_vals_CSC_fptr + from_blk_offset, from_blk_nnzs);

        // for (doc_id_t doc = doc_begin; doc < doc_end; ++doc) {
        //     if (select_docs == NULL || select_docs[doc]) {
        //         for (offset_t pos = from.offsets_CSC[doc]; pos < from.offsets_CSC[doc + 1]; ++pos) {
        //             fromT val;
        //             if (std::is_same<fromT, FPTYPE>::value)
        //                 val = std::round(from.normalized_vals_CSC[pos]);
        //             else if (std::is_same<fromT, count_t>::value)
        //                 val = from.normalized_vals_CSC[pos];
        //             else assert(false);
        //             if (val >= zetas[from.rows_CSC[pos]]) {
        //                 vals_CSC[this_pos] = (FPTYPE)std::sqrt(zetas[from.rows_CSC[pos]]);
        //                 rows_CSC[this_pos] = from.rows_CSC[pos];
        //                 ++this_pos;
        //             }
        //         }
        //     }
        //     if (this_pos > offsets_CSC[nz_docs]) {
        //         offsets_CSC[nz_docs + 1] = this_pos;
        //         nz_docs++;
        //         original_cols.push_back(doc);
        //     }
        // }
        for (doc_id_t doc = doc_begin; doc < doc_end; ++doc){
            if (select_docs == NULL || select_docs[doc]){
                for (offset_t pos = from.offsets_CSC[doc] - from_blk_offset;
                            pos < from.offsets_CSC[doc + 1] - from_blk_offset; ++pos){
                    word_id_t from_row_at_pos = from_blk_rows_CSC[pos];
                    fromT from_val_at_pos = from_blk_vals_CSC[pos];

                    fromT val;
                    if (std::is_same<fromT, FPTYPE>::value)
                        val = std::round(from_val_at_pos);
                    else if (std::is_same<fromT, count_t>::value)
                        val = from_val_at_pos;
                    else
                        assert(false);
                    if (val >= zetas[from_row_at_pos])
                    {
                        to_blk_vals_CSC[to_blk_nnzs] = (FPTYPE)std::sqrt(zetas[from_row_at_pos]);
                        to_blk_rows_CSC[to_blk_nnzs] = from_row_at_pos;
                        ++to_blk_nnzs;
                    }
                }
            }
            if (this_pos + to_blk_nnzs > offsets_CSC[nz_docs])
            {
                offsets_CSC[nz_docs + 1] = this_pos + to_blk_nnzs;
                nz_docs++;
                original_cols.push_back(doc);
            }
        }

        // write nz-docs from cur-block
        flash::write_sync(this->rows_CSC_fptr + to_blk_offset, to_blk_rows_CSC, to_blk_nnzs);
        flash::write_sync(this->vals_CSC_fptr + to_blk_offset, to_blk_vals_CSC, to_blk_nnzs);

        // update this_pos
        this_pos += to_blk_nnzs;
        GLOG_INFO("added nnzs=", to_blk_nnzs, ", range=[", to_blk_offset, ", ", this_pos, "]");

        // cleanup
        delete[] from_blk_rows_CSC;
        delete[] from_blk_vals_CSC;
        delete[] to_blk_rows_CSC;
        delete[] to_blk_vals_CSC;
    }

    template<class FPTYPE>
    template <class fromT>
    void FPSparseMatrix<FPTYPE>::sampled_threshold_and_copy(
        const SparseMatrix<fromT>& from,
        const std::vector<fromT>& zetas,
        const offset_t nnzs,
        std::vector<doc_id_t>& original_cols,
        const FPTYPE sample_rate)
    {
        assert(vocab_size() == from.vocab_size() && num_docs() == from.num_docs());
        assert(original_cols.size() == 0); assert(zetas.size() == vocab_size());

        //TODO: Need to cut down this buffer size.
        // In case filtration has small error
        size_t extra = 1000;
        this->allocate_flash(nnzs + extra);
        offsets_CSC[0] = 0;
        offset_t pos = 0, this_pos = 0;
        doc_id_t nz_docs = 0;

        auto doc_weights = new FPTYPE[from.num_docs()];
        FPscal(from.num_docs(), 0.0, doc_weights, 1);
        // Compute weights
        {
            // pfor_dynamic_131072(int64_t doc = 0; doc < num_docs(); ++doc) {
            //     for (offset_t pos = from.offsets_CSC[doc]; pos < from.offsets_CSC[doc + 1]; ++pos) {
            //         fromT val;
            //         if (std::is_same<fromT, FPTYPE>::value)
            //             val = std::round(from.normalized_vals_CSC[pos]);
            //         else if (std::is_same<fromT, count_t>::value)
            //             val = from.normalized_vals_CSC[pos];
            //         else assert(false);
            //         if (val >= zetas[from.rows_CSC[pos]])
            //             doc_weights[doc] += zetas[from.rows_CSC[pos]];
            //     }
            // }
        }
        // Compute weights using flash
        {
            uint64_t doc_blk_size = ((uint64_t)1 << 20);
            uint64_t from_num_docs = from.num_docs();
            uint64_t n_blks = divide_round_up(from_num_docs, doc_blk_size);

            // create and launch l2 compute tasks
            L2ThresholdedTask<fromT>** l2_tasks = new L2ThresholdedTask<fromT>*[n_blks];
            for(uint64_t blk = 0; blk < n_blks; blk++){
                uint64_t blk_start = doc_blk_size * blk;
                uint64_t blk_size = std::min(from_num_docs - blk_start, doc_blk_size);
                l2_tasks[blk] = new L2ThresholdedTask<fromT>(from.offsets_CSC, from.rows_CSC_fptr, from.normalized_vals_CSC_fptr,
                                                      zetas, doc_weights, blk_start, blk_size);
                flash::sched.add_task(l2_tasks[blk]);
            }
            
            // wait for tasks to complete
            flash::sleep_wait_for_complete(l2_tasks, n_blks);

            // cleanup
            for(uint64_t blk = 0; blk < n_blks; blk++){
                delete l2_tasks[blk];
            }
            delete[] l2_tasks;
        }

        auto dice = new FPTYPE[from.num_docs()];

        pfor_dynamic_131072(int64_t doc = 0; doc < num_docs(); ++doc) {
            dice[doc] = doc_weights[doc] == 0.0 ? 0.0 :
                std::pow(rand_fraction(), 1 / doc_weights[doc]);
            doc_weights[doc] = dice[doc];
        }
        std::nth_element(
            dice, dice + (size_t)(sample_rate*(FPTYPE)from.num_docs()), dice + from.num_docs(),
            std::greater<>());
        auto pivot = dice[(size_t)(sample_rate*(FPTYPE)from.num_docs())];
        std::cout << "sampling docs: pivot: " << pivot << std::endl;

        auto select_docs = new bool[num_docs()];
        pfor_dynamic_131072(int64_t doc = 0; doc < num_docs(); ++doc) {
            select_docs[doc] = (doc_weights[doc] >= pivot);
        }

        doc_id_t num_doc_blocks = divide_round_up(num_docs(), (doc_id_t)DOC_BLOCK_SIZE);
        // Do not parallelize this loop.
        for (doc_id_t block = 0; block < num_doc_blocks; ++block)
            threshold_and_copy_doc_block(block * DOC_BLOCK_SIZE,
                std::min((block + 1) * DOC_BLOCK_SIZE, num_docs()),
                this_pos, nz_docs, select_docs, from, zetas, nnzs, original_cols);

        _num_docs = nz_docs;
        assert(original_cols.size() == nz_docs);
        std::cout << "After sampling docs: cols remaining: " << nz_docs << "\n";

        assert(offsets_CSC[nz_docs] < get_nnzs() + extra);
        _nnzs = offsets_CSC[nz_docs];
        
        // write to disk
        flash::write_sync(this->offsets_CSC_fptr, this->offsets_CSC, this->num_docs() + 1);

        // TODO: RESIZE vals to something smaller.
        this->shrink(offsets_CSC[nz_docs]);

        delete[] select_docs;
        delete[] doc_weights;
        delete[] dice;

        // for continuity
        this->read_flash();
    }

    template<class FPTYPE>
    void FPSparseMatrix<FPTYPE>::left_multiply_by_U_Spectra(
        FPTYPE *const out,
        const FPTYPE *in,
        const doc_id_t ld_in,
        const doc_id_t ncols)
    {
        assert(U_rows == vocab_size());
        assert(ld_in >= (doc_id_t)U_cols);
        FPgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            (MKL_INT)vocab_size(), (MKL_INT)ncols, (MKL_INT)U_cols,
            (FPTYPE)1.0, U_colmajor, (MKL_INT)vocab_size(), in, (MKL_INT)ld_in,
            (FPTYPE)0.0, out, (MKL_INT)U_rows);
    }

    template<class FPTYPE>
    void FPSparseMatrix<FPTYPE>::copy_col_to(
        FPTYPE *const dst,
        const doc_id_t doc) const
    {
        FPscal(vocab_size(), 0.0, dst, 1);
        for (auto witer = offsets_CSC[doc]; witer < offsets_CSC[doc + 1]; ++witer)
            *(dst + rows_CSC[witer]) = vals_CSC[witer];
    }

    // pt must have at least vocab_size() entries
    template<class FPTYPE>
    inline FPTYPE FPSparseMatrix<FPTYPE>::distsq_doc_to_pt(
        const doc_id_t doc,
        const FPTYPE *const pt,
        const FPTYPE pt_l2sq) const
    {
        FPTYPE ret = (pt_l2sq == -1.0) ? FPdot(vocab_size(), pt, 1, pt, 1) : pt_l2sq;
        ret += FPdot(offsets_CSC[doc + 1] - offsets_CSC[doc],
            vals_CSC + offsets_CSC[doc], 1,
            vals_CSC + offsets_CSC[doc], 1);
        for (auto witer = offsets_CSC[doc]; witer < offsets_CSC[doc + 1]; ++witer)
            ret -= 2 * vals_CSC[witer] * pt[rows_CSC[witer]];
        return ret;
    }

    template<class FPTYPE>
    inline FPTYPE FPSparseMatrix<FPTYPE>::distsq_normalized_doc_to_pt(
        const doc_id_t doc,
        const FPTYPE *const pt,
        const FPTYPE pt_l2sq) const
    {
        FPTYPE ret = (pt_l2sq == -1.0) ? FPdot(vocab_size(), pt, 1, pt, 1) : pt_l2sq;
        ret += FPdot(offsets_CSC[doc + 1] - offsets_CSC[doc],
            normalized_vals_CSC + offsets_CSC[doc], 1,
            normalized_vals_CSC + offsets_CSC[doc], 1);
        for (auto witer = offsets_CSC[doc]; witer < offsets_CSC[doc + 1]; ++witer)
            ret -= 2 * normalized_vals_CSC[witer] * pt[rows_CSC[witer]];
        return ret;
    }

    template<class FPTYPE>
    void FPSparseMatrix<FPTYPE>::distsq_docs_to_centers(
        const word_id_t dim,
        doc_id_t num_centers,
        const FPTYPE *const centers,
        const FPTYPE *const centers_l2sq,
        const doc_id_t doc_begin,
        const doc_id_t doc_end,
        const FPTYPE *const docs_l2sq,
        FPTYPE *dist_matrix)
    {
        assert(doc_begin < doc_end);
        assert(doc_end <= num_docs());
        assert(sizeof(MKL_INT) == sizeof(offset_t));
        //assert(num_docs() >= num_centers);

        FPTYPE *ones_vec = new FPTYPE[std::max(doc_end-doc_begin, num_centers)];
        std::fill_n(ones_vec, std::max(doc_end - doc_begin, num_centers), (FPTYPE)1.0);

        FPTYPE *centers_tr = new FPTYPE[(size_t)num_centers*(size_t)vocab_size()];
        // Improve this
        for (word_id_t r = 0; r < vocab_size(); ++r)
            for (auto c = 0; c < num_centers; ++c)
                centers_tr[(size_t)c + (size_t)r * (size_t)num_centers]
                = centers[(size_t)r + (size_t)c * (size_t)vocab_size()];

        const char transa = 'N';
        const MKL_INT m = doc_end - doc_begin;
        const MKL_INT n = num_centers;
        const MKL_INT k = vocab_size();
        const char matdescra[6] = { 'G',0,0,'C',0,0 };
        FPTYPE alpha = -2.0; FPTYPE beta = 0.0;

        auto shifted_offsets_CSC = new MKL_INT[doc_end - doc_begin + 1];
        for (auto doc = doc_begin; doc <= doc_end; ++doc)
            shifted_offsets_CSC[doc - doc_begin] = offsets_CSC[doc] - offsets_CSC[doc_begin];

        FPcsrmm(&transa, &m, &n, &k, &alpha, matdescra,
            vals_CSC + offsets_CSC[doc_begin], (const MKL_INT*)(rows_CSC + offsets_CSC[doc_begin]),
            (const MKL_INT*)(shifted_offsets_CSC), (const MKL_INT*)(shifted_offsets_CSC + 1),
            centers_tr, &n, &beta, dist_matrix, &n);

        FPgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            doc_end - doc_begin, num_centers, 1,
            (FPTYPE)1.0, ones_vec, doc_end - doc_begin, centers_l2sq, num_centers,
            (FPTYPE)1.0, dist_matrix, num_centers);
        FPgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            doc_end - doc_begin, num_centers, 1,
            (FPTYPE)1.0, docs_l2sq, doc_end - doc_begin, ones_vec, num_centers,
            (FPTYPE)1.0, dist_matrix, num_centers);

        delete[] shifted_offsets_CSC;
        delete[] ones_vec;
        delete[] centers_tr;
    }

    template<class FPTYPE>
    void FPSparseMatrix<FPTYPE>::closest_centers(
        const doc_id_t num_centers,
        const FPTYPE *const centers,
        const FPTYPE *const centers_l2sq,
        const doc_id_t doc_begin,
        const doc_id_t doc_end,
        const FPTYPE *const docs_l2sq,
        doc_id_t *center_index,
        FPTYPE *const dist_matrix) 
    {
        assert(doc_begin < doc_end);
        assert(doc_end <= num_docs());
        distsq_docs_to_centers(vocab_size(), 
            num_centers, centers, centers_l2sq,
            doc_begin, doc_end, docs_l2sq, dist_matrix);

        pfor_static_131072(int64_t d = 0; d < doc_end - doc_begin; ++d)
            center_index[d] = (doc_id_t)FPimin(num_centers,
                dist_matrix + (size_t)d * (size_t)num_centers, 1);
    }

    template<class FPTYPE>
    void FPSparseMatrix<FPTYPE>::compute_centers_l2sq(
        FPTYPE * centers,
        FPTYPE * centers_l2sq,
        const doc_id_t num_centers)
    {
        pfor_static_256(int64_t c = 0; c < num_centers; ++c)
            centers_l2sq[c] = FPdot(vocab_size(),
                centers + (size_t)c * (size_t)vocab_size(), 1,
                centers + (size_t)c * (size_t)vocab_size(), 1);
    }

    template<class FPTYPE>
    FPTYPE FPSparseMatrix<FPTYPE>::lloyds_iter(
        const doc_id_t num_centers,
        FPTYPE *centers,
        const FPTYPE *const docs_l2sq,
        std::vector<doc_id_t> *closest_docs,
        bool compute_residual)
    {
        Timer timer;
        bool return_doc_partition = (closest_docs != NULL);

        doc_id_t doc_block_size = DOC_BLOCK_SIZE;
        doc_id_t num_doc_blocks = divide_round_up(num_docs(), doc_block_size);

        FPTYPE *const centers_l2sq = new FPTYPE[num_centers];
        FPTYPE *dist_matrix = new FPTYPE[(size_t)num_centers * (size_t)doc_block_size];
        doc_id_t *const closest_center = new doc_id_t[num_docs()];
        
        compute_centers_l2sq(centers, centers_l2sq, num_centers);
        for (doc_id_t block = 0; block < num_doc_blocks; ++block) {
            closest_centers(num_centers, centers, centers_l2sq,
                block * doc_block_size, std::min((block + 1) * doc_block_size, num_docs()),
                docs_l2sq + block * doc_block_size,
                closest_center + block * doc_block_size, dist_matrix);
        }
        timer.next_time_secs("lloyd: closest center", 30);

        memset(centers, 0, sizeof(FPTYPE) * (size_t)num_centers * (size_t)vocab_size());
        std::vector<size_t> cluster_sizes(num_centers, 0);
        for (doc_id_t block = 0; block < num_doc_blocks; ++block) {

            if (closest_docs == NULL)
                closest_docs = new std::vector<doc_id_t>[num_centers];
            else
                for (doc_id_t c = 0; c < num_centers; ++c)
                    closest_docs[c].clear();

            doc_id_t num_docs_in_block = std::min(doc_block_size, num_docs() - block*doc_block_size);

            for (doc_id_t d = block * doc_block_size; d < block*doc_block_size + num_docs_in_block; ++d) 
                closest_docs[closest_center[d]].push_back(d);

            for (size_t c = 0; c < num_centers; ++c)
                cluster_sizes[c] += closest_docs[c].size();

            pfor_dynamic_1(int c = 0; c < num_centers; ++c) {
                auto center = centers + (size_t)c * (size_t)vocab_size();
                auto div = (FPTYPE)closest_docs[c].size();
                for (auto diter = closest_docs[c].begin(); diter != closest_docs[c].end(); ++diter)
                    for (auto witer = offsets_CSC[*diter]; witer < offsets_CSC[1 + *diter]; ++witer)
                        *(center + rows_CSC[witer]) += vals_CSC[witer];
            }
        }

        // divide by number of points to obtain centroid
        pfor(auto center_id = 0; center_id < num_centers; ++center_id) {
            auto div = (FPTYPE)cluster_sizes[center_id];
            if (div > 0.0f)
                for (auto dim = 0; dim < vocab_size(); ++dim)
                    centers[(center_id * vocab_size()) + dim] /= div;
        }
        timer.next_time_secs("lloyd: find centers", 30);

        FPTYPE residual = 0.0;
        if (compute_residual) {
            int	BUF_PAD = 32;
            int CHUNK_SIZE = 8192;
            int nchunks = num_docs() / CHUNK_SIZE + (num_docs() % CHUNK_SIZE == 0 ? 0 : 1);
            std::vector<FPTYPE> residuals(nchunks*BUF_PAD, 0.0);

            pfor(int chunk = 0; chunk < nchunks; ++chunk)
                for (doc_id_t d = chunk*CHUNK_SIZE; d < num_docs() && d < (chunk + 1)*CHUNK_SIZE; ++d)
                    residuals[chunk*BUF_PAD] += distsq_doc_to_pt(d, centers + (size_t)closest_center[d] * (size_t)vocab_size());

            for (int chunk = 0; chunk < nchunks; ++chunk)
                residual += residuals[chunk*BUF_PAD];
            timer.next_time_secs("lloyd: residual", 30);
        }

        if (!return_doc_partition)
            delete[] closest_docs;
        else {
            for (doc_id_t c = 0; c < num_centers; ++c)
                closest_docs[c].clear();
            for (doc_id_t d = 0; d < num_docs(); ++d)
                closest_docs[closest_center[d]].push_back(d);
        }
        delete[] closest_center;
        delete[] dist_matrix;
        delete[] centers_l2sq;
        return residual;
    }

    template<class FPTYPE>
    void FPSparseMatrix<FPTYPE>::compute_docs_l2sq(FPTYPE *const docs_l2sq)
    {
        assert(docs_l2sq != NULL);
        FPscal(num_docs(), 0.0, docs_l2sq, 1);
        pfor_static_131072(int64_t d = 0; d < num_docs(); ++d)
            for (auto witer = offsets_CSC[d]; witer < offsets_CSC[d + 1]; ++witer)
                docs_l2sq[d] += vals_CSC[witer] * vals_CSC[witer];
    }

    template<class FPTYPE>
    FPTYPE FPSparseMatrix<FPTYPE>::run_lloyds(
        const doc_id_t			num_centers,
        FPTYPE					*centers,
        std::vector<doc_id_t>	*closest_docs, // Pass NULL if you dont want closest_docs
        const int				max_reps)
    {
        FPTYPE residual;
        bool return_clusters = (closest_docs != NULL);

        if (return_clusters)
            for (int center = 0; center < num_centers; ++center)
                assert(closest_docs[center].size() == 0);
        else
            closest_docs = new std::vector<doc_id_t>[num_centers];

        FPTYPE *docs_l2sq = new FPTYPE[num_docs()];
        compute_docs_l2sq(docs_l2sq);

        std::vector<size_t> prev_cl_sizes(num_centers, 0);
        auto prev_closest_docs = new std::vector<doc_id_t>[num_centers];

        Timer timer;
        for (int i = 0; i < max_reps; ++i) {
            residual = lloyds_iter(num_centers, centers, docs_l2sq, closest_docs);
            std::cout << "Lloyd's iter " << i << "  dist_sq residual: " << std::sqrt(residual) << "\n";
            timer.next_time_secs("run_lloyds: lloyds iter", 30);

            Timer timer;
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
            timer.next_time_secs("run_lloyds: check conv.", 30);
        }
        delete[] docs_l2sq;
        delete[] prev_closest_docs;
        if (!return_clusters)
            delete[] closest_docs;
        return residual;
    }

    template<class FPTYPE>
    void FPSparseMatrix<FPTYPE>::multiply_with(
        const doc_id_t doc_begin,
        const doc_id_t doc_end,
        const FPTYPE *const in,
        FPTYPE *const out,
        const MKL_INT cols)
    {
        assert(doc_begin < doc_end);
        assert(doc_end <= num_docs());

        assert(sizeof(MKL_INT) == sizeof(offset_t));
        assert(sizeof(word_id_t) == sizeof(MKL_INT));
        assert(sizeof(offset_t) == sizeof(MKL_INT));

        offset_t* offsets_CSC = this->offsets_CSC + doc_begin;
        flash_ptr<word_id_t> shifted_rows_CSC_fptr = this->rows_CSC_fptr + this->offsets_CSC[doc_begin];
        flash_ptr<FPTYPE> shifted_vals_CSC_fptr = this->vals_CSC_fptr + this->offsets_CSC[doc_begin];
        const doc_id_t doc_blk_size = doc_end - doc_begin;
        const word_id_t vocab_sz = vocab_size();
        Kmeans::multiply_with(offsets_CSC, shifted_rows_CSC_fptr, shifted_vals_CSC_fptr, doc_blk_size,
                              vocab_sz, in, out, cols);
        // // create shifted copy of offsets array
        // MKL_INT * shifted_offsets_CSC = new MKL_INT[doc_end - doc_begin + 1];
        // for (doc_id_t d = doc_begin; d <= doc_end; ++d) {
        //     shifted_offsets_CSC[d - doc_begin] = offsets_CSC[d] - offsets_CSC[doc_begin];
        // }

        // uint64_t doc_blk_size = doc_end - doc_begin;

        // uint64_t nnzs = shifted_offsets_CSC[doc_blk_size];
        // word_id_t *shifted_rows_CSC = new word_id_t[nnzs];
        // FPTYPE *shifted_vals_CSC = new FPTYPE[nnzs];

        // flash_ptr<word_id_t> shifted_rows_CSC_fptr = this->rows_CSC_fptr + offsets_CSC[doc_begin];
        // flash_ptr<FPTYPE> shifted_vals_CSC_fptr = this->vals_CSC_fptr + offsets_CSC[doc_begin];
        // flash::read_sync(shifted_rows_CSC, shifted_rows_CSC_fptr, nnzs);
        // flash::read_sync(shifted_vals_CSC, shifted_vals_CSC_fptr, nnzs);

        // const char transa = 'N';
        // const MKL_INT m = doc_blk_size;
        // const MKL_INT n = cols;
        // const MKL_INT k = vocab_size();
        // const char matdescra[6] = { 'G',0,0,'C',0,0 };
        // FPTYPE alpha = 1.0; FPTYPE beta = 0.0;

        // FPcsrmm(&transa, &m, &n, &k, &alpha, matdescra,
        //     shifted_vals_CSC, (const MKL_INT*)shifted_rows_CSC,
        //     (const MKL_INT*)shifted_offsets_CSC, (const MKL_INT*)(shifted_offsets_CSC + 1),
        //     in, &n, &beta, out, &n);

        // delete[] shifted_offsets_CSC;
        // delete[] shifted_rows_CSC;
        // delete[] shifted_vals_CSC;
    }

    template<class FPTYPE>
    void FPSparseMatrix<FPTYPE>::UT_times_docs(
        const doc_id_t doc_begin,
        const doc_id_t doc_end,
        FPTYPE* const projected_docs) 
    {
        multiply_with(doc_begin, doc_end, U_rowmajor, projected_docs, U_cols);
    }

    template<class FPTYPE>
    void FPSparseMatrix<FPTYPE>::distsq_projected_docs_to_projected_centers(
        const word_id_t dim,
        doc_id_t num_centers,
        const FPTYPE *const projected_centers_tr,  // This is row-major, of size (U_cols * num_centers)
        const FPTYPE *const projected_centers_l2sq,
        const doc_id_t doc_begin,
        const doc_id_t doc_end,
        const FPTYPE *const projected_docs_l2sq,
        FPTYPE *projected_dist_matrix)
    {
        assert(doc_begin < doc_end);
        assert(doc_end <= num_docs());
        //assert(num_centers == U_cols);
        assert(sizeof(MKL_INT) == sizeof(offset_t));
        //assert(num_docs() >= num_centers);
        offset_t *offsets_CSC = this->offsets_CSC + doc_begin;
        flash_ptr<word_id_t> shifted_rows_CSC_fptr = this->rows_CSC_fptr + this->offsets_CSC[doc_begin];
        flash_ptr<FPTYPE> shifted_vals_CSC_fptr = this->vals_CSC_fptr + this->offsets_CSC[doc_begin];
        const doc_id_t doc_blk_size = doc_end - doc_begin;
        const word_id_t vocab_sz = vocab_size();
        Kmeans::distsq_projected_docs_to_projected_centers(projected_centers_tr, projected_centers_l2sq,
                                                           offsets_CSC, shifted_rows_CSC_fptr,
                                                           shifted_vals_CSC_fptr, doc_blk_size,
                                                           projected_docs_l2sq, projected_dist_matrix,
                                                           dim, num_centers, U_rowmajor, U_rows, U_cols);

        // FPTYPE *ones_vec = new FPTYPE[std::max(doc_end - doc_begin, num_centers)];
        // std::fill_n(ones_vec, std::max(doc_end - doc_begin, num_centers), (FPTYPE)1.0);

        // const char transa = 'N';
        // const MKL_INT m = (doc_end - doc_begin);
        // const MKL_INT n = num_centers;
        // const MKL_INT k = U_cols;
        // const char matdescra[6] = {'G',0,0,'C',0,0};

        // FPTYPE *UUTrC = new FPTYPE[U_rows * num_centers];
        // FPgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        //     U_rows, num_centers, U_cols, (FPTYPE)-2.0,
        //     U_rowmajor, U_cols, projected_centers_tr, num_centers,
        //     (FPTYPE)0.0, UUTrC, num_centers);
        // multiply_with(doc_begin, doc_end, UUTrC,
        //     projected_dist_matrix, num_centers);
        // delete[] UUTrC;

        // /*FPTYPE *projected_docs = new FPTYPE[U_cols * (doc_end - doc_begin) * sizeof(FPTYPE)];
        // // projected_docs : (doc_end-doc_begin) x num_topics (U_cols) [row-major]
        // UT_times_docs(doc_begin, doc_end, projected_docs);   
        // // data_block^T, U (data block is in col_major, 
        // FPgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        //     m, n, k, (FPTYPE)-2.0,
        //     projected_docs, k, projected_centers_tr, n,
        //     (FPTYPE)0.0, projected_dist_matrix, n);
        // delete[] projected_docs;*/

        // FPgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
        //     m, n, 1, (FPTYPE)1.0,
        //     ones_vec, m, projected_centers_l2sq, n,
        //     (FPTYPE)1.0, projected_dist_matrix, n);

        // FPgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
        //     m, n, 1, (FPTYPE)1.0,
        //     projected_docs_l2sq, m, ones_vec, n,
        //     (FPTYPE)1.0, projected_dist_matrix, n);

        // delete[] ones_vec;
    }

    template<class FPTYPE>
    void FPSparseMatrix<FPTYPE>::projected_closest_centers(
        const doc_id_t num_centers,
        const FPTYPE *const projected_centers_tr,
        const FPTYPE *const projected_centers_l2sq,
        const doc_id_t doc_begin,
        const doc_id_t doc_end,
        const FPTYPE *const projected_docs_l2sq,
        doc_id_t *center_index,
        FPTYPE *const projected_dist_matrix)
    {
        assert(doc_begin < doc_end);
        assert(doc_end <= num_docs());

        offset_t *offsets_CSC = this->offsets_CSC + doc_begin;
        flash_ptr<word_id_t> shifted_rows_CSC_fptr = this->rows_CSC_fptr + this->offsets_CSC[doc_begin];
        flash_ptr<FPTYPE> shifted_vals_CSC_fptr = this->vals_CSC_fptr + this->offsets_CSC[doc_begin];
        const doc_id_t doc_blk_size = doc_end - doc_begin;
        const word_id_t vocab_sz = vocab_size();
        Kmeans::projected_closest_centers(projected_centers_tr, projected_centers_l2sq,
                                          offsets_CSC, shifted_rows_CSC_fptr, shifted_vals_CSC_fptr,
                                          doc_blk_size, num_centers, vocab_sz, U_rowmajor, 
                                          U_rows, U_cols, projected_docs_l2sq, center_index,
                                          projected_dist_matrix);
        // distsq_projected_docs_to_projected_centers(vocab_size(),
        //     num_centers, projected_centers_tr, projected_centers_l2sq,
        //     doc_begin, doc_end, projected_docs_l2sq, projected_dist_matrix);

        // pfor_static_131072(int64_t d = 0; d < doc_end - doc_begin; ++d)
        //     center_index[d] = (doc_id_t)FPimin(num_centers,
        //         projected_dist_matrix + (size_t)d * (size_t)num_centers, 1);
    }

    template<class FPTYPE>
    void FPSparseMatrix<FPTYPE>::compute_projected_centers_l2sq(
        FPTYPE * projected_centers,
        FPTYPE * projected_centers_l2sq,
        const doc_id_t num_centers)
    {
        assert(U_cols == num_centers);
        pfor_static_256(int64_t c = 0; c < num_centers; ++c)
            projected_centers_l2sq[c] = FPdot(num_centers,
                projected_centers + (size_t)c * (size_t)num_centers, 1,
                projected_centers + (size_t)c * (size_t)num_centers, 1);
    }


    template<class FPTYPE>
    void FPSparseMatrix<FPTYPE>::compute_projected_docs_l2sq(
        FPTYPE *const projected_docs_l2sq)
    {
        assert(projected_docs_l2sq != NULL);

        const doc_id_t doc_block_size = DOC_BLOCK_SIZE;
        const doc_id_t num_doc_blocks = divide_round_up(num_docs(), doc_block_size);

        auto projected_doc_block = new FPTYPE[doc_block_size * U_cols * sizeof(FPTYPE)];

        for (doc_id_t block = 0; block < num_doc_blocks; ++block) {
            const doc_id_t doc_begin = block * doc_block_size;
            const doc_id_t doc_end = std::min(num_docs(), (block + 1)* doc_block_size);

            // project  docs
            UT_times_docs(doc_begin, doc_end, projected_doc_block);
            
            // compute l2sq norm of docs
            pfor_static_131072(int d = 0; d < (doc_end - doc_begin); ++d)
                projected_docs_l2sq[d + doc_begin] 
                = FPdot(U_cols, 
                    projected_doc_block + d * U_cols, 1,
                    projected_doc_block + d * U_cols, 1);

            // reset buffer
            memset(projected_doc_block, 0, doc_block_size * U_cols * sizeof(FPTYPE));
        }

        // free mem
        delete[] projected_doc_block;
    }
    
    template<class FPTYPE>
    FPTYPE FPSparseMatrix<FPTYPE>::lloyds_iter_on_projected_space(
        const doc_id_t num_centers,
        FPTYPE *projected_centers,
        const FPTYPE *const projected_docs_l2sq,
        std::vector<doc_id_t> *closest_docs,
        bool compute_residual) 
    {
        Timer timer;
        bool return_doc_partition = (closest_docs != NULL);

        doc_id_t doc_block_size = DOC_BLOCK_SIZE;
        doc_id_t num_doc_blocks = divide_round_up(num_docs(), doc_block_size);

        FPTYPE *const projected_centers_l2sq = new FPTYPE[num_centers];
        FPTYPE *projected_dist_matrix = new FPTYPE[(size_t)num_centers * (size_t)doc_block_size];
        doc_id_t *const closest_center = new doc_id_t[num_docs()];

        compute_projected_centers_l2sq(projected_centers, projected_centers_l2sq, num_centers);
        
        // projected_centers_tr = num_centers x num_topics [row-major, each column a center]
        FPTYPE *projected_centers_tr = new FPTYPE[(size_t)num_centers * (size_t)U_cols];
        for (word_id_t r = 0; r < U_cols; ++r)
            for (auto c = 0; c < num_centers; ++c)
                projected_centers_tr[(size_t)c + (size_t)r * (size_t)num_centers]
                = projected_centers[(size_t)r + (size_t)c * (size_t)U_cols];

        for (doc_id_t block = 0; block < num_doc_blocks; ++block) {
            projected_closest_centers(num_centers, projected_centers_tr, projected_centers_l2sq,
                block * doc_block_size, std::min((block + 1) * doc_block_size, num_docs()),
                projected_docs_l2sq + block * doc_block_size,
                closest_center + block * doc_block_size, projected_dist_matrix);
        }
        timer.next_time_secs("lloyd: closest center", 30);

        memset(projected_centers, 0, sizeof(FPTYPE) * (size_t)num_centers * (size_t)num_centers);
        std::vector<size_t> cluster_sizes(num_centers, 0);

        FPTYPE *projected_docs = new FPTYPE[doc_block_size * num_centers];
        for (doc_id_t block = 0; block < num_doc_blocks; ++block) {

            if (closest_docs == NULL)
                closest_docs = new std::vector<doc_id_t>[num_centers];
            else
                for (doc_id_t c = 0; c < num_centers; ++c)
                    closest_docs[c].clear();

            doc_id_t num_docs_in_block = std::min(doc_block_size, num_docs() - block*doc_block_size);

            for (doc_id_t d = block * doc_block_size; d < block*doc_block_size + num_docs_in_block; ++d)
                closest_docs[closest_center[d]].push_back(d);

            for (size_t c = 0; c < num_centers; ++c)
                cluster_sizes[c] += closest_docs[c].size();

            UT_times_docs(block * doc_block_size,
                block * doc_block_size + num_docs_in_block,
                projected_docs);

            pfor_dynamic_1(int c = 0; c < num_centers; ++c) {
                FPTYPE* center = projected_centers + (size_t)c * (size_t)num_centers;
                for (auto diter = closest_docs[c].begin(); diter != closest_docs[c].end(); ++diter)
                    FPaxpy(num_centers, 1.0, projected_docs + ((*diter) - block*doc_block_size)*num_centers, 1, center, 1);
            }
        }
        delete[] projected_docs;

        // divide by number of points to obtain centroid
        for (auto center_id = 0; center_id < num_centers; ++center_id) {
            auto div = (FPTYPE)cluster_sizes[center_id];
            if (div > 0.0f)
                FPscal(num_centers, 1.0f / div, projected_centers + center_id * num_centers, 1);
        }
        timer.next_time_secs("lloyd: find centers", 30);

        FPTYPE residual = 0.0;
        if (compute_residual) {
            assert(false); // Need to fill in
        }

        if (!return_doc_partition)
            delete[] closest_docs;
        else {
            for (doc_id_t c = 0; c < num_centers; ++c)
                closest_docs[c].clear();
            for (doc_id_t d = 0; d < num_docs(); ++d)
                closest_docs[closest_center[d]].push_back(d);
        }
        delete[] closest_center;
        delete[] projected_centers_tr;
        delete[] projected_dist_matrix;
        delete[] projected_centers_l2sq;
        return residual;
    }

    template<class FPTYPE>
    FPTYPE FPSparseMatrix<FPTYPE>::run_lloyds_on_projected_space(
        const doc_id_t			num_centers,
        FPTYPE					*projected_centers,
        std::vector<doc_id_t>	*closest_docs,
        const int				max_reps)
    {
        FPTYPE residual;
        bool return_clusters = (closest_docs != NULL);

        if (return_clusters)
            for (int center = 0; center < num_centers; ++center)
                assert(closest_docs[center].size() == 0);
        else
            closest_docs = new std::vector<doc_id_t>[num_centers];

        FPTYPE *projected_docs_l2sq = new FPTYPE[num_docs()];
        compute_projected_docs_l2sq(projected_docs_l2sq);
        
        std::vector<size_t> prev_cl_sizes(num_centers, 0);
        auto prev_closest_docs = new std::vector<doc_id_t>[num_centers];

        Timer timer;
        for (int i = 0; i < max_reps; ++i) {
            residual = lloyds_iter_on_projected_space(num_centers, 
                projected_centers, projected_docs_l2sq, closest_docs);
            timer.next_time_secs("run_lloyds: lloyds iter", 30);

            Timer timer;
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
            timer.next_time_secs("run_lloyds: check conv.", 30);
        }
        delete[] projected_docs_l2sq;
        delete[] prev_closest_docs;
        if (!return_clusters)
            delete[] closest_docs;
        return residual;
    }

    template<class FPTYPE>
    void FPSparseMatrix<FPTYPE>::update_min_distsq_to_projected_centers(
        const word_id_t dim,
        const doc_id_t num_centers,
        const FPTYPE *const projected_centers,
        const doc_id_t doc_begin,
        const doc_id_t doc_end,
        const FPTYPE *const projected_docs_l2sq,
        FPTYPE *min_dist,
        FPTYPE *projected_dist) // preallocated scratch of space num_docs
    {
        assert(doc_end >= doc_begin);
        assert(doc_end <= num_docs());
        assert(min_dist != NULL);

        bool dist_alloc = false;
        if (projected_dist == NULL) {
            projected_dist = new FPTYPE[(size_t)(doc_end - doc_begin) * (size_t)num_centers];
            dist_alloc = true;
        }
        FPTYPE *projected_center_l2sq = new FPTYPE[num_centers];
        for (auto c = 0; c < num_centers; ++c)
            projected_center_l2sq[c] = FPdot(dim,
                projected_centers + (size_t)c * (size_t)dim, 1,
                projected_centers + (size_t)c * (size_t)dim, 1);

        FPTYPE *projected_centers_tr = new FPTYPE[(size_t)num_centers * (size_t)U_cols];
        // Improve this
        for (word_id_t r = 0; r < U_cols; ++r)
            for (auto c = 0; c < num_centers; ++c)
                projected_centers_tr[(size_t)c + (size_t)r * (size_t)num_centers]
                = projected_centers[(size_t)r + (size_t)c * (size_t)U_cols];

        distsq_projected_docs_to_projected_centers(dim,
            num_centers, projected_centers_tr, projected_center_l2sq,
            doc_begin, doc_end, projected_docs_l2sq,
            projected_dist);

        pfor_static_131072(int d = doc_begin; d < doc_end; ++d) {
            if (num_centers == 1) {
                // Round about for small negative distances
                size_t pos = (size_t)d - (size_t)doc_begin;
                projected_dist[pos] = std::max(projected_dist[pos], (FPTYPE)0.0);
                min_dist[d] = std::min(min_dist[d], projected_dist[pos]);
            }
            else {
                for (doc_id_t c = 0; c < num_centers; ++c) {
                    size_t pos = (size_t)c + (size_t)d * (size_t)num_centers - (size_t)doc_begin * (size_t)num_centers;
                    projected_dist[pos] = std::max(projected_dist[pos], (FPTYPE)0.0);
                    min_dist[d] = std::min(min_dist[d], projected_dist[pos]);
                }
            }
        }
        delete[] projected_center_l2sq;
        delete[] projected_centers_tr;
        if (dist_alloc) delete[] projected_dist;
    }

    template<class FPTYPE>
    FPTYPE FPSparseMatrix<FPTYPE>::kmeanspp_on_projected_space(
        const doc_id_t k,
        std::vector<doc_id_t>&centers)
    {
        assert(num_docs() >= k);

        FPTYPE *const centers_l2sq = new FPTYPE[k];
        FPTYPE *const projected_docs_l2sq = new FPTYPE[num_docs()];
        FPTYPE *const min_dist = new FPTYPE[num_docs()];
        FPTYPE *const centers_coords = new FPTYPE[(size_t)k * (size_t)U_cols];
        std::vector<FPTYPE> dist_cumul(num_docs() + 1);

        memset(projected_docs_l2sq, 0, sizeof(FPTYPE) * num_docs());
        compute_projected_docs_l2sq(projected_docs_l2sq);

        memset(centers_coords, 0, sizeof(FPTYPE) * (size_t)k * (size_t)U_cols);
        std::fill_n(min_dist, num_docs(), FP_MAX);
        centers.push_back((doc_id_t)((size_t)rand() * (size_t)84619573 % (size_t)num_docs())); // Pick a random center
        UT_times_docs(centers[0], centers[0] + 1, centers_coords);
        centers_l2sq[0] = FPdot(U_cols, centers_coords, 1, centers_coords, 1);
        assert(std::abs(centers_l2sq[0] - projected_docs_l2sq[centers[0]]) < 1e-6);

        const doc_id_t doc_block_size = DOC_BLOCK_SIZE;
        const doc_id_t num_doc_blocks = divide_round_up(num_docs(), doc_block_size);
        GLOG_DEBUG("num_doc_blks=", num_doc_blocks);

        FPTYPE *dist_scratch_space = new FPTYPE[num_docs()];
        FPTYPE *ones_vec = new FPTYPE[num_docs()];
        std::fill_n(ones_vec, num_docs(), (FPTYPE)1.0);
        int new_centers_added = 1;
        while (centers.size() < k) {
            std::cout << "centers.size():  " << centers.size() 
                << "   new_centers_added: " << new_centers_added << std::endl;
            for(int64_t block = 0; block < divide_round_up(num_docs(), (doc_id_t)DOC_BLOCK_SIZE); ++block)
                update_min_distsq_to_projected_centers(U_cols, new_centers_added,
                    centers_coords + (size_t)(centers.size() - new_centers_added) * (size_t)U_cols,
                    block*DOC_BLOCK_SIZE, std::min(((doc_id_t)block + 1)*(doc_id_t)DOC_BLOCK_SIZE, num_docs()),
                    projected_docs_l2sq + block*DOC_BLOCK_SIZE, min_dist, NULL);
            dist_cumul[0] = 0;
            for (doc_id_t doc = 0; doc < num_docs(); ++doc)
                dist_cumul[doc + 1] = dist_cumul[doc] + min_dist[doc];
            for (auto iter = centers.begin(); iter != centers.end(); ++iter) {
                // Distance from center to its closest center == 0
                assert(abs(dist_cumul[(*iter) + 1] - dist_cumul[*iter]) < 1e-4);
                // Center is not replicated
                assert(std::find(centers.begin(), centers.end(), *iter) == iter);
                assert(std::find(iter + 1, centers.end(), *iter) == centers.end());
            }

            int s = centers.size();
            new_centers_added = 0;
            for (int c = 0; (c < 1 + std::sqrt(s - 5 > 0 ? s - 5 : 0)) && (centers.size() < k); ++c) {
                auto dice_throw = dist_cumul[num_docs()] * rand_fraction();
                assert(dice_throw < dist_cumul[num_docs()]);
                doc_id_t new_center
                    = (doc_id_t)(std::upper_bound(dist_cumul.begin(), dist_cumul.end(), dice_throw)
                        - 1 - dist_cumul.begin());
                assert(new_center < num_docs());
                if (std::find(centers.begin(), centers.end(), new_center) == centers.end()) {
                    UT_times_docs(new_center, new_center + 1, 
                        centers_coords + centers.size() * (size_t)U_cols);
                    centers_l2sq[centers.size()] = FPdot(U_cols,
                        centers_coords + centers.size() * (size_t)U_cols, 1,
                        centers_coords + centers.size() * (size_t)U_cols, 1);
                    centers.push_back(new_center);
                    new_centers_added++;
                }
            }

        }
        delete[] ones_vec;
        delete[] dist_scratch_space;
        delete[] centers_l2sq;
        delete[] projected_docs_l2sq;
        delete[] min_dist;
        delete[] centers_coords;
        return dist_cumul[num_docs() - 1];
    }

    template<class FPTYPE>
    FPTYPE FPSparseMatrix<FPTYPE>::kmeans_init_on_projected_space(
        const int num_centers,
        const int max_reps,
        std::vector<doc_id_t>&	best_seed,   // Wont be initialized if method==KMEANSBB
        FPTYPE *const			best_centers_coords) // Wont be initialized if null
    {
        FPTYPE min_total_dist_to_centers = FP_MAX;
        int best_rep;

        auto kmeans_seeds = new std::vector<doc_id_t>[max_reps];
        for (int rep = 0; rep < max_reps; ++rep) {
            FPTYPE dist;
            dist = kmeanspp_on_projected_space(num_centers, kmeans_seeds[rep]);
            std::cout << "k-means init residual: " << dist << std::endl;
            if (dist < min_total_dist_to_centers) {
                min_total_dist_to_centers = dist;
                best_rep = rep;
            }
        }
        best_seed = kmeans_seeds[best_rep];
        for (doc_id_t d = 0; d < num_centers; ++d)
            UT_times_docs(best_seed[d], best_seed[d] + 1,
                best_centers_coords + (size_t)d * (size_t)num_centers);
        delete[] kmeans_seeds;
        
        return min_total_dist_to_centers;
    }

    // Input: @num_centers, @centers: coords of centers to start the iteration, @print_residual
    // Output: @closest_docs: if NULL, nothing is returned; is !NULL, return partition of docs between centers
    template<class FPTYPE>
    FPTYPE FPSparseMatrix<FPTYPE>::run_elkans(
        const doc_id_t			num_centers,
        FPTYPE					*centers,
        std::vector<doc_id_t>	*closest_docs, // Pass NULL if you dont want closest_docs
        const int				max_reps)
    {
        Timer   timer;
        FPTYPE	residual;
        bool	return_clusters = (closest_docs != NULL);

        if (return_clusters)
            for (int center = 0; center < num_centers; ++center)
                assert(closest_docs[center].size() == 0);
        else
            closest_docs = new std::vector<doc_id_t>[num_centers];
        assert(closest_docs != NULL);

        /*
        Initialization:
        � Pick initial centers and set l(x, c) = 0 for all x, c.
        � Assign c(x) = argmin_c d(x, c), d(c, c0) from Step - 1.
        � Every time we compute d(x, c), set l(x, c) = d(x, c).
        � Assign u(x) = min_c d(x, c) = d(x, c(x)).
        */
        FPTYPE   *const docs_l2sq = new FPTYPE[num_docs()];
        FPTYPE   *const ub_distsq = new FPTYPE[num_docs()];
        FPTYPE   *const lb_distsq_matrix = new FPTYPE[(size_t)num_centers * (size_t)num_docs()];
        FPTYPE   *const centers_distsq = new FPTYPE[(size_t)num_centers * (size_t)num_centers];
        doc_id_t *const closest_center = new doc_id_t[num_docs()];
        FPTYPE	 *const centers_l2sq = new FPTYPE[num_centers];

        timer.next_time_secs("Elkan: Alloc", 30);

        compute_centers_l2sq(centers, centers_l2sq, num_centers);
        compute_docs_l2sq(docs_l2sq);
        closest_centers(num_centers, centers, centers_l2sq, 
            0, num_docs(), docs_l2sq, closest_center, lb_distsq_matrix);
        //pfor_static_131072 (auto doc = 0; doc < num_docs(); ++doc)
        //	ub_distsq[doc] = distsq_doc_to_pt(doc, centers + vocab_size() * closest_center[doc]);
        auto nearest_docs = new std::vector<doc_id_t>[num_centers];
        for (doc_id_t doc = 0; doc < num_docs(); ++doc)
            nearest_docs[closest_center[doc]].push_back(doc);
        pfor_dynamic_16(auto c = 0; c < num_centers; ++c) {
            centers_l2sq[c] = FPdot(vocab_size(),
                centers + (size_t)c * (size_t)vocab_size(), 1,
                centers + (size_t)c * (size_t)vocab_size(), 1);
            for (auto diter = nearest_docs[c].begin(); diter != nearest_docs[c].end(); ++diter)
                ub_distsq[*diter] = distsq_doc_to_pt(*diter, centers + (size_t)vocab_size() * (size_t)c, centers_l2sq[c]);
        }
        delete[] nearest_docs;

        FPscal((size_t)num_centers * (size_t)num_centers, 0.0, centers_distsq, 1);
        timer.next_time_secs("Elkan: init centers", 30);

        for (int rep = 0; rep < MAX_KMEANS_REPS; ++rep)
        {
            // 1. Compute d(c, c0) for all c, c0. 
            //    Compute s(c) = 1 / 2 min_{ c'!=c} d(c, c0) for all c.
            auto ones_vec = new FPTYPE[num_centers];
            pfor_static_256(auto i = 0; i < num_centers; ++i) {
                centers_l2sq[i] = FPdot(vocab_size(),
                    centers + (size_t)i * (size_t)vocab_size(), 1,
                    centers + (size_t)i * (size_t)vocab_size(), 1);
                ones_vec[i] = 1.0;
            }
            {
                FPgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    num_centers, num_centers, vocab_size(),
                    (FPTYPE)-2.0, centers, vocab_size(), centers, vocab_size(),
                    (FPTYPE)0.0, centers_distsq, num_centers);
                FPgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    num_centers, num_centers, 1,
                    (FPTYPE)1.0, ones_vec, num_centers, centers_l2sq, num_centers,
                    (FPTYPE)1.0, centers_distsq, num_centers);
                FPgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    num_centers, num_centers, 1,
                    (FPTYPE)1.0, centers_l2sq, num_centers, ones_vec, num_centers,
                    (FPTYPE)1.0, centers_distsq, num_centers);
            }
            std::vector<FPTYPE> distsq_nearest_other_center(num_centers);
            pfor_dynamic_1024(auto c = 0; c < num_centers; ++c) {
                auto imin1 = FPimin(c, centers_distsq + (size_t)c * (size_t)num_centers, 1);
                auto imin2 = FPimin(num_centers - 1 - c, centers_distsq + (size_t)c * (size_t)num_centers + c + 1, 1);
                auto min1 = centers_distsq[(size_t)c * (size_t)num_centers + imin1];
                auto min2 = centers_distsq[(size_t)c * (size_t)num_centers + c + 1 + imin2];
                distsq_nearest_other_center[c] = min1 < min2 ? min1 : min2;
            }
            timer.next_time_secs("Elkan: Center distances", 30);

            // 2. Identify x such that u(x) <= s(c(x)).
            // avoid calculations for all such x beyond this point.
            // 3. For all remaining x, c such that 
            //    (i) c != c(x), (ii) u(x) > l(x,c) (iii) u(x) > 1/2 * d(c(x),c)
            // (a) Compute d(x, c(x)) and update u(x) = d(x, c(x)).
            // (b) If d(x,c(x)) > l(x,c) or d(x,c(x)) > 1/2 * d(c(x),c), compute d(x,c).
            //              Now, if d(x,c) < d(x,c(x)), then assign c(x)=c & update u(x) = d(x,c(x)).
            auto nearest_docs = new std::vector<doc_id_t>[num_centers];
            for (doc_id_t doc = 0; doc < num_docs(); ++doc)
                nearest_docs[closest_center[doc]].push_back(doc);
            pfor_dynamic_16(auto c = 0; c < num_centers; ++c)
                for (auto diter = nearest_docs[c].begin(); diter != nearest_docs[c].end(); ++diter)
                    ub_distsq[*diter] = distsq_doc_to_pt(*diter, centers + (size_t)vocab_size() * (size_t)c, centers_l2sq[c]);
            timer.next_time_secs("Elkan: Updating UB", 30);
            delete[] nearest_docs;


            auto doc_batch_size = (doc_id_t)131072;
            auto num_doc_batches = divide_round_up(num_docs(), doc_batch_size);
            auto moving_docs = new std::vector<doc_id_t>[num_doc_batches];
            pfor_dynamic_1(auto batch = 0; batch < num_doc_batches; ++batch)
                for (doc_id_t doc = batch*doc_batch_size;
                    doc < (batch + 1)*doc_batch_size && doc < num_docs(); ++doc) {
                auto distsq_to_current_center = ub_distsq[doc];
                auto c = closest_center[doc];
                if (distsq_to_current_center < 0.25 * distsq_nearest_other_center[c])
                    continue;
                auto imin1 = FPimin(c, lb_distsq_matrix + (size_t)doc * (size_t)num_centers, 1);
                auto imin2 = FPimin(num_centers - 1 - c, lb_distsq_matrix + (size_t)doc * (size_t)num_centers + c + 1, 1);
                auto min1 = lb_distsq_matrix[(size_t)doc * (size_t)num_centers + imin1];
                auto min2 = lb_distsq_matrix[(size_t)doc * (size_t)num_centers + c + 1 + imin2];
                auto min_distsq_to_other_center = min1 < min2 ? min1 : min2;

                if (distsq_to_current_center > min_distsq_to_other_center)
                    moving_docs[batch].push_back(doc);
            }
            size_t moved_docs = 0;
            for (doc_id_t b = 0; b < num_doc_batches; ++b) moved_docs += moving_docs[b].size();
            std::cout << "\n#Docs moved: " << moved_docs << "\n";
            timer.next_time_secs("Elkan: Finding moving docs", 30);

            pfor_dynamic_1(auto batch = 0; batch < num_doc_batches; ++batch) {
                size_t buffer_sz = 0;
                for (auto diter = moving_docs[batch].begin(); diter != moving_docs[batch].end(); ++diter)
                    buffer_sz += offset_CSC(*diter + 1) - offset_CSC(*diter);
                auto buffer_vals_CSC = new FPTYPE[buffer_sz];
                auto buffer_rows_CSC = new word_id_t[buffer_sz];
                auto buffer_offsets_CSC = new offset_t[moving_docs[batch].size() + 1];
                auto buffer_dist_matrix = new FPTYPE[moving_docs[batch].size() * (size_t)num_centers];
                buffer_offsets_CSC[0] = 0;
                for (size_t i = 0; i != moving_docs[batch].size(); ++i) {
                    auto doc = moving_docs[batch][i];
                    offset_t num_nnzs = offset_CSC(doc + 1) - offset_CSC(doc);
                    memcpy(buffer_vals_CSC + buffer_offsets_CSC[i],
                        vals_CSC + offset_CSC(doc), sizeof(FPTYPE) * (size_t)num_nnzs);
                    memcpy(buffer_rows_CSC + buffer_offsets_CSC[i],
                        rows_CSC + offset_CSC(doc), sizeof(word_id_t) * (size_t)num_nnzs);
                    buffer_offsets_CSC[i + 1] = buffer_offsets_CSC[i] + num_nnzs;
                }

                const char transa = 'N';
                const MKL_INT m = moving_docs[batch].size();
                const MKL_INT n = num_centers;
                const MKL_INT k = vocab_size();
                const char matdescra[6] = { 'G',0,0,'C',0,0 };
                FPTYPE alpha = -2.0; FPTYPE beta = 0.0;

                FPTYPE *ones_vec = new FPTYPE[m > n ? m : n];
                std::fill_n(ones_vec, m > n ? m : n, (FPTYPE)1.0);

                FPTYPE *centers_tr = new FPTYPE[(size_t)n*(size_t)vocab_size()];
                for (word_id_t r = 0; r < vocab_size(); ++r)
                    for (auto c = 0; c < n; ++c)
                        centers_tr[(size_t)c + (size_t)r * (size_t)n]
                        = centers[(size_t)r + (size_t)c * (size_t)vocab_size()];

                FPcsrmm(&transa, &m, &n, &k, &alpha, matdescra,
                    buffer_vals_CSC, (const MKL_INT*)buffer_rows_CSC,
                    (const MKL_INT*)buffer_offsets_CSC, (const MKL_INT*)(buffer_offsets_CSC + 1),
                    centers_tr, &n,
                    &beta, buffer_dist_matrix, &n);
                FPgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    m, n, 1,
                    (FPTYPE)1.0, ones_vec, m, centers_l2sq, n,
                    (FPTYPE)1.0, buffer_dist_matrix, n);
                FPgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    m, n, 1,
                    (FPTYPE)1.0, docs_l2sq, m, ones_vec, n,
                    (FPTYPE)1.0, buffer_dist_matrix, n);
                delete[] ones_vec;
                delete[] centers_tr;

                for (size_t i = 0; i < moving_docs[batch].size(); ++i) {
                    const auto doc = moving_docs[batch][i];
                    memcpy(lb_distsq_matrix + (size_t)doc * (size_t)num_centers,
                        buffer_dist_matrix + (size_t)i * (size_t)num_centers, num_centers * sizeof(FPTYPE));
                    closest_center[doc] = FPimin(num_centers, lb_distsq_matrix + (size_t)doc * (size_t)num_centers, 1);
                    ub_distsq[doc] = lb_distsq_matrix[(size_t)doc * (size_t)num_centers + closest_center[doc]];
                }
                delete[] buffer_vals_CSC, buffer_rows_CSC, buffer_offsets_CSC, buffer_dist_matrix;
            }
            timer.next_time_secs("Elkan: Distances of moving docs", 30);


            // 4. Let m(c) be the mean of points assigned to center c.
            for (doc_id_t c = 0; c < num_centers; ++c)
                closest_docs[c].clear();
            for (doc_id_t doc = 0; doc < num_docs(); ++doc)
                closest_docs[closest_center[doc]].push_back(doc);

            auto new_centers = new FPTYPE[(size_t)vocab_size()*(size_t)num_centers];
            FPscal((size_t)vocab_size() * (size_t)num_centers, 0.0, new_centers, 1);
            pfor_dynamic_16(auto c = 0; c < num_centers; ++c) {
                auto div = (FPTYPE)closest_docs[c].size();
                for (auto diter = closest_docs[c].begin(); diter != closest_docs[c].end(); ++diter)
                    for (auto witer = offset_CSC(*diter); witer < offset_CSC(1 + *diter); ++witer)
                        *(new_centers + (size_t)c * (size_t)vocab_size() + row_CSC(witer)) += val_CSC(witer) / div;
            }
            timer.next_time_secs("Elkan: New centers", 30);

            // 5. Assign l(x, c) = max{ l(x, c) - d(c, m(c)), 0 } for all x, c
            // 6. For all x, assign: u(x) = u(x) + d(m(c(x)), c(x)).
            std::vector<FPTYPE> distsq_center_mv(num_centers);
            FPaxpy((size_t)vocab_size() * (size_t)num_centers, -1.0, new_centers, 1, centers, 1);
            pfor_dynamic_16(auto c = 0; c < num_centers; ++c)
                distsq_center_mv[c] = FPdot(vocab_size(),
                    centers + (size_t)c * (size_t)vocab_size(), 1,
                    centers + (size_t)c * (size_t)vocab_size(), 1);

            pfor_dynamic_1(auto batch = 0; batch < num_doc_batches; ++batch)
                for (auto diter = moving_docs[batch].begin(); diter != moving_docs[batch].end(); ++diter) {
                    for (auto c = 0; c < num_centers; ++c) {
                        if (lb_distsq_matrix[(*diter) * (size_t)num_centers + (size_t)c] < distsq_center_mv[c])
                            lb_distsq_matrix[(*diter) * (size_t)num_centers + (size_t)c] = 0.0;
                        else
                            lb_distsq_matrix[(*diter) * (size_t)num_centers + (size_t)c]
                            = std::pow(std::sqrt(lb_distsq_matrix[(*diter) * (size_t)num_centers + (size_t)c])
                                - std::sqrt(distsq_center_mv[c]), 2);
                    }
                    ub_distsq[*diter] = std::pow(std::sqrt(ub_distsq[*diter])
                        + std::sqrt(distsq_center_mv[closest_center[*diter]]), 2);
                }

            // 7. Replace c by m(c).
            //FPblascopy((size_t)num_centers* (size_t)vocab_size(), new_centers, 1, centers, 1);
            memcpy(centers, new_centers, sizeof(FPTYPE) * (size_t)num_centers* (size_t)vocab_size());
            delete[] new_centers;
            delete[] moving_docs;
            delete[] ones_vec;
            timer.next_time_secs("Elkan: Update dist bounds", 30);
        }

        delete[] closest_center;
        delete[] lb_distsq_matrix;
        delete[] ub_distsq;
        delete[] docs_l2sq;
        delete[] centers_l2sq;
        if (!return_clusters)
            delete[] closest_docs;
        return residual;
    }
}

template class ISLE::SparseMatrix<float>;
template class ISLE::FPSparseMatrix<float>;

template void ISLE::FPSparseMatrix<float>::threshold_and_copy(
    const SparseMatrix<float>& from,
    const std::vector<float>& zetas,
    const offset_t nnzs,
    std::vector<doc_id_t>& original_cols
);

template void ISLE::FPSparseMatrix<float>::sampled_threshold_and_copy(
    const SparseMatrix<float>& from,
    const std::vector<float>& zetas,
    const offset_t nnzs,
    std::vector<doc_id_t>& original_cols,
    const FPTYPE sample_rate
);
