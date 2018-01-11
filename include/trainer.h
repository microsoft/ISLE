// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <iostream>
#include <algorithm>
#include <string>
#include <cassert>

#include <mkl.h>

#include "Eigen/Core"
#include "Eigen/SparseCore"
#include "spectra-master/include/SymEigsSolver.h"
//#include "spectra-master/include/MatOp/DenseGenMatProd.h"

#ifdef _MSC_VER
#include <Windows.h>
#endif

#include "types.h"
#include "timer.h"
#include "logger.h"
#include "utils.h"
#include "hyperparams.h"
#include "denseMatrix.h"
#include "sparseMatrix.h"

namespace ISLE
{
    //
    // ISLETrainer class
    //
    class ISLETrainer
    {
        word_id_t			vocab_size;
        doc_id_t			num_docs;
        const offset_t		max_entries;
        const doc_id_t		num_topics;
        const std::string	input_file;
        const std::string	vocab_file;
        const std::string	output_path_base;
        const bool	        flag_sample_docs;
        const FPTYPE	    sample_rate;
        const bool	        flag_compute_log_combinatorial;
        const bool	        flag_compute_distinct_top_five_sets;
        const bool	        flag_compute_avg_coherence;
        const bool	        flag_construct_edge_topics;
        const int	        max_edge_topics;
        const bool	        flag_print_doctopic;
        const bool          flag_print_top_two_topics;

        std::string			log_dir;
#ifdef _MSC_VER
        std::wstring		log_dir_wstr;
        LPCWSTR				log_dir_lpcwstr;
#endif
#ifdef LINUX
        std::string         log_dir_wstr;
        char*               log_dir_lpcwstr;
#endif

        Logger*				out_log;
        Timer*				timer;

        bool				is_data_loaded;
        bool				is_training_complete;
        int					how_data_loaded;

        std::vector<DocWordEntry<count_t> > entries;

        SparseMatrix<A_TYPE>* A_sp;
        FloatingPointSparseMatrix<FPTYPE> *B_fl_CSC;

        std::vector<std::string> vocab_words;

        std::vector<word_id_t>* catchwords;
        std::vector<std::pair<word_id_t, FPTYPE> >* topwords;

        DenseMatrix<FPTYPE>* Model;
        DenseMatrix<FPTYPE>* EdgeModel;

        std::vector<doc_id_t> *closest_docs;
        A_TYPE* catchword_thresholds;
        FPTYPE* centers;

    public:
        enum data_ingest { FILE_DATA_LOAD, ITERATIVE_DATA_LOAD };

        ISLETrainer(
            const word_id_t		vocab_size_,        // If vocab_size_==0, #words initialized from dataset, keep below 2^31
            const doc_id_t		num_docs_,          // If num_docs_==0,   #docs initialized from dataset, keep below 2^31
            const offset_t		max_entries_,       // Maximum number of non-zero doc-word counts. If 0, read unlimited #entries
            const doc_id_t		num_topics_,        // Number of topics you want to extract
            const bool			sample_docs_,       // Pick a subset of documents for SVD after thresholding
            const FPTYPE		sample_rate_,       // If sampling, specify sample rate in (0, 1]
            const data_ingest	how_data_loaded_,   // Has data been loaded from file or fed point by point
            const std::string&	input_file_ = std::string(""),   // This file contains one triple <docid> <wordid> <freq> per line, upto max_entries line (1-based numbering)
            const std::string&	vocab_file_ = std::string(""),   // This file contains word names for each wordid. Line i lists the full string for wordid i.
            const std::string&	output_path_base_ = std::string(""), // Logs and output are written in a subdir under this directory
            const bool	construct_edge_topics_ = false,          // Construct edge topics (aka compound topics) from simple topics
            const int	max_edge_topics_ = 100000,           // Maximum number of edge topics
            const bool	compute_log_combinatorial_ = false,      // Compute log(#TotalWordsInDoc)/\prod_{word \in doc}log(freq(word)) for each doc in input
            const bool	compute_distinct_top_five_sets_ = false, // Compute how many distinct quintuplets of words occur as top five in a doc
            const bool	compute_avg_coherence_ = false,          // Compute the average coherence of the topics, relevant constants in hyperparams.h
            const bool	print_doctopic_ = false,                 // How many of each topic's catchwords are in a doc?
            const bool  print_top_two_topics_ = true);

        ~ISLETrainer();


        //
        // Load a vocabulary list from file
        // Load a sparse data matrix from file
        //
        void load_data_from_file();

        //
        // Feed a document, i.e., list of words and their counts
        //
        inline void feed_data(
            const doc_id_t doc,
            const word_id_t *const words,
            const count_t *const counts,
            const offset_t num_words);

        //
        // Call when calls to feed_data are done
        //
        void finalize_data();

        //
        // Create a file which computes the combinatorial function of word counts in each doc
        //
        void print_log_combinatorial();

        //
        // Create a multiset of top 5 words for each doc
        // Count how many of these multi-instances occur more than x times
        //
        void print_distinct_top_five_sets();

        //
        // Computing singular vals of A_sp
        //
        void compute_input_svd();

        //
        // Build the basic (and edge) models
        //
        void train();

        //
        // Calculate coherence based on cluster averages, without using catchwords
        //	
        void output_avg_topic_coherence(
            FPTYPE& avg_nl_coherence,
            std::vector<FPTYPE>& nl_coherences);

        //
        // Calculate Topic Diversity
        //
        void output_topic_diversity();

        //
        // Output Catchwords and Dominant Words for each topic
        //
        void output_cluster_summary(
            const std::vector<FPTYPE>& coherences,
            const FPTYPE& avg_coherence,
            const std::vector<FPTYPE>& nl_coherences,
            const FPTYPE& avg_nl_coherence,
            const FloatingPointSparseMatrix<FPTYPE> *const A_sp);

        //
        // Output Model to file
        //
        void output_model(bool output_sparse = false);

        //
        // Output EdgeModel to file
        //
        void output_edge_model(bool output_sparse = false);

        //
        // Ouput Top Words to File
        //
        void output_top_words();

        //
        // Output document catchword frequencies in 1-based index
        // Output doc-topic catchword sums in 1-based index
        //
        void output_doc_topic(
            std::vector<std::pair<word_id_t, int> >& catchword_topics,
            std::vector<std::tuple<doc_id_t, doc_id_t,
            FPTYPE> >& doc_topic_sum);

        void get_basic_model(FPTYPE *const basicModel);

        // 
        // The number of edge topics; the threshold for minimum #samples to define an edge topic is in hyperparams.h
        //
        int get_num_edge_topics();

        //
        // The final edge model
        //
        void get_edge_model(FPTYPE *const edgeModel);

        //
        // Find Top Two topics for each doc and print them to file
        //
        void print_top_two_topics(
            std::vector<std::tuple<int, int, doc_id_t> >& top_topic_pairs);

        //
        // Compute edge topics based on the list of top-2 topics for each file
        //
        void construct_edge_topics_v1(
            std::vector<std::tuple<int, int, doc_id_t> >& top_topic_pairs,
            bool flag_print_edge_topic_composition = true);

        // 
        // Compute edge topics based on the list of top-2 topics for each file
        // Edge topics are linear combinations of pairs of topics
        //
        void construct_edge_topics_v2(
            std::vector<std::tuple<int, int, doc_id_t> >& top_topic_pairs,
            bool flag_print_edge_topic_composition=true);

        //
        // Print the basic topics that made up the edge topic
        // Line i (i-th edge topic): <primary_topic> <secondary_topic> <doc_count>
        //
        void print_edge_topic_composition(
            std::vector<std::tuple<int, int, count_t> >& edge_topics);
    };
}
