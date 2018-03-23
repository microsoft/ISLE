// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "infer.h"

/*const std::string dir
= "C:\\Users\\HARSHASI\\Source\\Repos\\ISLE\\ISLE\\data\\NYTimes_Vocab5k_TrainDocs30k";
const std::string infer_file
= "TrainData.VocabIndex1.DocIndex1.txt";
const std::string model_file
= "log_t_50_eps1_0.016667_eps2_0.333333_eps3_5.000000_kMppReps_1_kMLowDReps_10_kMReps_10_sample_0\\M_hat_catch";
const std::string sparse_model_file
= "log_t_50_eps1_0.016667_eps2_0.333333_eps3_5.000000_kMppReps_1_kMLowDReps_10_kMReps_10_sample_0\\M_hat_catch_sparse";
const offset_t M_hat_catch_sparse_entries = 248029;
const doc_id_t num_topics = 50;
const word_id_t vocab_size = 5000;
const doc_id_t num_docs = 30000;
const offset_t max_entries = 4886003;*/

//const std::string dir
//    = "C:\\Users\\HARSHASI\\Source\\Repos\\ISLE\\ISLE\\data\\NYTimes_Vocab5004_TrainDocs296784";
//const std::string infer_file
//    = "docword.nytimes.proc.txt";
//const std::string model_file
//    = "log_t_100_eps1_0.016667_eps2_0.333333_eps3_5.000000_kMppReps_4_kMLowDReps_30_kMReps_10_sample_0\\M_hat_catch";
//const doc_id_t num_topics = 100;
//const word_id_t vocab_size = 5004;
//const doc_id_t num_docs = 296784;
//const offset_t max_entries = 47978132;

//const std::string dir
//    = "C:\\Users\\HARSHASI\\Desktop\\topic_modeling_data\\NYTimes_New";
//const std::string infer_file
//    = "TrainData.VocabIndex1.tsvd";
//const std::string model_file
//    = "log_t_1000_eps1_0.016667_eps2_0.333333_eps3_5.000000_kMppReps_4_kMLowDReps_30_kMReps_10_sample_0\\M_hat_catch";
//const doc_id_t num_topics = 1000;
//const word_id_t vocab_size = 101504;
//const doc_id_t num_docs = 269714;
//const offset_t max_entries = 57289130;

//#if defined(_MSC_VER)
//const std::string dir
//= "C:\\Users\\HARSHASI\\Desktop\\topic_modeling_data\\DSA_UK_Travel";
//const std::string infer_file
//= "OrganicSearchData\\LPData_Docid_TokenId_TF.txt";
//const std::string model_file
//= "log_t_1000_eps1_0.016667_eps2_0.333333_eps3_5.000000_kMppReps_1_kMLowDReps_10_kMReps_10_sample_1_Rate_0.100000\\M_hat_catch";
//const std::string sparse_model_file
//= "log_t_1000_eps1_0.016667_eps2_0.333333_eps3_5.000000_kMppReps_1_kMLowDReps_10_kMReps_10_sample_1_Rate_0.100000\\M_hat_catch_sparse";
//#elif defined(LINUX)
//const std::string dir
//= "/mnt/DSA_UK_Travel";
//const std::string infer_file
//= "OrganicSearchData/LPData_Docid_TokenId_TF.txt";
//const std::string model_file
//= "log_t_1000_eps1_0.016667_eps2_0.333333_eps3_5.000000_kMppReps_1_kMLowDReps_10_kMReps_10_sample_1_Rate_0.100000/M_hat_catch";
//const std::string sparse_model_file
//= "log_t_1000_eps1_0.016667_eps2_0.333333_eps3_5.000000_kMppReps_1_kMLowDReps_10_kMReps_10_sample_1_Rate_0.100000/M_hat_catch_sparse";
//#endif
//const doc_id_t num_topics = 1000;
//const word_id_t vocab_size = 1562308;
//const doc_id_t num_docs = 7415;
//const offset_t max_entries = 4594436;
////const offset_t M_hat_catch_sparse_entries = 60447915; // 1000 topics
//const offset_t M_hat_catch_sparse_entries = 73842125; // 2000 topics
