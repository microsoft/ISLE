// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "trainer.h"

using namespace ISLE;

int main(int argv, char **argc) {
    if (argv != 10) {
        std::cout << "Incorrect usage of ISLETrain. Use: \n"
            << "trainFromFile <tdf_file> <vocab_file> <output_dir> "
            << "<vocab_size> <num_docs> <max_entries> <num_topics> "
            << "<sample(0/1)> <sample_rate>" << std::endl;
        exit(-1);
    }

    const std::string tdf_file = argc[1];
    const std::string vocab_file = argc[2]; 
    const std::string output_dir = argc[3];

    const word_id_t	vocab_size = atol(argc[4]);
    const doc_id_t	num_docs = atol(argc[5]);
    const offset_t	max_entries = atol(argc[6]);
    const doc_id_t	num_topics = atol(argc[7]);

    const bool sample = atoi(argc[8]);
    const FPTYPE sample_rate = atoi(argc[9]);

    try {
        ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
            sample, sample_rate, ISLE::ISLETrainer::data_ingest::FILE_DATA_LOAD,
            tdf_file, vocab_file, output_dir);
        trainer.train();
    }
    catch (...) {
        std::cerr << "ISLE Trainer failed" << std::endl;
    }
}
