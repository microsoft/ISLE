// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "include/trainer.h"

#ifdef _MSC_VER
#define EXPORT_API(ret) extern "C" __declspec(dllexport) ret __stdcall
#else
#define EXPORT_API(ret) extern "C" __attribute__((visisbility("default"))) ret
#endif

//
// Question: Are doc and word IDs 0-based or 1-based? ISLE assumes 0-base for everything.
//
// Assumption: trained model are in 4-byte floating point
//
// Call in the following order:
//	1. CreateTrainer(), feedData(), finalizeData(), Train()
//  2. pre-allocate sufficient space and then call getModel() to get basic model
//  3. GetNumEdgeTopics(), pre-allocate sufficient space and get EdgeModel to get Edge model
//	4. Other statistics to be added....
//  5. DestrotTrainer
//
//  ToDo:
//	1. LogUtils callback
//	2. Other statistics
//

namespace ISLE
{
    EXPORT_API(ISLETrainer*) CreateTrainer(
        const vocabSz_t		vocab_size_,
        const docsSz_t		num_docs_,
        const offset_t		max_entries_,
        const docsSz_t		num_topics_,
        const bool			sample_docs_ = false,
        const FPTYPE		sample_rate_ = -1.0)
    {
        return new ISLETrainer(vocab_size_, num_docs_, max_entries_, num_topics_,
            sample_docs_, sample_rate_, ISLETrainer::data_ingest::ITERATIVE_DATA_LOAD);
    }

    EXPORT_API(void) DestroyTrainer(ISLETrainer* trainer)
    {
        delete trainer;
    }

    EXPORT_API(void) feedData(ISLETrainer* trainer,
        const docsSz_t doc, const vocabSz_t *const words, const count_t *const counts, const offset_t num_words)
    {
        trainer->feed_data(doc, words, counts, num_words);
    }

    EXPORT_API(void) finalizeData(ISLETrainer* trainer)
    {
        trainer->finalize_data();
    }

    EXPORT_API(void) Train(ISLETrainer* trainer)
    {
        trainer->train();
    }

    //
    // Copies the basic topic model to 'basicModel'
    // basicModel must be pre-allocated to size trainer->num_topics * trainer->vocab_size
    //		It consists of 'num_topics' probability distributions over the vocabulary
    //		basicModel[topic*vocab_size + word] provides the weight of 'word' in 'topic'.
    //
    EXPORT_API(void) GetBasicModel(ISLETrainer* trainer, FPTYPE* basicModel)
    {
        trainer->get_basic_model(basicModel);
    }

    //
    // Returns number of EdgeTopics discovered
    //
    EXPORT_API(int) GetNumEdgeTopics(ISLETrainer* trainer)
    {
        return trainer->get_num_edge_topics();
    }

    //
    // Copies the edge topic model to 'edgeModel'
    // edgeModel must be pre-allocated to size GetNumEdgeTopics() * trainer->vocab_size
    //		It consists of 'GetNumEdgeTopics()' probability distributions over the vocabulary
    //		edgeModel[topic*vocab_size + word] provides the weight of 'word' in 'topic'.
    //
    EXPORT_API(void) GetEdgeModel(ISLETrainer* trainer, FPTYPE* edgeModel)
    {
        trainer->get_edge_model(edgeModel);
    }

    EXPORT_API(void) GetTopicsCatchWords(ISLETrainer* trainer,
        const int topic, vocabSz_t *const catchwods, int& numCatchwodsForTopic)
    {

    }
}
