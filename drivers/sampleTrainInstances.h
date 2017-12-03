// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

//try { // NIPS
//	vocabSz_t	vocab_size = 5002;
//	docsSz_t	num_docs = 1491;
//	offset_t	max_entries = 639743;
//	docsSz_t	num_topics = 15;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		false, -1.0, ISLE::ISLETrainer::data_ingest::FILE_DATA_LOAD,
//		std::string("data\\nips\\docword.nips.proc.txt"),
//		std::string("data\\nips\\docword.nips.vocab.trunc.txt"),
//		std::string("data\\nips"));
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "NIPS data failed" << std::endl;
//}
//try { // Cosomos
//	vocabSz_t	vocab_size = 1058546;
//	docsSz_t	num_docs = 4026;
//	offset_t	max_entries = 8976858;
//	docsSz_t	num_topics = 15;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		false, -1.0, ISLE::ISLETrainer::data_ingest::FILE_DATA_LOAD,
//		std::string("data\\sruti\\user_documents_NMF_format_2017-05-26-0h_to_2017-06-14-23h.txt"),
//		std::string("data\\sruti\\all_path_tokens_NMF_format_2017-05-26-0h_to_2017-06-14-23h.txt"),
//		std::string("data\\sruti\\output"));
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "NIPS data failed" << std::endl;
//}
//try { // Enron
//	vocabSz_t	vocab_size = 5003;
//	docsSz_t	num_docs = 29823;
//	offset_t	max_entries = 2722177;
//	docsSz_t	num_topics = 20;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("data\\enron\\docword.enron.proc.txt"),
//		std::string("data\\enron\\docword.enron.vocab.trunc.txt"),
//		std::string("data\\enron")); 
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "Enron data failed" << std::endl;
//}
//    try { // nytimes 30k
//        vocabSz_t	vocab_size = 5000;
//        docsSz_t	num_docs = 30000;
//        offset_t	max_entries = 4886003;
//        docsSz_t	num_topics = 50;
//        ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//            false, -1.0, ISLE::ISLETrainer::data_ingest::FILE_DATA_LOAD,
//#if defined(_MSC_VER)
//            std::string("data\\nytimes_vocab5k_traindocs30k\\traindata.vocabindex1.docindex1.txt"),
//            std::string("data\\nytimes_vocab5k_traindocs30k\\vocab.txt"),
//            std::string("data\\nytimes_vocab5k_traindocs30k"));
//#elif defined(LINUX)
//            std::string("data/NYTimes_Vocab5k_TrainDocs30k/TrainData.VocabIndex1.DocIndex1.txt"),
//            std::string("data/NYTimes_Vocab5k_TrainDocs30k/vocab.txt"),
//            std::string("data/NYTimes_Vocab5k_TrainDocs30k"));
//#else
//            assert(false);
//#endif
//        trainer.train();
//    }
//    catch (...) {
//        std::cerr << "PubMed 500K failed" << std::endl;
//    }
/*try { // NYTimes ~300K
vocabSz_t	vocab_size = 5004;
docsSz_t	num_docs = 296784;
offset_t	max_entries = 47978132;
docsSz_t	num_topics = 100;
ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
true, 0.1, ISLE::ISLETrainer::data_ingest::FILE_DATA_LOAD,
#if defined(_MSC_VER)
std::string("data\\NYTimes_Vocab5004_TrainDocs296784\\docword.nytimes.proc.txt"),
std::string("data\\NYTimes_Vocab5004_TrainDocs296784\\docword.nytimes.vocab.trunc.txt"),
std::string("data\\NYTimes_Vocab5004_TrainDocs296784"));
#elif defined(LINUX)
std::string("data/NYTimes_Vocab5004_TrainDocs296784/docword.nytimes.proc.txt"),
std::string("data/NYTimes_Vocab5004_TrainDocs296784/docword.nytimes.vocab.trunc.txt"),
std::string("data/NYTimes_Vocab5004_TrainDocs296784"),
false, false, false, false, 0, false);
#else
assert(false);
#endif
trainer.train();
}
catch (...) {
std::cerr << "NYTimes failed" << std::endl;
}
try { // NYTimes ~300K
vocabSz_t	vocab_size = 5004;
docsSz_t	num_docs = 296784;
offset_t	max_entries = 47978132;
docsSz_t	num_topics = 1000;
ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
true, 0.1, ISLE::ISLETrainer::data_ingest::FILE_DATA_LOAD,
#if defined(_MSC_VER)
std::string("data\\NYTimes_Vocab5004_TrainDocs296784\\docword.nytimes.proc.txt"),
std::string("data\\NYTimes_Vocab5004_TrainDocs296784\\docword.nytimes.vocab.trunc.txt"),
std::string("data\\NYTimes_Vocab5004_TrainDocs296784"));
#elif defined(LINUX)
std::string("data/NYTimes_Vocab5004_TrainDocs296784/docword.nytimes.proc.txt"),
std::string("data/NYTimes_Vocab5004_TrainDocs296784/docword.nytimes.vocab.trunc.txt"),
std::string("data/NYTimes_Vocab5004_TrainDocs296784"),
false, false, false, false, 0, false);
#else
assert(false);
#endif
trainer.train();
}
catch (...) {
std::cerr << "NYTimes failed" << std::endl;
}
try { // NYTimes ~300K
vocabSz_t	vocab_size = 5004;
docsSz_t	num_docs = 296784;
offset_t	max_entries = 47978132;
docsSz_t	num_topics = 2000;
ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
true, 0.1, ISLE::ISLETrainer::data_ingest::FILE_DATA_LOAD,
#if defined(_MSC_VER)
std::string("data\\NYTimes_Vocab5004_TrainDocs296784\\docword.nytimes.proc.txt"),
std::string("data\\NYTimes_Vocab5004_TrainDocs296784\\docword.nytimes.vocab.trunc.txt"),
std::string("data\\NYTimes_Vocab5004_TrainDocs296784"));
#elif defined(LINUX)
std::string("data/NYTimes_Vocab5004_TrainDocs296784/docword.nytimes.proc.txt"),
std::string("data/NYTimes_Vocab5004_TrainDocs296784/docword.nytimes.vocab.trunc.txt"),
std::string("data/NYTimes_Vocab5004_TrainDocs296784"),
false, false, false, false, 0, false);
#else
assert(false);
#endif
trainer.train();
}
catch (...) {
std::cerr << "NYTimes failed" << std::endl;
}

try { // NYTimes ~300K
vocabSz_t	vocab_size = 5004;
docsSz_t	num_docs = 296784;
offset_t	max_entries = 47978132;
docsSz_t	num_topics = 100;
ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
false, 1.0, ISLE::ISLETrainer::data_ingest::FILE_DATA_LOAD,
#if defined(_MSC_VER)
std::string("data\\NYTimes_Vocab5004_TrainDocs296784\\docword.nytimes.proc.txt"),
std::string("data\\NYTimes_Vocab5004_TrainDocs296784\\docword.nytimes.vocab.trunc.txt"),
std::string("data\\NYTimes_Vocab5004_TrainDocs296784"));
#elif defined(LINUX)
std::string("data/NYTimes_Vocab5004_TrainDocs296784/docword.nytimes.proc.txt"),
std::string("data/NYTimes_Vocab5004_TrainDocs296784/docword.nytimes.vocab.trunc.txt"),
std::string("data/NYTimes_Vocab5004_TrainDocs296784"),
false, false, false, false, 0, false);
#else
assert(false);
#endif
trainer.train();
}
catch (...) {
std::cerr << "NYTimes failed" << std::endl;
}
try { // NYTimes ~300K
vocabSz_t	vocab_size = 5004;
docsSz_t	num_docs = 296784;
offset_t	max_entries = 47978132;
docsSz_t	num_topics = 1000;
ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
false, -1.0, ISLE::ISLETrainer::data_ingest::FILE_DATA_LOAD,
#if defined(_MSC_VER)
std::string("data\\NYTimes_Vocab5004_TrainDocs296784\\docword.nytimes.proc.txt"),
std::string("data\\NYTimes_Vocab5004_TrainDocs296784\\docword.nytimes.vocab.trunc.txt"),
std::string("data\\NYTimes_Vocab5004_TrainDocs296784"));
#elif defined(LINUX)
std::string("data/NYTimes_Vocab5004_TrainDocs296784/docword.nytimes.proc.txt"),
std::string("data/NYTimes_Vocab5004_TrainDocs296784/docword.nytimes.vocab.trunc.txt"),
std::string("data/NYTimes_Vocab5004_TrainDocs296784"),
false, false, false, false, 0, false);
#else
assert(false);
#endif
trainer.train();
}
catch (...) {
std::cerr << "NYTimes failed" << std::endl;
}
try { // NYTimes ~300K
vocabSz_t	vocab_size = 5004;
docsSz_t	num_docs = 296784;
offset_t	max_entries = 47978132;
docsSz_t	num_topics = 2000;
ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
false, -1.0, ISLE::ISLETrainer::data_ingest::FILE_DATA_LOAD,
#if defined(_MSC_VER)
std::string("data\\NYTimes_Vocab5004_TrainDocs296784\\docword.nytimes.proc.txt"),
std::string("data\\NYTimes_Vocab5004_TrainDocs296784\\docword.nytimes.vocab.trunc.txt"),
std::string("data\\NYTimes_Vocab5004_TrainDocs296784"));
#elif defined(LINUX)
std::string("data/NYTimes_Vocab5004_TrainDocs296784/docword.nytimes.proc.txt"),
std::string("data/NYTimes_Vocab5004_TrainDocs296784/docword.nytimes.vocab.trunc.txt"),
std::string("data/NYTimes_Vocab5004_TrainDocs296784"),
false, false, false, false, 0, false);
#else
assert(false);
#endif
trainer.train();
}
catch (...) {
std::cerr << "NYTimes failed" << std::endl;
}*/

//try { // PubMed 30K
//	vocabSz_t	vocab_size = 5000;
//	docsSz_t	num_docs = 30000;
//	offset_t	max_entries = 1308971;
//	docsSz_t	num_topics = 50;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("data\\Pubmed_Vocab5k_TrainDocs30k\\TrainData.VocabIndex1.DocIndex1.txt"),
//		std::string("data\\Pubmed_Vocab5k_TrainDocs30k\\vocab.txt"),
//		std::string("data\\Pubmed_Vocab5k_TrainDocs30k"));
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "PubMed 30K failed" << std::endl;
//}
//try { // PubMed 500K
//	vocabSz_t	vocab_size = 5000;
//	docsSz_t	num_docs = 500000;
//	offset_t	max_entries = 21881399;
//	docsSz_t	num_topics = 100;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("data\\Pubmed_Vocab5k_TrainDocs500k\\TrainData.VocabIndex1.tsvd"),
//		std::string("data\\Pubmed_Vocab5k_TrainDocs500k\\vocab.tsvd"),
//		std::string("data\\Pubmed_Vocab5k_TrainDocs500k")); 
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "PubMed 500K failed" << std::endl;
//}
//try { // PubMed 30K; Full Vocab
//	vocabSz_t	vocab_size = 140762;
//	docsSz_t	num_docs = 30000;
//	offset_t	max_entries = 1430182;
//	docsSz_t	num_topics = 50;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("data\\Pudmed_Vocab140762_TrainDocs30k\\TrainData30K.VocabIndex1.tsvd"),
//		std::string("data\\Pudmed_Vocab140762_TrainDocs30k\\PubMedFullVocab.txt"),
//		std::string("data\\Pudmed_Vocab140762_TrainDocs30k")); 
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "PubMed 30K failed" << std::endl;
//}
//try { // PubMed 500K; Full Vocab
//	vocabSz_t	vocab_size = 140762;
//	docsSz_t	num_docs = 500000;
//	offset_t	max_entries = 22666049;
//	docsSz_t	num_topics = 100;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("data\\Pudmed_Vocab140762_TrainDocs500k\\TrainData500K.VocabIndex1.tsvd"),
//		std::string("data\\Pudmed_Vocab140762_TrainDocs500k\\PubMedFullVocab.txt"),
//		std::string("data\\Pudmed_Vocab140762_TrainDocs500k"));
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "PubMed 500K failed" << std::endl;
//}
//try { // Bing Ads small
//	vocabSz_t	vocab_size = 10000;
//	docsSz_t	num_docs = 1000000;
//	offset_t	max_entries = -1;
//	docsSz_t	num_topics = 100;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("C:\\Users\\HARSHASI\\Desktop\\1M-10K"),
//		std::string("C:\\Users\\HARSHASI\\Desktop\\Vocab_trunc.tsv"),
//		std::string("C:\\Users\\HARSHASI\\Desktop\\")); 
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "PubMed 500K failed" << std::endl;
//}
//try {
//	vocabSz_t	vocab_size = 5000;
//	docsSz_t	num_docs = 7482201;
//	offset_t	max_entries = 327373030;
//	docsSz_t	num_topics = 100;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("C:\\Users\\HARSHASI\\Desktop\\Pubmed_Vocab5k_7.7MTrainDocs\\TrainData.VocabIndex1.tsvd"),
//		std::string("C:\\Users\\HARSHASI\\Desktop\\Pubmed_Vocab5k_7.7MTrainDocs\\Vocab.tsvd"),
//		std::string("C:\\Users\\HARSHASI\\Desktop\\Pubmed_Vocab5k_7.7MTrainDocs")); 
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "Pubmed 7.7M failed" << std::endl;
//}
//try {
//	vocabSz_t	vocab_size = 5000;
//	docsSz_t	num_docs = 7482201;
//	offset_t	max_entries = 327373030;
//	docsSz_t	num_topics = 500;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("C:\\Users\\HARSHASI\\Desktop\\Pubmed_Vocab5k_7.7MTrainDocs\\TrainData.VocabIndex1.tsvd"),
//		std::string("C:\\Users\\HARSHASI\\Desktop\\Pubmed_Vocab5k_7.7MTrainDocs\\Vocab.tsvd"),
//		std::string("C:\\Users\\HARSHASI\\Desktop\\Pubmed_Vocab5k_7.7MTrainDocs")); 
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "Pubmed 7.7M failed" << std::endl;
//}
//try {
//	vocabSz_t	vocab_size = 140762;
//	docsSz_t	num_docs = 8000000;
//	offset_t	max_entries = 471652462;
//	docsSz_t	num_topics = 100;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("C:\\Users\\harshasi\\Desktop\\Pubmed_Vocab140762_8MTrainDocs\\TrainData.VocabIndex1.tsvd"),
//		std::string("C:\\Users\\harshasi\\Desktop\\Pubmed_Vocab140762_8MTrainDocs\\PubMedFullVocab.txt"),
//		std::string("C:\\Users\\harshasi\\Desktop\\Pubmed_Vocab140762_8MTrainDocs"));
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "Pubmed 7.7M failed" << std::endl;
//}
//try {
//	vocabSz_t	vocab_size = 140762;
//	docsSz_t	num_docs = 8000000;
//	offset_t	max_entries = 471652462;
//	docsSz_t	num_topics = 500;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("C:\\Users\\harshasi\\Desktop\\Pubmed_Vocab140762_8MTrainDocs\\TrainData.VocabIndex1.tsvd"),
//		std::string("C:\\Users\\harshasi\\Desktop\\Pubmed_Vocab140762_8MTrainDocs\\PubMedFullVocab.txt"),
//		std::string("C:\\Users\\harshasi\\Desktop\\Pubmed_Vocab140762_8MTrainDocs"));
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "Pubmed 7.7M failed" << std::endl;
//}
//try {
//	vocabSz_t	vocab_size = 5000;
//	docsSz_t	num_docs = 100000;
//	offset_t	max_entries = 4377699;
//	docsSz_t	num_topics = 100;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("data\\Pubmed_100K_samples1\\samples.tsvd"),
//		std::string("data\\Pubmed_100K_samples1\\Vocab.tsvd"),
//		std::string("data\\Pubmed_100K_samples1")); 
//	trainer.train();
//}
//catch (...) {
//std::cerr << "Pubmed 100K(1) of 7.7M failed" << std::endl;
//}
//try {
//	vocabSz_t	vocab_size = 5000;
//	docsSz_t	num_docs = 100000;
//	offset_t	max_entries = 4378227;
//	docsSz_t	num_topics = 100;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("data\\Pubmed_100K_samples2\\samples.tsvd"),
//		std::string("data\\Pubmed_100K_samples2\\Vocab.tsvd"),
//		std::string("data\\Pubmed_100K_samples2")); 
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "Pubmed 100K(1) of 7.7M failed" << std::endl;
//}
//    try { // Pubmed New
//        vocabSz_t	vocab_size = 140577;
//        docsSz_t	num_docs = 8150000;
//        offset_t	max_entries = 428645007;
//        docsSz_t	num_topics = 10;
//        ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//            false, 1.0, ISLE::ISLETrainer::data_ingest::FILE_DATA_LOAD,
//#if defined(_MSC_VER)
//            std::string("C:\\Users\\harshasi\\Desktop\\Pubmed_New\\TrainData.VocabIndex1.tsvd"),
//            std::string("C:\\Users\\harshasi\\Desktop\\Pubmed_New\\Vocab.txt"),
//            std::string("C:\\Users\\harshasi\\Desktop\\Pubmed_New\\"));
//#elif defined(LINUX)
//            std::string("/mnt/Pubmed_New/TrainData.VocabIndex1.tsvd"),
//            std::string("/mnt/Pubmed_New/Vocab.txt"),
//            std::string("/mnt/Pubmed_New/"));
//
//#endif
//        trainer.train();
//    }
//    catch (...) {
//        std::cerr << "Pubmed New failed" << std::endl;
//    }
//try { // Product Ads 500K docs, ~300K vocab., 100 topics.
//	vocabSz_t	vocab_size = 299950;
//	docsSz_t	num_docs = 500000;
//	offset_t	max_entries = 19781349;
//	docsSz_t	num_topics = 100;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("C:\\Users\\harshasi\\Desktop\\ProductAds\\500K-300K\\TrainData.VocabIndex1.txt"),
//		std::string("C:\\Users\\harshasi\\Desktop\\ProductAds\\500K-300K\\Vocab.tsv"),
//		std::string("C:\\Users\\harshasi\\Desktop\\ProductAds\\500K-300K")); 
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "ProductAds 500K failed" << std::endl;
//}
//try {
//	vocabSz_t	vocab_size = 299950;
//	docsSz_t	num_docs = 500000;
//	offset_t	max_entries = 19781349;
//	docsSz_t	num_topics = 500;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("C:\\Users\\harshasi\\Desktop\\ProductAds\\500K-300K\\TrainData.VocabIndex1.txt"),
//		std::string("C:\\Users\\harshasi\\Desktop\\ProductAds\\500K-300K\\Vocab.tsv"),
//		std::string("C:\\Users\\harshasi\\Desktop\\ProductAds\\500K-300K")); 
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "ProductAds 500K failed" << std::endl;
//}
//try { // Product Ads 10M docs, ~300K vocab., 100 topics.
//	vocabSz_t	vocab_size = 299950;
//	docsSz_t	num_docs = 10000000;
//	offset_t	max_entries = 395382196;
//	docsSz_t	num_topics = 100;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("C:\\Users\\harshasi\\Desktop\\ProductAds\\10M-300K\\TrainData.VocabIndex1.txt"),
//		std::string("C:\\Users\\harshasi\\Desktop\\ProductAds\\10M-300K\\Vocab.tsv"),
//		std::string("C:\\Users\\harshasi\\Desktop\\ProductAds\\10M-300K")); 
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "ProductAds 10M failed" << std::endl;
//}
//try {
//	vocabSz_t	vocab_size = 299950;
//	docsSz_t	num_docs = 10000000;
//	offset_t	max_entries = 395382196;
//	docsSz_t	num_topics = 500;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("C:\\Users\\harshasi\\Desktop\\ProductAds\\10M-300K\\TrainData.VocabIndex1.txt"),
//		std::string("C:\\Users\\harshasi\\Desktop\\ProductAds\\10M-300K\\Vocab.tsv"),
//		std::string("C:\\Users\\harshasi\\Desktop\\ProductAds\\10M-300K")); 
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "ProductAds 10M failed" << std::endl;
//}
//try { // Wikipedia ~1M docs, ~1M words
//	vocabSz_t	vocab_size = 999910;
//	docsSz_t	num_docs = 1001466;
//	offset_t	max_entries = 278126734;
//	docsSz_t	num_topics = 100;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("C:\\Users\\harshasi\\Desktop\\Wikipedia\\1M-1M\\Train.SelectedVocab.Index1.tsvd"),
//		std::string("C:\\Users\\harshasi\\Desktop\\Wikipedia\\1M-1M\\Vocab.txt"),
//		std::string("C:\\Users\\harshasi\\Desktop\\Wikipedia\\1M-1M"),
//		false);
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "Wikipedia 1M-1M failed" << std::endl;
//}
//try { // Wikipedia ~1M docs, ~1M words
//	vocabSz_t	vocab_size = 999910;
//	docsSz_t	num_docs = 1001466;
//	offset_t	max_entries = 278126734;
//	docsSz_t	num_topics = 500;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("C:\\Users\\harshasi\\Desktop\\Wikipedia\\1M-1M\\Train.SelectedVocab.Index1.tsvd"),
//		std::string("C:\\Users\\harshasi\\Desktop\\Wikipedia\\1M-1M\\Vocab.txt"),
//		std::string("C:\\Users\\harshasi\\Desktop\\Wikipedia\\1M-1M"),
//		false);
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "Wikipedia 1M-1M failed" << std::endl;
//}
//try { // Wikipedia ~All docs, ~200K words
//	vocabSz_t	vocab_size = 200000;
//	docsSz_t	num_docs = 11702401;
//	offset_t	max_entries = 1232424700;
//	docsSz_t	num_topics = 100;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("C:\\Users\\harshasi\\Desktop\\Wikipedia\\AllDocs-200K\\Train.SelectedVocab.Index1.tsvd"),
//		std::string("C:\\Users\\harshasi\\Desktop\\Wikipedia\\AllDocs-200K\\Vocab.txt"),
//		std::string("C:\\Users\\harshasi\\Desktop\\Wikipedia\\AllDocs-200K")); 
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "Wikipedia All docs-200K failed" << std::endl;
//}
//try { // Wiki All Docs -- 20K 
//	vocabSz_t	vocab_size = 20000;
//	docsSz_t	num_docs = 11702401;
//	offset_t	max_entries = 863945055;
//	docsSz_t	num_topics = 100;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("C:\\Users\\harshasi\\Desktop\\Wikipedia\\AllDocs-20K\\Train.SelectedVocab.Index1.tsvd"),
//		std::string("C:\\Users\\harshasi\\Desktop\\Wikipedia\\AllDocs-20K\\Vocab.txt"),
//		std::string("C:\\Users\\harshasi\\Desktop\\Wikipedia\\AllDocs-20K")); 
//	trainer.train();
//}
//catch (...)
//{
//	std::cerr << "Wikipedia All docs-20K failed" << std::endl;

//}
//try { // Wikipedia ~All docs, ~200K words
//	vocabSz_t	vocab_size = 200000;
//	docsSz_t	num_docs = 11702401;
//	offset_t	max_entries = 1232424700;
//	docsSz_t	num_topics = 500;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("C:\\Users\\harshasi\\Desktop\\Wikipedia\\AllDocs-200K\\Train.SelectedVocab.Index1.tsvd"),
//		std::string("C:\\Users\\harshasi\\Desktop\\Wikipedia\\AllDocs-200K\\Vocab.txt"),
//		std::string("C:\\Users\\harshasi\\Desktop\\Wikipedia\\AllDocs-200K")); 
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "Wikipedia All docs-200K failed" << std::endl;
//}
//try { // Wikipedia ~All docs, ~1M words
//	vocabSz_t	vocab_size = 999910;
//	docsSz_t	num_docs = 11702401;
//	offset_t	max_entries = 1275104915;
//	docsSz_t	num_topics = 100;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("C:\\Users\\harshasi\\Desktop\\Wikipedia\\AllDocs-1M\\Train.SelectedVocab.Index1.tsvd"),
//		std::string("C:\\Users\\harshasi\\Desktop\\Wikipedia\\AllDocs-1M\\Vocab.txt"),
//		std::string("C:\\Users\\harshasi\\Desktop\\Wikipedia\\AllDocs-1M")); 
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "Wikipedia All docs-1M failed" << std::endl;
//}
//try { // Wikipedia ~All docs, ~1M words
//	vocabSz_t	vocab_size = 999910;
//	docsSz_t	num_docs = 11702401;
//	offset_t	max_entries = 1275104915;
//	docsSz_t	num_topics = 500;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("C:\\Users\\harshasi\\Desktop\\Wikipedia\\AllDocs-1M\\Train.SelectedVocab.Index1.tsvd"),
//		std::string("C:\\Users\\harshasi\\Desktop\\Wikipedia\\AllDocs-1M\\Vocab.txt"),
//		std::string("C:\\Users\\harshasi\\Desktop\\Wikipedia\\AllDocs-1M"));
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "Wikipedia All docs-1M failed" << std::endl;
//}
//try { // NYTimes New ~300K docs, ~100K words
//	vocabSz_t	vocab_size = 101504;
//	docsSz_t	num_docs = 269714;
//	offset_t	max_entries = 57289130;
//	docsSz_t	num_topics = 100;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("C:\\Users\\harshasi\\Desktop\\topic_modeling_data\\NYTimes_New\\TrainData.VocabIndex1.tsvd"),
//		std::string("C:\\Users\\harshasi\\Desktop\\topic_modeling_data\\NYTimes_New\\Vocab.txt"),
//		std::string("C:\\Users\\harshasi\\Desktop\\topic_modeling_data\\NYTimes_New"),
//		false);
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "NYTimes New ~300K docs, ~100K failed" << std::endl;
//}
//try { // Wikipedia NEW ~All docs, ~200K words
//	vocabSz_t	vocab_size = 200000;
//	docsSz_t	num_docs = 11702401;
//	offset_t	max_entries = 981338788;
//	docsSz_t	num_topics = 100;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("C:\\Users\\harshasi\\Desktop\\Wikipedia_New\\AllDocs-200K\\Train.SelectedVocab.Index1.tsvd"),
//		std::string("C:\\Users\\harshasi\\Desktop\\Wikipedia_New\\AllDocs-200K\\Vocab.txt"),
//		std::string("C:\\Users\\harshasi\\Desktop\\Wikipedia_New\\AllDocs-200K"));
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "Wikipedia NEW All docs-200K failed" << std::endl;
//}
/*    try { // Wikipedia NEW truncated docs words between [20,2000], ~200K words
vocabSz_t	vocab_size = 199985;
docsSz_t	num_docs = 11702401;
offset_t	max_entries = 774669204;
docsSz_t	num_topics = 10;
ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
false, 1.0, ISLE::ISLETrainer::data_ingest::FILE_DATA_LOAD,
std::string("/mnt/Wikipedia_New_Truncated_20_2000/Train.SelectedVocab.Index1.tsvd"),
std::string("/mnt/Wikipedia_New_Truncated_20_2000/Vocab.txt"),
std::string("/mnt/Wikipedia_New_Truncated_20_2000"),
false, false, false, false, 0, false);
trainer.train();
}
catch (...) {
std::cerr << "Wikipedia NEW truncated-20-2000 failed" << std::endl;
}*/
/*try { // Wikipedia NEW truncated docs words between [20,2000], ~200K words
vocabSz_t	vocab_size = 199985;
docsSz_t	num_docs = 11702401;
offset_t	max_entries = 774669204;
docsSz_t	num_topics = 100;
ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
true, 0.1, ISLE::ISLETrainer::data_ingest::FILE_DATA_LOAD,
std::string("/mnt/Wikipedia_New_Truncated_20_2000/Train.SelectedVocab.Index1.tsvd"),
std::string("/mnt/Wikipedia_New_Truncated_20_2000/Vocab.txt"),
std::string("/mnt/Wikipedia_New_Truncated_20_2000"),
false, false, false, false, 0, false);
trainer.train();
}
catch (...) {
std::cerr << "Wikipedia NEW truncated-20-2000 failed" << std::endl;
}
try { // Wikipedia NEW truncated docs words between [20,2000], ~200K words
vocabSz_t	vocab_size = 199985;
docsSz_t	num_docs = 11702401;
offset_t	max_entries = 774669204;
docsSz_t	num_topics = 1000;
ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
true, 0.1, ISLE::ISLETrainer::data_ingest::FILE_DATA_LOAD,
std::string("/mnt/Wikipedia_New_Truncated_20_2000/Train.SelectedVocab.Index1.tsvd"),
std::string("/mnt/Wikipedia_New_Truncated_20_2000/Vocab.txt"),
std::string("/mnt/Wikipedia_New_Truncated_20_2000"));
//false, false, false, false, 0, false);
trainer.train();
}
catch (...) {
std::cerr << "Wikipedia NEW truncated-20-2000 failed" << std::endl;
}
try { // Wikipedia NEW truncated docs words between [20,2000], ~200K words
vocabSz_t	vocab_size = 199985;
docsSz_t	num_docs = 11702401;
offset_t	max_entries = 774669204;
docsSz_t	num_topics = 2000;
ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
true, 0.1, ISLE::ISLETrainer::data_ingest::FILE_DATA_LOAD,
std::string("/mnt/Wikipedia_New_Truncated_20_2000/Train.SelectedVocab.Index1.tsvd"),
std::string("/mnt/Wikipedia_New_Truncated_20_2000/Vocab.txt"),
std::string("/mnt/Wikipedia_New_Truncated_20_2000"));
//false, false, false, false, 0, false);
trainer.train();
}
catch (...) {
std::cerr << "Wikipedia NEW truncated-20-2000 failed" << std::endl;
}*/
//try { // Product Ads 100M docs, ~300K vocab., 100 topics.
//	vocabSz_t	vocab_size = 299950;
//	docsSz_t	num_docs = 100000000;
//	offset_t	max_entries = 3953650137;
//	docsSz_t	num_topics = 1000;
//	ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//		std::string("C:\\Users\\harshasi\\Desktop\\ProductAds\\100M-300K\\TrainData.VocabIndex1.txt"),
//		std::string("C:\\Users\\harshasi\\Desktop\\ProductAds\\100M-300K\\Vocabulary.tsv_300k"),
//		std::string("C:\\Users\\harshasi\\Desktop\\ProductAds\\100M-300K"));
//	trainer.train();
//}
//catch (...) {
//	std::cerr << "ProductAds 100M failed" << std::endl;
//}
/*    try { // Product Ads 100M docs, ~300K vocab., 100 topics.
vocabSz_t	vocab_size = 299950;
docsSz_t	num_docs = 0;
offset_t	max_entries = 4084444648;
docsSz_t	num_topics = 350;
ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
true, 0.1, ISLE::ISLETrainer::data_ingest::FILE_DATA_LOAD,
#if defined(_MSC_VER)
std::string("C:\\Users\\harshasi\\Desktop\\ProductAds\\100M-300K\\100M-300K.tsv"),
std::string("C:\\Users\\harshasi\\Desktop\\ProductAds\\100M-300K\\FullVocabulary.tsv"),
std::string("C:\\Users\\harshasi\\Desktop\\ProductAds\\100M-300K"));
#elif defined(LINUX)
std::string("/mnt/ProductAds100M-300K/100M-300K.tsv"),
std::string("/mnt/ProductAds100M-300K/Vocab1M.txt"),
std::string("/mnt/ProductAds100M-300K"),
false, false, false, false, 0, false);
#endif

trainer.train();
}
catch (...) {
std::cerr << "ProductAds 100M failed" << std::endl;
}*/
/*    try { // Product Ads 100M docs, ~300K vocab., 100 topics.
vocabSz_t	vocab_size = 299950;
docsSz_t	num_docs = 0;
offset_t	max_entries = 4084444648;
docsSz_t	num_topics = 100;
ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
true, 0.1, ISLE::ISLETrainer::data_ingest::FILE_DATA_LOAD,
#if defined(_MSC_VER)
std::string("C:\\Users\\harshasi\\Desktop\\ProductAds\\100M-300K\\100M-300K.tsv"),
std::string("C:\\Users\\harshasi\\Desktop\\ProductAds\\100M-300K\\FullVocabulary.tsv"),
std::string("C:\\Users\\harshasi\\Desktop\\ProductAds\\100M-300K"));
#elif defined(LINUX)
std::string("/mnt/ProductAds100M-300K/100M-300K.tsv"),
std::string("/mnt/ProductAds100M-300K/Vocab1M.txt"),
std::string("/mnt/ProductAds100M-300K"));
#endif

trainer.train();
}
catch (...) {
std::cerr << "ProductAds 100M failed" << std::endl;
} */
/*    try { // Product Ads 100M docs, ~300K vocab., 100 topics.
vocabSz_t	vocab_size = 299950;
docsSz_t	num_docs = 0;
offset_t	max_entries = 4084444648;
docsSz_t	num_topics = 1000;
ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
true, 0.1, ISLE::ISLETrainer::data_ingest::FILE_DATA_LOAD,
#if defined(_MSC_VER)
std::string("C:\\Users\\harshasi\\Desktop\\ProductAds\\100M-300K\\100M-300K.tsv"),
std::string("C:\\Users\\harshasi\\Desktop\\ProductAds\\100M-300K\\FullVocabulary.tsv"),
std::string("C:\\Users\\harshasi\\Desktop\\ProductAds\\100M-300K"));
#elif defined(LINUX)
std::string("/mnt/ProductAds100M-300K/100M-300K.tsv"),
std::string("/mnt/ProductAds100M-300K/Vocab1M.txt"),
std::string("/mnt/ProductAds100M-300K"));
#endif

trainer.train();
}
catch (...) {
std::cerr << "ProductAds 100M failed" << std::endl;
}*/
//try { // DSA UK Travel
//    vocabSz_t       vocab_size = 1562308;
//    docsSz_t        num_docs = 22083141;
//    offset_t        max_entries = 6322275419;
//    docsSz_t        num_topics = 2000;
//    ISLE::ISLETrainer trainer(vocab_size, num_docs, max_entries, num_topics,
//        true, 0.1, ISLE::ISLETrainer::data_ingest::FILE_DATA_LOAD,
//        std::string("/mnt/DSA_UK_Travel/train.tsvd"),
//        std::string("/mnt/DSA_UK_Travel/vocab.txt"),
//        std::string("/mnt/DSA_UK_Travel"));
//    trainer.train();
//}
//catch (...) {
//    std::cerr << "DSA UK Travel" << std::endl;
//}
