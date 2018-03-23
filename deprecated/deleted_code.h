// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

// Abstract base class for different Matrix representations
/*template<class T>
class Matrix {
protected:
word_id_t _vocab_size;	// Size of the vocabulary
doc_id_t  _num_docs;		// Number of documents
size_t    _nnzs;			// Number of non-zero entries
public:
Matrix(word_id_t d, doc_id_t s, size_t nnzs_=0)
: _vocab_size(d), _num_docs(s), _nnzs(nnzs_) {}

// Input: word-id and doc-id, both in 0-based indexing
// Output: A reference to the entry in the matrix
virtual inline T elem(word_id_t word, doc_id_t doc) const = 0;

inline size_t    get_nnzs()  const { return _nnzs; }
inline doc_id_t  num_docs()  const { return _num_docs;	}
inline word_id_t vocab_size()const { return _vocab_size; }
};*/


/*{  //Does not work.
FPTYPE eigen1, eigen100;
eigen1 = B_d_fl.get_singular_vals(0); eigen100 = B_d_fl.get_singular_vals(99);
FloatingPointSparseMatrix<float> B_sp_fl(s, s);
auto inner_nnzs = B_sp_fl.populate_with_inner_product(B_CSC, B_CSC, s*s/1);
std::cout << "#NNZs in the sparse representation of innerp product of thresholed matrix: "
<< inner_nnzs << std::endl;
B_sp_fl.init_feast_eigen_solve(eigen100*eigen100 - 0.00001, eigen1*eigen1 + 0.00001, 100);
B_sp_fl.compute_feast_eigen_vals();
}*/

/* // Can do k-means on (Sigma*VT) of B_d_fl instead
FloatingPointDenseMatrix<FPTYPE> B_topk_d_fl(vocab_size, num_docs);
B_topk_d_fl.populate_with_topk_singulars(B_d_fl, num_topics);
timer.next_time_secs("Computing best rank-k approx from SVD"); 

for (int i = 0; i < LLOYDS_REPS; ++i)
	B_topk_d_fl.lloyds_iter(num_topics, centers, NULL, true);
timer.next_time_secs("LLoyds k-means on B_k");*/

//WordCountDenseMatrix A(vocab_size, s); A.populate(entries);

//Model.print_words_above_threshold(t, 
//std::ceil((FPTYPE)0.1*(FPTYPE)A_sp.num_docs()/(FPTYPE)num_topics), vocab_words);

/*inline FPTYPE sparse_dot(const doc_id_t i, const doc_id_t j) const {
	auto offset_i = offsets_CSC[i];
	auto offset_j = offsets_CSC[j];
	FPTYPE ret = 0.0;

	while (offset_i != offsets_CSC[i + 1]
		&& offset_j != offsets_CSC[j + 1]) {
		if (rows_CSC[offset_i] == rows_CSC[offset_j])
			ret += vals_CSC[offset_i++] * vals_CSC[offset_j++];
		else if (rows_CSC[offset_i] > rows_CSC[offset_j])
			++offset_j;
		else
			++offset_i;
	}
	return ret;
}
*/

/*auto closest_docs_lowd = new std::vector<doc_id_t>[num_topics];
FPTYPE *docs_l2sq_lowd = new FPTYPE[B_spectraSigmaVT_d_fl.num_docs()];
B_spectraSigmaVT_d_fl.compute_docs_l2sq(docs_l2sq_lowd);

std::vector<size_t> prev_cl_sizes(num_topics, 0);
auto prev_closest_docs_lowd = new std::vector<doc_id_t>[num_topics];

for (int i = 0; i < MAX_LLOYDS_LOWD_REPS; ++i) {
B_spectraSigmaVT_d_fl.lloyds_iter(num_topics, centers_lowd, docs_l2sq_lowd, closest_docs_lowd, true);

bool clusters_changed = false;
for (int topic = 0; topic < num_topics; ++topic) {
if (prev_cl_sizes[topic] != closest_docs_lowd[topic].size())
clusters_changed = true;
prev_cl_sizes[topic] = closest_docs_lowd[topic].size();
}

if (!clusters_changed)
for (int topic = 0; topic < num_topics; ++topic) {
std::sort(closest_docs_lowd[topic].begin(), closest_docs_lowd[topic].end());
std::sort(prev_closest_docs_lowd[topic].begin(), prev_closest_docs_lowd[topic].end());

if (prev_closest_docs_lowd[topic] != closest_docs_lowd[topic])
clusters_changed = true;
prev_closest_docs_lowd[topic] = closest_docs_lowd[topic];
}

if (!clusters_changed)
break;
}
delete docs_l2sq_lowd;
delete[] closest_docs_lowd;
delete[] prev_closest_docs_lowd;
*/


/*
// Full SVD with LAPACKe_?FPgesvd
B_d_fl.initialize_for_full_svd();
B_d_fl.compute_full_svd();
timer.next_time_secs("Computing SVD with LAPACK");
B_d_fl.compare_LAPACK_Spectra(num_topics, 0.0001, 0.001);
B_d_fl.cleanup_full_svd();
*/

//FPTYPE dist = B_d_fl.kmeanspp_on_col_space(num_topics, kmeans_seeds[rep], EIGEN_SOURCE_SPECTRA);

//auto kmeans_seeds = new std::vector<doc_id_t>[KMEANS_INIT_REPS];
//FPTYPE min_total_dist_to_centers = FP_MAX; int best_seed;
//for (int rep = 0; rep < KMEANS_INIT_REPS; ++rep) {
//	FPTYPE dist = B_spectraSigmaVT_d_fl.kmeanspp(
//					USE_TWO_STEP_SEEDING ? (word_id_t) CL_OVERSAMPLING*num_topics : num_topics,
//					kmeans_seeds[rep]);
//	if (dist < min_total_dist_to_centers) {
//		min_total_dist_to_centers = dist;
//		best_seed = rep;
//	}
//}
//std::vector<doc_id_t> best_kmeans_seeds(kmeans_seeds[best_seed]);
//delete[] kmeans_seeds;


// Compute out = mat * tr(mat)
// mat in row-major format
//void bit_outer_prod(FPTYPE *out, 
//					const uint64_t* mat,
//					const MKL_INT nrows, 
//					const MKL_INT ncols) {
//
//	for (auto row1 = 0; row1 < nrows; ++row1) {
//		for (auto row2 = 0; row2 < row1; ++row2) {
//			MKL_INT count = 0;
//			size_t row_offset1 = row1 * (ncols / 64);
//			size_t row_offset2 = row2 * (ncols / 64);
//			for (size_t blk = 0; blk < ncols / 64; ++blk)
//				count += __popcnt64(mat[row_offset1 + blk] & mat[row_offset2 + blk]);
//
//			out[row1*ncols + row2] = (FPTYPE)count;
//			out[row2*ncols + row1] = (FPTYPE)count;
//		}
//	}
//}

// Call truncated Symm Eigensolve on BBT to get squared singular vals and U_trunc
/*	Spectra::DenseSymMatProd<FPTYPE> op(BBT);
Spectra::SymEigsSolver<FPTYPE, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<FPTYPE> >
eigs(&op, num_topics, 2 * num_topics + 1);*/
/*	MKL_DenseGenMatProd<FPTYPE> op(BBT);
Spectra::SymEigsSolver<FPTYPE, Spectra::LARGEST_ALGE, MKL_DenseGenMatProd<FPTYPE> >
eigs(&op, num_topics, 2 * num_topics + 1);*/

/* char* write_buf = new char[write_buf_size];
DWORD dw_bytes_written;
assert(WriteFile(hFile, write_buf, total_bytes, &dw_bytes_written, NULL));
assert(dw_bytes_written == total_bytes);*/


/*
//
// Output doc-topic catchword sums in 1-based index
//
#if FILE_IO_MODE == NAIVE_FILE_IO
{ // Naive File IO
std::ofstream out_doc_topic;
out_doc_topic.open(log_dir + "\\DocTopicCatchwordSums.tsv");
for (auto doc = 0; doc < A_sp.num_docs(); ++doc)
for (auto topic = 0; topic < num_topics; ++topic) {
FPTYPE sum = 0.0;
for (auto w_iter = catchwords[topic].begin(); w_iter != catchwords[topic].end(); ++w_iter)
sum += A_sp.normalized(*w_iter, doc);
if (sum > 0.0)
out_doc_topic << doc + 1 << "\t" << topic + 1 << "\t" << sum << "\n";
}
out_doc_topic;
timer.next_time_secs("Writing document-topic catchword sums");
}
#endif
#if FILE_IO_MODE == WIN_MMAP_FILE_IO
{ // Memory mapped File IO
std::string filename = log_dir + "\\DocTopicCatchwordSums.tsv";
//size_t max_write_buf_size = num_docs * num_topics * 30;
MMappedOutput out(filename);// , max_write_buf_size);
for (auto doc = 0; doc < A_sp.num_docs(); ++doc)
for (auto topic = 0; topic < num_topics; ++topic) {
FPTYPE sum = 0.0;
for (auto w_iter = catchwords[topic].begin();
w_iter != catchwords[topic].end(); ++w_iter)
sum += A_sp.normalized(*w_iter, doc);
if (sum > 0.0) {
out.concat_int(doc + 1, '\t');
out.concat_int(topic + 1, '\t');
out.concat_float(sum, '\n');
}
}
out.flush_and_close();
timer.next_time_secs("Writing document-topic catchword sums");
}
#endif*/

//void compute_outer_product_via_bitrep(Eigen::MatrixX& BBT,
//	const std::vector<A_TYPE>& thresholds) {
//	Timer timer;
//	std::vector<WordDocPair> entries;
//	std::vector<offset_t> word_offsets;
//	get_word_major_list(entries, word_offsets);
//	timer.next_time_secs("list and sort entries", 30);
//
//
//	MKL_INT vec_blk_size = 256; // Vector unit size
//	doc_id_t num_docs_blks = num_docs() / vec_blk_size;
//	if (num_docs() % vec_blk_size > 0)
//		++num_docs_blks;
//	doc_id_t num_docs_rnd = num_docs_blks * vec_blk_size;
//	size_t bitrep_sz = vocab_size() * (num_docs_rnd / 64);
//#if VECTOR_INTRINSICS
//	auto bitrep = (uint64_t*)_aligned_malloc(bitrep_sz * sizeof(uint64_t), vec_blk_size);
//#else
//	auto bitrep = new uint64_t[bitrep_sz];
//#endif
//	memset((void*)bitrep, 0, sizeof(uint64_t)*bitrep_sz);
//	timer.next_time_secs("Alloc and set bitrep", 30);
//
//	pfor(auto word = 0; word < vocab_size(); ++word) {
//		//size_t off_before = 0;	int doc_before = -1;
//		size_t word_offset = word * (num_docs_rnd / 64);
//		for (auto diter = entries.begin() + word_offsets[word];
//			diter < entries.begin() + word_offsets[word + 1]; ++diter) {
//
//			size_t doc_offset = (diter->doc / 64) + word_offset;
//			int bit = diter->doc % 64;
//			bitrep[doc_offset] |= (1ULL << (63 - bit));
//		}
//	}
//	size_t total_popcnt = 0;
//	for (auto blk = 0; blk < vocab_size()* (num_docs_rnd / 64); ++blk)
//		total_popcnt += __popcnt64(bitrep[blk]);
//	assert(total_popcnt == get_nnzs());
//	assert(total_popcnt == entries.size());
//	timer.next_time_secs("Populating bitrep", 30);
//
//	auto zetas = new FPTYPE[vocab_size()];
//	for (auto word = 0; word < vocab_size(); ++word)
//		zetas[word] = std::sqrt((FPTYPE)thresholds[word]);
//
//	{ // Iterative version of bit matrix multiplication
//	  #ifdef OPENMP
//	  #pragma omp parallel for schedule(dynamic, 16)
//	  #endif	
//		pfor_dynamic_16(auto word1 = 0; word1 < vocab_size(); ++word1) {
//			for (auto word2 = 0; word2 <= word1; ++word2) {
//				MKL_INT count = 0;
//				size_t offset1 = word1 * (num_docs_rnd / 64);
//				size_t offset2 = word2 * (num_docs_rnd / 64);
//				for (size_t blk = 0; blk < num_docs_rnd / 64; ++blk)
//					count += __popcnt64(bitrep[offset1 + blk] & bitrep[offset2 + blk]);
//				// #include <smmintrin.h> _mmpopcnt_u64();
//				BBT(word2, word1) = zetas[word1] * zetas[word2] * (FPTYPE)count;
//			}
//		}
//		for (auto word1 = 0; word1 < vocab_size(); ++word1)
//			for (auto word2 = 0; word2 < word1; ++word2)
//				BBT(word1, word2) = BBT(word2, word1);
//		timer.next_time_secs("iterative bitvec prod", 30);
//	}
//
//	// The following works, but needs to be parallelized.
//	{ // recursive version of bit matrix multiplication
//	memset((void*)bbt.data(), (fptype)0.0, sizeof(fptype)*bbt.rows()*bbt.cols());
//	bit_outer_prod(bbt.data(), bitrep, vocab_size(), num_docs_rnd);
//	for (auto word1 = 0; word1 < vocab_size(); ++word1)
//	for (auto word2 = 0; word2 < vocab_size(); ++word2)
//	bbt(word1, word2) = zetas[word1] * zetas[word2] * bbt(word1, word2);
//	timer.next_time_secs("recursive bitvec prod", 30);
//	}
//
//#if VECTOR_INTRINSICS
//	_aligned_free(bitrep);
//#else
//	delete[] bitrep;
//#endif
//	delete[] zetas;
//}

// Deleted methods from FloatingPointSparseMatrix
// {
//
//	/*eigenvals(NULL), residual(NULL), eigenvecs(NULL),
//	est_num_eigen_vals(0), found_num_eigen_vals(0), eigen_range_min(0), eigen_range_max(0) {}*/
//
//	/*	if (eigenvals) delete[] eigenvals;
//		if (residual)  delete[] residual;
//		if (eigenvecs) delete[] eigenvecs; */
//
//
//	void sp_sp_product(const std::vector<DocWordEntry<FPTYPE> >& entries,
//		const std::vector<offset_t >& word_offsets,
//		Eigen::MatrixX& BBT,
//		const word_id_t r1_b, const word_id_t r1_e,
//		const word_id_t r2_b, const word_id_t r2_e) {
//		const offset_t len1 = word_offsets[r1_e] - word_offsets[r1_b];
//		const offset_t len2 = word_offsets[r2_e] - word_offsets[r2_b];
//		int LEN_THRESHOLD = 1 << 15;
//		if (r1_e - r1_b <= 1 || r2_e - r2_b <= 2
//			|| (len1 <= LEN_THRESHOLD && len2 <= LEN_THRESHOLD)) {
//			for (auto i = r1_b; i < r1_e; ++i)
//				for (auto j = r2_b; j < r2_e && j <= r1_e; ++j) {
//
//					FPTYPE prod = 0.0;
//
//					auto pos1 = word_offsets[i];
//					auto pos2 = word_offsets[j];
//
//					while (pos1 < word_offsets[i + 1] && pos2 < word_offsets[j + 1]) {
//						if (entries[pos1].doc == entries[pos2].doc)
//							prod += entries[pos1++].count * entries[pos2++].count;
//						else if (entries[pos1].doc < entries[pos2].doc)
//							pos1++;
//						else
//							pos2++;
//					}
//
//					BBT(i, j) = prod;
//					BBT(j, i) = prod;
//				}
//		}
//		else if (len1 > len2) {
//			sp_sp_product(entries, word_offsets, BBT, r1_b, (r1_b + r1_e) / 2, r2_b, r2_e);
//			sp_sp_product(entries, word_offsets, BBT, (r1_b + r1_e) / 2, r1_e, r2_b, r2_e);
//		}
//		else {
//
//			sp_sp_product(entries, word_offsets, BBT, r1_b, r1_e, r2_b, (r2_b + r2_e) / 2);
//			sp_sp_product(entries, word_offsets, BBT, r1_b, r1_e, (r2_b + r2_e) / 2, r2_e);
//		}
//	}
//
//	void compute_outer_product_via_sp_sp_prod(Eigen::MatrixX& BBT) {
//		std::vector<DocWordEntry<FPTYPE> > entries;
//		std::vector<offset_t> word_offsets;
//		get_word_major_list(entries, word_offsets);
//		sp_sp_product(entries, word_offsets, BBT, 0, vocab_size(), 0, vocab_size());
//	}
//
//FPTYPE sparse_vectors_inner_product(const WordCountSparseMatrix& B,
//									const doc_id_t i,
//									const WordCountSparseMatrix& C,
//									const doc_id_t j) {
//	FPTYPE prod = 0;
//	if (B.offset_CSC(i + 1) == B.offset_CSC(i) || C.offset_CSC(j + 1) == C.offset_CSC(j))
//		return 0;
//	auto B_pos = B.offset_CSC(i), C_pos = C.offset_CSC(j);
//	while (B_pos < B.offset_CSC(i + 1) && C_pos < C.offset_CSC(j + 1))
//		if (B.row_CSC(B_pos) == C.row_CSC(C_pos)) {
//			prod += (FPTYPE)B.val_CSC(B_pos) * (FPTYPE)C.val_CSC(C_pos);
//			++B_pos; ++C_pos;
//		}
//		else if (B.row_CSC(B_pos) < C.row_CSC(C_pos))
//			++B_pos;
//		else
//			++C_pos;
//	return prod;
//}
//// Unoptimized, returns B*C
//offset_t populate_with_outer_product(WordCountSparseMatrix& B,
//									WordCountSparseMatrix &C,
//									offset_t estimated_nnzs) {
//	allocate(estimated_nnzs);
//	assert(B.vocab_size() == C.vocab_size());
//	assert(num_docs() == num_docs());
//	assert(vocab_size() == num_docs());
//	offsets_CSC[0] = 0;
//	offset_t pos = 0;
//	for (doc_id_t i = 0; i < num_docs(); ++i) {
//		for (doc_id_t j = 0; j < num_docs(); ++j) {
//			auto prod = sparse_vectors_inner_product(B, i, C, j);
//			if (prod > 0) {
//				vals_CSC[pos] = prod;
//				rows_CSC[pos] = j;
//				++pos;
//				assert(pos <= estimated_nnzs);
//			}
//		}
//		offsets_CSC[i + 1] = pos;
//	}
//	nnzs = pos;
//	return nnzs;
//}
// For FEAST EigenSolve, FEAST needs to be tested.
//FPTYPE eigen_range_min, eigen_range_max;
//FPTYPE *eigenvals;
//FPTYPE *eigenvecs;
//FPTYPE *residual;
//MKL_INT est_num_eigen_vals;
//MKL_INT found_num_eigen_vals;
//
//	void init_feast_eigen_solve(FPTYPE eigen_range_min_, FPTYPE eigen_range_max_, const int est_num_eigen_vals_) {
//		est_num_eigen_vals = est_num_eigen_vals_;
//		eigen_range_min = eigen_range_min_; eigen_range_max = eigen_range_max_;
//		eigenvals = new FPTYPE[est_num_eigen_vals];
//		residual = new FPTYPE[est_num_eigen_vals];
//		eigenvecs = new FPTYPE[est_num_eigen_vals*num_docs()];
//	}
//
//	void compute_feast_eigen_vals() {
//		assert(eigen_range_min < eigen_range_max);
//		MKL_INT fpm[128]; // Input parameters
//		MKL_INT info; // 0 for success
//		MKL_INT loops;
//		FPTYPE epsout;
//		char UPLO = 'F';
//		feastinit(fpm);
//		sfeast_scsrev(&UPLO, (const int*)&s,
//			vals_CSC, (const int*)offsets_CSC, (const int*)rows_CSC,
//			fpm, &epsout, &loops,
//			&eigen_range_min, &eigen_range_max, &est_num_eigen_vals,
//			eigenvals, eigenvecs, &found_num_eigen_vals, residual, &info);
//		assert(info == 0); // Succesfull solve
//		std::cout << "FEAST eigen solve successfully completed" << std::endl;
//		std::cout << "\t Num of loops: " << loops << std::endl;
//		std::cout << "\t Trace error" << epsout << std::endl;
//
//	}
//};


//class WordCountSparseMatrix : public SparseMatrix<count_t> {
//public:
//	WordCountSparseMatrix(word_id_t d_, doc_id_t s_, offset_t nnzs_ = 0)
//		: SparseMatrix<count_t>(d_, s_, nnzs_) {}
//
//	~WordCountSparseMatrix() {}
//
//
//	void populate_from_dense(WordCountDenseMatrix& from, offset_t nnzs) {
//		allocate(nnzs);
//		offset_t nz_count = 0;
//		offsets_CSC[0] = 0;
//		uint64_t total_word_count;
//		for (doc_id_t j = 0; j < num_docs(); ++j) {
//			for (word_id_t i = 0; i < vocab_size(); ++i) {
//				if (from.elem(i, j) != 0) {
//					vals_CSC[nz_count] = from.elem(i, j);
//					total_word_count += vals_CSC[nz_count];
//					rows_CSC[nz_count] = i;
//					++nz_count;
//				}
//			}
//			offsets_CSC[j + 1] = nz_count;
//		}
//		avg_doc_sz = (count_t)(total_word_count / num_docs());
//		assert(nz_count == nnzs);
//	}
//
//	friend class FloatingPointSparseMatrix<FPTYPE>;
//};



// Compute BBT = this*(this^T)
// if @normalized==true, use normalized_vals_CSC instead of vals_CSC
//void compute_outer_product_via_dense(
//	Eigen::MatrixX& BBT,
//	const bool normalized = false) 
//{
//	// Set BBT to all zeros
//	FPscal(vocab_size()* vocab_size(), (FPTYPE)0.0, BBT.data(), 1);

//	doc_id_t slice_sz = 131072;
//	size_t num_slices = num_docs() / slice_sz;
//	if (num_docs() % slice_sz > 0)
//		num_slices++;

//	FPTYPE *temp = new FPTYPE[vocab_size()*slice_sz];
//	Timer t;
//	for (auto slice = 0; slice < num_slices; ++slice) {
//		FPscal(slice_sz*vocab_size(), (FPTYPE)0.0, temp, 1);

//		doc_id_t doc_b = slice*slice_sz;
//		doc_id_t doc_e = (slice + 1)*slice_sz > num_docs()
//			? num_docs()
//			: (slice + 1) * slice_sz;
//		std::cout << "Slice: " << slice << "[" << doc_b << "," << doc_e << ")\n";

//		csc_slice_to_dns(temp, doc_b, doc_e, normalized);
//		t.next_time_secs("slice");
//		FPgemm(CblasColMajor, CblasNoTrans, CblasTrans,
//			(MKL_INT)vocab_size(), (MKL_INT)vocab_size(), (MKL_INT)(doc_e - doc_b),
//			1.0, temp, (MKL_INT)vocab_size(), temp, (MKL_INT)vocab_size(),
//			1.0, BBT.data(), vocab_size());
//		t.next_time_secs("sgemm");
//	}
//	delete[] temp;
//}


// Uses Intel MKL's LAPACKe_?FPgesvd; matrix = U*Sigma*VT
//Call initialize_for_full_svd() before calling this method
//void compute_full_svd()
//{
//	auto superb = new FPTYPE[num_singular_vals];
//	auto info = FPgesvd(LAPACK_COL_MAJOR, 'S', 'S',
//		(lapack_int)vocab_size(), (lapack_int)num_docs(),
//		svd_temp, (lapack_int)vocab_size(),				// data and its lda
//		Sigma,
//		U, (lapack_int)vocab_size(),
//		VT, (lapack_int)num_singular_vals,
//		superb);
//	assert(info == 0); // FPgesvd has converged
//	delete[] superb;
//}

//// Find the distance to the center from @centers to which column @c is closest 
//FPTYPE distsq_to_closest_center_naive(
//	const doc_id_t c,
//	std::vector<doc_id_t> &centers) const
//{
//	assert(centers.size() > 0);
//	FPTYPE min_dist = FPdot(vocab_size(), data() + centers[0]*vocab_size(), 1, 
//										data() + centers[0] * vocab_size(), 1)
//					- 2*FPdot(vocab_size(), data() + c*vocab_size(), 1,
//										data() + centers[0] * vocab_size(), 1);
//	for (auto idx_iter = centers.begin() + 1; idx_iter != centers.end(); ++idx_iter) {
//		FPTYPE dist = FPdot(vocab_size(), data() + *idx_iter*vocab_size(), 1,
//										data() + *idx_iter*vocab_size(), 1)
//					- 2*FPdot(vocab_size(), data() + c*vocab_size(), 1,
//										data() + *idx_iter*vocab_size(), 1);
//		min_dist = (dist > min_dist) ? min_dist : dist;
//	}
//	return min_dist + FPdot(vocab_size(), data() + c*vocab_size(), 1, data() + c*vocab_size(), 1);
//}

// Input @k: number of centers, @centers: reference to vector of indices of chosen seeds
// Output: Sum of distances of all points to chosen seeds
//FPTYPE kmeanspp_naive(
//	const doc_id_t k,
//	std::vector<doc_id_t> &centers)
//{
//	centers.push_back((doc_id_t)(rand() * 84619573 % num_docs()));
//	std::vector<FPTYPE> dist_cumul(num_docs());
//	while (centers.size() < k) {
//		dist_cumul[0] = distsq_to_closest_center_naive(0, centers);
//		for (doc_id_t c = 1; c < num_docs(); ++c)
//			dist_cumul[c] = dist_cumul[c - 1] + distsq_to_closest_center_naive(c, centers);
//		for (auto iter = centers.begin(); iter != centers.end(); ++iter) { // Sanity checks
//			// Disance from center to its closest center == 0
//			assert(0 == (*iter > 0 ? dist_cumul[*iter] - dist_cumul[(*iter) - 1] : dist_cumul[0]));
//			// Center is not replicated
//			assert(std::find(centers.begin(), centers.end(), *iter) == iter); 
//			assert(std::find(iter+1, centers.end(), *iter) == centers.end()); 
//		}
//		FPTYPE dice_throw = dist_cumul[num_docs() - 1] * (FPTYPE)rand() / (FPTYPE)RAND_MAX;
//		auto new_center = (doc_id_t)(std::lower_bound(dist_cumul.begin(), dist_cumul.end(), dice_throw) 
//									- dist_cumul.begin());
//		std::cout << new_center << std::endl;
//		centers.push_back(new_center);
//	}
//	return dist_cumul[num_docs() - 1];
//}


/** BBT option in compute_svd */
//#if USE_DENSE_BBT
//assert(BBT.rows() == vocab_size() && BBT.cols() == vocab_size());
//MKL_DenseSymMatProd<FPTYPE> op(BBT);
//Spectra::SymEigsSolver<FPTYPE, Spectra::LARGEST_ALGE, MKL_DenseSymMatProd<FPTYPE> >
//eigs(&op, (MKL_INT)num_topics, 2 * (MKL_INT)num_topics + 1);
//#endif


// TODO: convert to mkl_?FPdnscsr
//void csc_slice_to_dns(
//	FPTYPE *const out,
//	const doc_id_t doc_b,
//	const doc_id_t doc_e,
//	const bool normalized)
//{
//	if (!normalized) {
//		pfor_static_1024(int doc = doc_b; doc < doc_e; ++doc)
//			for (offset_t pos = offset_CSC(doc); pos < offset_CSC(doc + 1); ++pos)
//				out[row_CSC(pos) + (doc - doc_b)*vocab_size()] = val_CSC(pos);
//	}
//	else {
//		assert(normalized_vals_CSC != NULL);
//		pfor_static_1024(int doc = doc_b; doc < doc_e; ++doc)
//			for (offset_t pos = offset_CSC(doc); pos < offset_CSC(doc + 1); ++pos)
//				out[row_CSC(pos) + (doc - doc_b)*vocab_size()] = normalized_vals_CSC[pos];
//	}
//}

//	// Initialize SigmaVT for num_topics columns, each of vocab_size
//	void initialize_for_Spectra_via_dns(
//		const doc_id_t num_topics,
//		const bool normalized = false)
//	{
//		SigmaVT = new FPTYPE[num_topics*num_docs()];
//
//		// Construct BBT = this * (this)^T, defaults to col-major
//#if USE_DENSE_BBT
//		BBT.resize(vocab_size(), vocab_size());
//		compute_outer_product_via_dense(BBT, normalized);
//		//compute_outer_product_via_sp_sp_prod(BBT);
//#endif
//	}
//
//	void initialize_for_Spectra_via_bitrep(
//		const doc_id_t num_topics,
//		std::vector<A_TYPE>& thresholds = NULL)
//	{
//		SigmaVT = new FPTYPE[num_topics*num_docs()];
//
//		// Construct BBT = this * (this)^T, defaults to col-major
//		assert(USE_DENSE_BBT);
//		BBT.resize(vocab_size(), vocab_size());
//		compute_outer_product_via_bitrep(BBT, thresholds);
//	}
