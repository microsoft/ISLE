#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <parallel/algorithm>
#include <sstream>
#include <string>

typedef struct TSVDEntry {
  int64_t col;  // or doc_id, the 1st col of TSVD file
  int64_t row;  // or word_od, the 2nd col of TSVD file
  float val;  // count (in float), the 3rd col of TSVD file is count in integer

  bool operator==(const TSVDEntry &other) {
    return this->col == other.col && this->row == other.row &&
           this->val == other.val;
  }

  std::string to_string() {
    return "(" + std::to_string(col) + ", " + std::to_string(row) + ", " +
           std::to_string(val) + ")";
  }
} TSVDEntry;

std::vector<std::string> split(std::string str, std::string token = "\t") {
  std::vector<std::string> result;
  while (str.size()) {
    int index = str.find(token);
    if (index != std::string::npos) {
      result.push_back(str.substr(0, index));
      str = str.substr(index + token.size());
      if (str.size() == 0)
        result.push_back(str);
    } else {
      result.push_back(str);
      str = "";
    }
  }
  return result;
}

void fill_entry(std::string line, int64_t line_no, int64_t nrows, int64_t ncols,
                TSVDEntry &entry) {
  std::vector<std::string> vstrings = split(line);
  //std::cout << vstrings[2].c_str() << "\n";
  if (vstrings.size() != 3) {
    std::cerr << "Ignoring " << line_no << ":" << " - expected 3 vals, got " << vstrings.size()  << "vstrings[0]= "<< vstrings[0]<< " - str = " << line << std::endl;
    entry.val = -1.0;
  } else {
    entry.row = std::stol(vstrings[1]) - 1;
    entry.col = std::stol(vstrings[0]) - 1;
    entry.val = (float) std::stof(vstrings[2]);
    if (entry.col >= ncols || entry.row >= nrows) {
      std::cerr << "Ignoring " << entry.to_string() << "\n";
      entry.val = -1;
    }
  }
}

void write_sum(std::string filename, float *arr, int num)
{ 
	std::ofstream ofs(filename);
	for (auto i = 0; i < num; i++){
		ofs << arr[i] << "\n";
	}
	ofs.close();
}

// write entries stored in `entries` into `csr_loc` in CSR format
void write_csr(std::vector<TSVDEntry> &entries, int64_t nrows, int64_t ncols,
               std::string csr_loc, float avg_doc_size) {
  // obtain filenames
  std::string csr_fname = csr_loc + ".csr";
  std::string col_fname = csr_loc + ".col";
  std::string off_fname = csr_loc + ".off";
  std::string info_fname = csr_loc + ".info";

  // create arrays to dump
  int64_t *offs = new int64_t[nrows + 1];
  int64_t *cur_cols = new int64_t[ncols];
  float *  cur_vals = new float[ncols];
  int64_t  cur_nnzs = 0;

  // nnzs
  int64_t nnzs = entries.size();

  std::ofstream csr_fs(csr_fname, std::ios::binary);
  std::ofstream col_fs(col_fname, std::ios::binary);

  offs[0] = 0;
  auto iter = entries.begin();

  for (int64_t r = 0; r < nrows; ++r) {
    cur_nnzs = 0;

    while (iter != entries.end() && iter->row == r) {
      cur_vals[cur_nnzs] = iter->val;
      cur_cols[cur_nnzs] = iter->col;
      ++iter;
      ++cur_nnzs;
    }
    offs[r + 1] = offs[r] + cur_nnzs;

    if (cur_nnzs > 0) {
      csr_fs.write((char *) cur_vals, cur_nnzs * sizeof(float));
      col_fs.write((char *) cur_cols, cur_nnzs * sizeof(int64_t));
    }
  }
  assert(offs[nrows] == nnzs);

  csr_fs.close();
  col_fs.close();

  // dump offs to disk
  std::ofstream offs_fs(off_fname, std::ios::binary);
  offs_fs.write((char *) offs, (nrows + 1) * sizeof(int64_t));
  offs_fs.close();

  // free memory
  delete[] offs;
  delete[] cur_cols;
  delete[] cur_vals;
  // dump info to disk
  std::ofstream info_fs(info_fname);
  info_fs << nrows << " " << ncols << " " << nnzs << " " << avg_doc_size << "\n";
  info_fs.close();
}
// write entries stored in `entries` into `csr_loc` in CSR format
void write_csc(std::vector<TSVDEntry> &entries, int64_t nrows, int64_t ncols,
               std::string csr_loc, float avg_doc_size) {
  // obtain filenames
  std::string csr_fname = csr_loc + ".csr";
  std::string col_fname = csr_loc + ".col";
  std::string off_fname = csr_loc + ".off";
  std::string info_fname = csr_loc + ".info";

  // create arrays to dump
  int64_t *offs = new int64_t[nrows + 1];
  int64_t *cur_cols = new int64_t[ncols];
  float *  cur_vals = new float[ncols];
  int64_t  cur_nnzs = 0;

  // nnzs
  int64_t nnzs = entries.size();

  std::ofstream csr_fs(csr_fname, std::ios::binary);
  std::ofstream col_fs(col_fname, std::ios::binary);

  offs[0] = 0;
  auto iter = entries.begin();

  for (int64_t r = 0; r < nrows; ++r) {
    cur_nnzs = 0;

    while (iter != entries.end() && iter->col == r) {
      cur_vals[cur_nnzs] = iter->val;
      cur_cols[cur_nnzs] = iter->row;
      ++iter;
      ++cur_nnzs;
    }
    offs[r + 1] = offs[r] + cur_nnzs;

    if (cur_nnzs > 0) {
      csr_fs.write((char *) cur_vals, cur_nnzs * sizeof(float));
      col_fs.write((char *) cur_cols, cur_nnzs * sizeof(int64_t));
    }
  }
  assert(offs[nrows] == nnzs);

  csr_fs.close();
  col_fs.close();

  // dump offs to disk
  std::ofstream offs_fs(off_fname, std::ios::binary);
  offs_fs.write((char *) offs, (nrows + 1) * sizeof(int64_t));
  offs_fs.close();

  // free memory
  delete[] offs;
  delete[] cur_cols;
  delete[] cur_vals;
  // dump info to disk
  std::ofstream info_fs(info_fname);
  info_fs << nrows << " " << ncols << " " << nnzs << " " << avg_doc_size << "\n";
  info_fs.close();
}

void write_tsvd(std::string filename, std::vector<TSVDEntry> &entries) {
  std::ofstream ofs(filename);
  for (TSVDEntry &entry : entries) {
    ofs << entry.col + 1 << " " << entry.row + 1 << " " << entry.val << "\n";
  }
  ofs.close();
}

void write_tsvd_binary(std::string filename, std::vector<TSVDEntry> &entries) {
  std::ofstream ofs(filename);
  ofs.write((char *) entries.data(), entries.size() * sizeof(TSVDEntry));
  ofs.close();
}

//function to find tf-idf scores 
void compute_tfidf(std::vector<TSVDEntry> &entries, int64_t num_docs) {
    //sort array in CSR_format 
    __gnu_parallel::sort(entries.begin(), entries.end(), [](const TSVDEntry &l,
                                                          const TSVDEntry &r) {
        return l.row < r.row || (l.row == r.row && l.col < r.col) ||
           (l.row == r.row && l.col == r.col && l.val < r.val);
    });
   
    //change the counts to tf-idf scores
    auto iter = entries.begin();
    auto pres = iter;  
    float idf = 0.0; 
    while(iter != entries.end()){
        auto curr_row = (*pres).row;
        auto doc_freq = 0; 
        while((*iter).row == curr_row && iter!=entries.end()){
            iter += 1; 
            doc_freq += 1;  
        }

       
        idf = std::log(((float)num_docs)/doc_freq); 
        for (auto it = pres; it < iter; it += 1){
            (*it).val = std::ceil((*it).val * idf);  
        }   
 
        pres = iter; 
    } 

    //sort the arr back to csc format
    __gnu_parallel::sort(
       entries.begin(), entries.end(),
       [](const TSVDEntry &l, const TSVDEntry &r) {
         return l.col < r.col || (l.col == r.col && l.row < r.row) ||
                (l.row == r.row && l.col == r.col && l.val < r.val);
    });  
}


int main(int argc, char **argv) {
  if (argc != 9) {
    std::cout << "usage: <exec> <tsvd_file> <csr location prefix> "
                 "<unique_tsvd(write 0 if you don't want)> <n_rows> "
                 "<n_cols> <n_nzs> <tf-idf(0/1)> <normalize(0/1)>\n";
    exit(-1);
  }

  std::string tsvdfile = argv[1];
  std::string csr_loc = argv[2];
  std::string tsvd_loc = argv[3];
  std::string csr_fname = csr_loc + ".csr";
  std::string col_fname = csr_loc + ".col";
  std::string off_fname = csr_loc + ".off";
  std::string info_fname = csr_loc + ".info";
  int64_t     nrows = std::stol(argv[4]);
  int64_t     ncols = std::stol(argv[5]);
  int64_t     nnzs = std::stol(argv[6]);
  bool        tf_idf = std::stoi(argv[7]);
  bool        normalize = std::stoi(argv[8]);

  std::string col_sum = csr_loc + ".col_sum"; 

  std::cout << "Reading TSVD file with nrows: " << nrows << "  ncols: " << ncols
            << "  nnzs: " << nnzs << std::endl;

  std::vector<TSVDEntry> entries;
  entries.reserve(nnzs);

  TSVDEntry entry;
  {  // Read the TSVD input file
    std::ifstream fs(tsvdfile);

    int64_t     line_no = 0;
    std::string line;
    for (std::string line; std::getline(fs, line) && (line_no < nnzs); line_no++) {
      fill_entry(line, line_no, nrows, ncols, entry);
      if (entry.val > 0) {
        entries.push_back(entry);
      }
      if (line_no % (10 * (1 << 20)) == 0) {
        std::cout << "Finished reading " << line_no << " lines" << std::endl;
      }
    }

    assert((int64_t) line_no == nnzs);
    fs.close();
    std::cout << "#entries (initial): " << entries.end() - entries.begin()
              << "\n";
    std::cout << "Finished reading from file, now sorting for CSC format"
              << std::endl;
    __gnu_parallel::sort(
        entries.begin(), entries.end(),
        [](const TSVDEntry &l, const TSVDEntry &r) {
          return l.col < r.col || (l.col == r.col && l.row < r.row) ||
                 (l.row == r.row && l.col == r.col && l.val < r.val);
        });
    assert(entries.back().col < ncols);
    std::cout << "Finished sorting for CSC format, now removing duplicates"
              << std::endl;
    // print out duplicates in sorted array
    for (auto iter = entries.begin(); iter != entries.end() - 1; ++iter) {
      if ((*iter == *(iter + 1)))
        std::cerr << "Duplicate found at location: " << iter - entries.begin()
                  << iter->to_string() << "  " << (iter + 1)->to_string()
                  << std::endl;
    }
    auto end_iter = std::unique(entries.begin(), entries.end(),
                                [](const TSVDEntry &l, const TSVDEntry &r) {
                                  return l.row == r.row && l.col == r.col;
                                });
    nnzs = end_iter - entries.begin();
    std::cout << "#entries (after unique): " << end_iter - entries.begin()
              << "\n";
    if (entries.size() < nnzs) {
      std::cerr << "detected " << nnzs - entries.size() << " duplicates"
                << std::endl;
    }
    entries.erase(end_iter, entries.end());
  }

  std::cout << "Finished deduplication, now writing CSC" << std::endl;
 

  if (tf_idf) {
      std::cout << "Computing tf-idf scores " << std::endl;
      compute_tfidf(entries, ncols);
  }
  
  // compute column sums
  float *sums = new float[ncols];
  memset(sums, 0, ncols * sizeof(float));
  for (auto entry : entries)
    sums[entry.col] += entry.val;
 
  int64_t empty_docs = 0;
  for (int64_t c = 0; c < ncols; ++c)
    if (sums[c] = 0.0)
      empty_docs++;
   
  // compute total sum
  float avg_doc_size = std::accumulate(sums, sums + ncols, 0.0) / ((ncols - empty_docs) * 1.0);
  std::cout << "Avg doc size: " << avg_doc_size << std::endl;

  if (normalize) {
#pragma omp parallel for
    for (int64_t idx = 0; idx < nnzs; ++idx) {
      entries[idx].val /= sums[entries.at(idx).col];
      entries[idx].val *= avg_doc_size;
    }
  }

  write_sum(col_sum, sums, ncols); 
  delete[] sums;

  // dump CSC to disk
  write_csc(entries, ncols, nrows, csr_loc + "_tr", avg_doc_size);
  std::cout << "Finished writing CSC, sorting for CSR format" << std::endl;
  // convert to csr
  __gnu_parallel::sort(entries.begin(), entries.end(), [](const TSVDEntry &l,
                                                          const TSVDEntry &r) {
    return l.row < r.row || (l.row == r.row && l.col < r.col) ||
           (l.row == r.row && l.col == r.col && l.val < r.val);
  });
  assert(entries.back().row < nrows);
  std::cout << "Finished sorting for CSR format, now writing CSR" << std::endl;
  // dump CSR to disk
  write_csr(entries, nrows, ncols, csr_loc, avg_doc_size);
  // std::cout << "Finished writing CSR, dumping binary TSVD" << std::endl;
  // dump sorted & de-duplicated entries to TSVD file
  if (tsvd_loc.compare("0") != 0) 
      write_tsvd(tsvd_loc, entries);
  //write_tsvd_binary(tsvd_loc, entries);

  // clear the vector
  entries.clear();
}
