// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "utils.h"

void ISLE::create_vocab_list(
    const std::string& vocab_file,
    std::vector<std::string>& words,
    const word_id_t max_words)
{
    assert(words.size() == 0);
    std::ifstream in(vocab_file, std::ifstream::in);
    assert(in.is_open());
    std::string word;

    while (!in.eof() && words.size() < max_words)
    {
        in >> word;
        words.push_back(word);
    }

    words.resize(max_words);
    assert(words.size() == max_words);
    in.close();
}


std::string ISLE::log_dir_name(
    const doc_id_t num_topics,
    const std::string& output_path_base,
    const bool& sample_docs,
    const FPTYPE& sample_rate)
{
    auto log_subdir = "log_t_"
        + std::to_string(num_topics)
        + "_eps1_" + std::to_string(eps1_c)
        + "_eps2_" + std::to_string(eps2_c)
        + "_eps3_" + std::to_string(eps3_c)
        + "_kMppReps_" + std::to_string(KMEANS_INIT_REPS)
        + "_kMLowDReps_" + std::to_string(MAX_KMEANS_LOWD_REPS)
        + "_kMReps_" + std::to_string(MAX_KMEANS_REPS)
        + "_sample_" + std::to_string(sample_docs);
    if (sample_docs)
        log_subdir += "_Rate_" + std::to_string(sample_rate);
    return concat_file_path(output_path_base, log_subdir);
}

void ISLE::create_dir(const std::string& dir_name)
{
#if defined(_MSC_VER)
    std::wstring log_dir_wstr = std::wstring(dir_name.begin(), dir_name.end());
    LPCWSTR log_dir_lpcwstr = log_dir_wstr.c_str();
    if (!(CreateDirectory(log_dir_lpcwstr, NULL) ||
        ERROR_ALREADY_EXISTS == GetLastError()))
        std::cerr << "Subdir creation error" << std::endl;
#elif defined(LINUX)
    std::string log_dir_wstr = std::string(dir_name.begin(), dir_name.end());
    char* log_dir_lpcwstr = new char[4000];
    strcpy(log_dir_lpcwstr, log_dir_wstr.c_str());
    struct stat st = { 0 };
    if (stat(log_dir_lpcwstr, &st) == -1)
        mkdir(log_dir_lpcwstr, S_IRWXU);
    else
        std::cerr << "Subdir exists already" << std::endl;
    delete[] log_dir_lpcwstr;
#else
    assert(false);
#endif
}

std::string ISLE::concat_file_path(
    const std::string& dir, 
    const std::string& filename)
{
#if defined(_MSC_VER)
    return dir + "\\" + filename;
#elif defined(LINUX)
    return dir + "/" + filename;
#else
    assert(false);
#endif
}

#if defined(_MSC_VER)
uint64_t  ISLE::open_win_mmapped_file_handle(
    const std::string filename,
    HANDLE& hFile, HANDLE& hMapFile, char** pBuf)
{
    hFile =
        CreateFileA(filename.c_str(),
            GENERIC_READ, FILE_SHARE_READ, NULL, // default security
            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    assert(hFile != INVALID_HANDLE_VALUE);

    LARGE_INTEGER fileSizeLI;
    GetFileSizeEx(hFile, &fileSizeLI);
    //assert(fileSizeLI.HighPart == 0); 		// Can map up to 2GB files??
    uint64_t fileSize = fileSizeLI.LowPart + (((uint64_t)fileSizeLI.HighPart) << 32);

    hMapFile =
        CreateFileMapping(hFile,
            NULL, PAGE_READONLY,	// default security & READ/WRITE mode
            0, 0,			// maximum object sizes (high-order, lower-order DWORD)
            NULL);					// name of mapping object
    assert(hMapFile != NULL);

    // 3rd and 4th params are higher and lower order bits of start offset
    // 5th param = 0 for mapping entire file.
    *pBuf = NULL;
    *pBuf = (char*)MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, 0);
    assert(*pBuf != NULL);
    return fileSize;
}

void ISLE::close_win_mmapped_file_handle(HANDLE& hFile, HANDLE& hMapFile)
{
    CloseHandle(hMapFile);
    CloseHandle(hFile);
}
#endif

#if defined(LINUX)

uint64_t ISLE::open_linux_mmapped_file_handle(
    const std::string filename,
    int& fd, void** buf)
{
    fd = open(filename.c_str(), O_RDONLY);
    if (!(fd > 0)) {
        std::cerr << "Data file " << filename << " not found. Program will stop now." << std::endl;
        assert(false);
    }
    struct stat sb;
    assert(fstat(fd, &sb) == 0);
    off_t fileSize = sb.st_size;
    assert(sizeof(off_t) == 8);

    *buf = mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
    assert(*buf);
    return fileSize;
}

void ISLE::close_linux_mmapped_file_handle(int fd, void* buf, off_t fileSize)
{
    assert(munmap(buf, fileSize) == 0);
    close(fd);
}
#endif
