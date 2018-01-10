// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <streambuf>
#include <functional>
#include <numeric>
#include <cmath>
#include <cassert>

#include "Eigen/Core"

#include "logger.h"
#include "types.h"
#include "hyperparams.h"

#define LINUX_MMAP_FILE_IO 1
#define WIN_MMAP_FILE_IO 2
#define NAIVE_FILE_IO 3


#ifdef _MSC_VER
#define FILE_IO_MODE WIN_MMAP_FILE_IO
#include <Windows.h>
#include <FileAPI.h>
#include <Winbase.h>
#endif

#ifdef LINUX
#define FILE_IO_MODE LINUX_MMAP_FILE_IO
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#endif


namespace ISLE
{

#if defined(_MSC_VER)
    uint64_t  open_win_mmapped_file_handle(
        const std::string filename,
        HANDLE& hFile, HANDLE& hMapFile, char** pBuf);

    void close_win_mmapped_file_handle(HANDLE& hFile, HANDLE& hMapFile);
#endif

#if defined(LINUX)
    uint64_t open_linux_mmapped_file_handle(
        const std::string filename,
        int& fd, void** buf);

    void close_linux_mmapped_file_handle(
        int fd, 
        void* buf,
        off_t fileSize);
#endif


    template<class COUNT_T>
    struct DocWordEntry
    {
        word_id_t word; 	// Zero based indexing 
        doc_id_t  doc;	// Zero based indexing
        COUNT_T   count;

        DocWordEntry() :
            word(0),
            doc(0),
            count((COUNT_T)0)
        {}

        DocWordEntry(const word_id_t& word_, const doc_id_t& doc_, const COUNT_T& count_) :
            word(word_),
            doc(doc_),
            count(count_)
        {}

        void operator=(const DocWordEntry& from)
        {
            doc = from.doc;
            word = from.word;
            count = from.count;
        }
    };

    struct DocWordEntriesReader
    {
        std::vector<DocWordEntry<count_t> >& entries;

        DocWordEntriesReader(std::vector<DocWordEntry<count_t> >& entries_)
            : entries(entries_)
        {}

        void read_from_file(
            const std::string& filename,
            const offset_t max_entries = 0)
        {
#if FILE_IO_MODE == NAIVE_FILE_IO
            {
                if (max_entries > 0)
                    entries.reserve(max_entries);
                DocWordEntry<count_t> entry;
                std::ifstream in(filename, std::ifstream::in);
                assert(in.is_open());
                while (max_entries == 0 || (max_entries > 0 && entries.size() < max_entries)) {
                    in >> entry.doc >> entry.word;
                    --entry.doc; --entry.word; // 1-based to 0-based indexing
                    if (!in.eof()) {
                        in >> entry.count;
                        entries.push_back(entry);
                    }
                    else break;
                }
                if (max_entries > 0)
                    assert(entries.size() == max_entries);
                in.close();
            }
#elif FILE_IO_MODE == WIN_MMAP_FILE_IO
            {
                HANDLE hFile, hMapFile;
                char *pBuf;
                uint64_t fileSize = open_win_mmapped_file_handle(filename, hFile, hMapFile, &pBuf);
                fill_doc_word_entries(max_entries, pBuf, fileSize);
                close_win_mmapped_file_handle(hFile, hMapFile);
            }
#elif FILE_IO_MODE == LINUX_MMAP_FILE_IO
            {
                int fd;
                void* buf;
                uint64_t fileSize = open_linux_mmapped_file_handle(filename, fd, &buf);
                fill_doc_word_entries(max_entries, (char*)buf, fileSize);
                close_linux_mmapped_file_handle(fd, buf, fileSize);
            }
#else
            assert(false);
#endif
        }


        struct membuf : std::streambuf
        {
            membuf(char* begin, char* end)
            {
                this->setg(begin, begin, end);
            }
        };

        void fill_doc_word_entries(
            offset_t max_entries,
            char* buf,
            uint64_t fileSize)
        {
            entries.resize(max_entries);
            doc_id_t doc = 0; word_id_t word = 0; count_t count = 0;
            offset_t nRead = 0;
            assert(sizeof(size_t) == 8);
            int state = 1; // Are we trying to read the 1st,2nd or 3rd entry of the line
            bool was_whitespace = false;
            for (size_t i = 0; i < fileSize; ++i) {
                assert(state < 4);
                switch (buf[i]) {
                case '\r':
                    assert(state == 3);
                    break;
                case '\n':
                    entries[nRead].doc = doc - 1;
                    entries[nRead].word = word - 1;
                    entries[nRead].count = count;
                    doc = word = count = 0; state = 1; nRead++;
                    break;
                case ' ':
                    was_whitespace = true;
                    //state++;
                    break;
                case '\t':
                    was_whitespace = true;
                    //state++;
                    break;
                case '0': case '1': case '2': case '3':	case '4':
                case '5': case '6': case '7': case '8': case '9':
                    //Allowing for multiple whitespaces
                    if (was_whitespace == true) {
                        state++;
                        was_whitespace = false;
                    }

                    switch (state) {
                    case 1:
                        doc *= 10; doc += buf[i] - '0';
                        break;
                    case 2:
                        word *= 10; word += buf[i] - '0';
                        break;
                    case 3:
                        count *= 10; count += buf[i] - '0';
                        break;
                    default:
                        assert(false);
                    }
                    break;
                default:
                    std::cerr << "Bad format\n";
                }
            }
            if (state == 3) { // Didnt reach "\n" on last line
                assert(doc > 0 && word > 0 && count > 0);
                entries[nRead].doc = doc - 1;
                entries[nRead].word = word - 1;
                entries[nRead].count = count;
                nRead++;
            }
            assert(nRead == max_entries);
        }
    };

#if FILE_IO_MODE == LINUX_MMAP_FILE_IO || FILE_IO_MODE == WIN_MMAP_FILE_IO
    class MMappedOutput
    {
#if FILE_IO_MODE == WIN_MMAP_FILE_IO
        HANDLE hFile;
        HANDLE hMapFile;
#elif FILE_IO_MODE == LINUX_MMAP_FILE_IO
        int fd;
#endif

        char *mmap_ptr;
        char *buf_base;
        char *buf_ptr;
        size_t write_buf_size;


    public:
        MMappedOutput(const std::string& filename)
        {
            write_buf_size = 1 << 20;
            buf_base = (char*)malloc(write_buf_size);
            buf_ptr = buf_base;

#if FILE_IO_MODE == WIN_MMAP_FILE_IO

            hFile = CreateFileA(
                filename.c_str(),
                GENERIC_WRITE | GENERIC_READ, 0, NULL, // default security
                CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
            assert(hFile != INVALID_HANDLE_VALUE);

#elif FILE_IO_MODE == LINUX_MMAP_FILE_IO

            fd = open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
            if (fd <= 0) {
                std::cerr << "File " << filename << " coult not be opened or accessed" << std::endl;
            }
#endif
        }


        ~MMappedOutput()
        {
            if (buf_base != NULL) free(buf_base);

#if FILE_IO_MODE == WIN_MMAP_FILE_IO

            if (mmap_ptr != NULL) UnmapViewOfFile(mmap_ptr);
            if (hMapFile != NULL) CloseHandle(hMapFile);
            if (hFile != NULL) CloseHandle(hFile);

#elif FILE_IO_MODE == LINUX_MMAP_FILE_IO

            if (mmap_ptr != 0) {
                std::cerr << "Error: mmap_ptr not null; you forgot to flush_and_close this mmap'd file ouptput" << std::endl;
                exit(-1);
            }
            if (fd != 0) {
                std::cerr << "Error: fd not 0; you forgot to flush_and_close this mmap'd file ouptput" << std::endl;
                exit(-1);
            }
#endif
        }

        void flush_and_close()
        {
            assert(buf_ptr - buf_base < write_buf_size);
            ptrdiff_t mmap_size = (buf_ptr - buf_base);

#if FILE_IO_MODE == WIN_MMAP_FILE_IO

            hMapFile =
                CreateFileMapping(hFile, NULL, PAGE_READWRITE,
                    mmap_size >> 32, mmap_size % (1ULL << 32), NULL);
            assert(hMapFile != NULL);

            mmap_ptr = (char*)MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, mmap_size);

#elif FILE_IO_MODE == LINUX_MMAP_FILE_IO

            if (lseek(fd, mmap_size - 1, SEEK_SET) == -1) {
                close(fd);
                std::cerr << "Error: can not seek to mmap_size offset" << std::endl;
                exit(-1);
            }

            if (write(fd, "", 1) == -1) {
                close(fd);
                std::cerr << "Error: can not write to end of the file" << std::endl;
                exit(-1);
            }

            mmap_ptr = (char*)mmap(NULL, mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (mmap_ptr == MAP_FAILED)
                std::cerr << "mmap for output failed with error: " << errno << std::endl;
#endif

            assert(mmap_ptr != NULL);
            memcpy(mmap_ptr, buf_base, mmap_size);
            //mmap_ptr[mmap_size] = 0;

            free(buf_base); buf_base = NULL;

#if FILE_IO_MODE == WIN_MMAP_FILE_IO

            UnmapViewOfFile(mmap_ptr); mmap_ptr = NULL;
            CloseHandle(hMapFile); hMapFile = NULL;
            CloseHandle(hFile); hFile = NULL;

#elif FILE_IO_MODE == LINUX_MMAP_FILE_IO

            if (munmap(mmap_ptr, mmap_size) == -1) {
                close(fd);
                std::cerr << "Error: munmap failed" << std::endl;
                exit(-1);
            }
            else {
                mmap_ptr = 0;
            }

            if (close(fd) == -1) {
                std::cerr << "Error: file close failed" << std::endl;
                exit(-1);
            }
            else {
                fd = 0;
            }
#endif
        }

        void double_buffer()
        {
            write_buf_size *= 2;
            auto len = buf_ptr - buf_base;
            buf_base = (char*)realloc((void*)buf_base, write_buf_size);
            assert(buf_base != NULL);
            buf_ptr = buf_base + len;
        }

        inline void reverse(
            char *const str,
            const int len)
        {
            int b = 0;
            int e = len - 1;
            while (b < e) {
                std::swap(*(str + b), *(str + e));
                e--; b++;
            }
        }

        // Base 10. 
        inline char* itoa_mv(
            int num,
            char* str,
            const char terminal)
        {
            //std::cout << num << "\n";
            int i = 0;
            if (num == 0) {
                str[i++] = '0';
            }
            else {
                bool neg = false;
                if (num < 0) {
                    neg = true;
                    num = -num;
                }
                while (num != 0) {
                    int rem = num % 10;
                    str[i++] = rem + '0';
                    num = num / 10;
                }
                if (neg)
                    str[i++] = '-';
                reverse(str, i);
            }
            str[i++] = terminal;
            return str + i;
        }

        inline void concat_int(const int64_t num, const char terminal = '\0')
        {
            if (buf_ptr - buf_base > write_buf_size - 100)
                double_buffer();
            buf_ptr = itoa_mv(num, buf_ptr, terminal);
        }

        template<class FPTYPE>
        inline char* ftoa_mv(
            FPTYPE num, char* str,
            const char terminal,
            const int before_dec = 6,
            const int after_dec = 6)
        {
            int i = 0;
            if (num == 0.0) {
                str[i++] = '0';
                str[i++] = '.';
                str[i++] = '0';
            }
            else {
                bool neg = false;
                if (num < 0.0) {
                    neg = true;
                    num = -num;
                }

                unsigned int num_int = (unsigned int)num;
                if (num_int == 0) {
                    str[i++] = '0';
                }
                else {
                    for (int d = 0; d < before_dec && num_int>0; ++d) {
                        int rem = num_int % 10;
                        str[i++] = rem + '0';
                        num_int = num_int / 10;
                    }
                    assert(num_int == 0);
                }
                if (neg)
                    str[i++] = '-';
                reverse(str, i);

                str[i++] = '.';
                FPTYPE frac = num - (FPTYPE)((int)num);
                for (int d = 0; d < after_dec; ++d) {
                    frac *= 10; assert((int)frac <= 9);
                    str[i++] = '0' + (int)frac;
                    frac -= (int)frac;
                }
            }
            str[i++] = terminal;
            return str + i;
        }

        template<class FPTYPE>
        void concat_float(
            FPTYPE num,
            const char terminal = '\0',
            const int before_dec = 3,
            const int after_dec = 6)
        {
            if (buf_ptr - buf_base > write_buf_size - 100)
                double_buffer();
            buf_ptr = ftoa_mv(num, buf_ptr, terminal);
        }

        void add_endline()
        {
            if (buf_ptr - buf_base > write_buf_size - 100)
                double_buffer();
            *buf_ptr = '\n';
            ++buf_ptr;
        }
    };
#endif

    template <class E>
    struct quintuple
    {
        E first, second, third, fourth, fifth;
        quintuple(
            const E& first_,
            const E& second_,
            const E& third_,
            const E& fourth_,
            const E& fifth_)
            : first(first_), second(second_), third(third_), fourth(fourth_), fifth(fifth_)
        {}

        inline bool operator== (const quintuple<E>& from)
        {
            return
                first == from.first &&
                second == from.second &&
                third == from.third &&
                fourth == from.fourth &&
                fifth == from.fifth;
        }
    };

    template <class T>
    struct quintuple_comp
    {
        inline bool operator() (
            const quintuple<T>& l,
            const quintuple<T>& r) const
        {
            return
                l.first < r.first ||
                (l.first == r.first && l.second < r.second) ||
                (l.first == r.first && l.second == r.second && l.third < r.third) ||
                (l.first == r.first && l.second == r.second && l.third == r.third
                    && l.fourth < r.fourth) ||
                    (l.first == r.first && l.second == r.second && l.third == r.third
                        && l.fourth == r.fourth && l.fifth < r.fifth);
        }
    };

    void create_vocab_list(
        const std::string& vocab_file,
        std::vector<std::string>& words,
        const word_id_t max_words);

    std::string log_dir_name(
        const doc_id_t num_topics,
        const std::string& output_path_base,
        const bool& sample_docs,
        const FPTYPE& sample_rate);

    template <class T>
    inline T divide_round_up(
        const T& num,
        const T& denom)
    {
        return (num%denom == 0)
            ? num / denom
            : (num / denom) + 1;
    }

    std::string concat_file_path(
        const std::string& dir,
        const std::string& filename);
}
