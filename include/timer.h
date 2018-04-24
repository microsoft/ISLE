// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <ctime>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>

#include "logger.h"

namespace ISLE
{
    class Timer {
        clock_t usert_begin, usert_before, usert_after, usert_last;
        std::chrono::time_point<std::chrono::high_resolution_clock> syst_begin, syst_before, syst_after, syst_last;
        std::ofstream time_log;
        bool is_log_file_open;
    public:
        Timer()
            : usert_last(0), is_log_file_open(false)
        {
            usert_before = usert_after = usert_begin = std::clock();
            syst_before = syst_after = syst_begin = std::chrono::high_resolution_clock::now();
        }

        Timer(const std::string& log_dir)
            : usert_last(0), is_log_file_open(true)
        {
            usert_before = usert_after = usert_begin = std::clock();
            syst_before = syst_after = syst_begin = std::chrono::high_resolution_clock::now();
            global_open_timer_log_file(log_dir);
            /*time_log.open(log_path);
            if (time_log.fail()) {
                std::cerr << "open failure error no: " << strerror(errno) << std::endl;
                exit(-1);
            }*/
        }

        ~Timer()
        {
            time_log.close();
        }

        inline void start()
        {
            usert_before = usert_after = usert_begin = std::clock();
            syst_before = syst_after = syst_begin = std::chrono::high_resolution_clock::now();
        }

        inline clock_t next_user_time_ticks()
        {
            usert_after = std::clock();
            clock_t delta_usert = usert_after - usert_before;
            usert_before = usert_after;
            return delta_usert;
        }

        inline std::chrono::duration<double> next_sys_time()
        {
            syst_after = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> delta_syst
                = std::chrono::duration_cast<std::chrono::duration<double>>(syst_after - syst_before);
            syst_before = syst_after;
            return delta_syst;

        }

        inline std::pair<double, double> next_time_secs(
            const std::string& text,
            int fill_len = 40)
        {
            double usert_secs = ((double)next_user_time_ticks()) / CLOCKS_PER_SEC;
            double syst_secs = next_sys_time().count();
            std::ostringstream ostr;
            ostr << "Time for " << std::setfill('.') << std::setw(fill_len) << std::left << text
                << usert_secs << "s(user)  " << syst_secs << "s(sys)" ;
            std::cout << ostr.str() << std::endl;
            if (is_log_file_open)	
                ISLE_LOG_TIMER(ostr.str());
            return std::make_pair(usert_secs, syst_secs);
        }

        inline std::pair<double, double> next_time_secs_silent()
        {
            return std::make_pair(((double)next_user_time_ticks()) / CLOCKS_PER_SEC, next_sys_time().count());
        }

        inline clock_t total_user_time_ticks()
        {
            usert_before = usert_after = usert_last = std::clock();
            auto usert_delta = usert_last - usert_begin;
            usert_begin = usert_last;
            return usert_delta;
        }

        inline std::chrono::duration<double> total_sys_time()
        {
            syst_before = syst_after = syst_last = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> delta_syst = std::chrono::duration_cast<std::chrono::duration<double>>(syst_last - syst_begin);
            syst_begin = syst_last;
            return delta_syst;

        }

        inline std::pair<double, double> total_time_secs(const std::string& text)
        {
            double user_secs = ((double)total_user_time_ticks()) / CLOCKS_PER_SEC;
            double sys_secs = total_sys_time().count();
            std::ostringstream ostr;
            ostr << "Total time for "
                << std::setfill('.') << std::setw(50) << std::left
                << text << user_secs << "s(user)  " << sys_secs << "s(secs)";
            std::cout << ostr.str() << std::endl;
            if (is_log_file_open)
                ISLE_LOG_TIMER(ostr.str());
            return std::make_pair(user_secs, sys_secs);
        }
    };
}
