// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#define LOGGER
//#define VERBOSE

#include <iostream>
#include <fstream>
#include <cassert>
#include <string>

namespace ISLE
{
  // Loggin call back signature for TLC
  typedef void(*ChannelFunc)(const char*);

  class Logger
  {
    static int level;
    int level_;
    std::string fn;

    ChannelFunc trace_print_func;
    ChannelFunc info_print_func;
    ChannelFunc warning_print_func;
    ChannelFunc error_print_func;
    ChannelFunc timer_print_func;

  public:
    Logger();
    ~Logger();
    Logger(std::string fnName);

    void log_info(const std::string& msg, const std::string& fileName, const std::string& fnName, int lineNo);
    void log_trace(const std::string& msg, const std::string& fileName, const std::string& fnName, int lineNo);
    void log_warning(const std::string& msg, const std::string& fileName, const std::string& fnName, int lineNo);
    void log_error(const std::string& msg, const std::string& fileName, const std::string& fnName, int lineNo);

    void log_timer(const std::string& msg, const std::string& fileName, const std::string& fnName, int lineNo);

    void log_diagnostic(const std::string& msg, const std::string& fileName, const std::string& fnName, int lineNo);
    void log_diagnostic(const int& num, const std::string& num_name, const std::string& fileName, int lineNo);

    void set_trace_func(ChannelFunc func_) { trace_print_func = func_; }
    void set_info_func(ChannelFunc func_) { info_print_func = func_; }
    void set_warning_func(ChannelFunc func_) { warning_print_func = func_; }
    void set_error_func(ChannelFunc func_) { error_print_func = func_; }

    bool isTimerFileOpen;
    std::ofstream timerLogStream;
    bool openTimerLogFile(const std::string& outDir);

    bool isDiagnosticFileOpen;
    std::ofstream diagnosticLogStream;
    bool openDiagnosticLogFile(const std::string& outDir);
  };

#define ISLE_LOG_INFO(msg)		global_log_info		  (msg, __FILE__, __func__, __LINE__)
#define ISLE_LOG_TRACE(msg)		global_log_trace	  (msg, __FILE__, __func__, __LINE__)
#define ISLE_LOG_WARNING(msg)	global_log_warning	  (msg, __FILE__, __func__, __LINE__)
#define ISLE_LOG_ERROR(msg)		global_log_error	  (msg, __FILE__, __func__, __LINE__)

#define ISLE_LOG_DIAGNOSTIC(var)			 global_log_diagnostic(var, #var, __FILE__, __LINE__)
#define ISLE_LOG_DIAGNOSTIC_MSG(msg)		 global_log_diagnostic(msg, __FILE__, __func__, __LINE__)
#define ISLE_OPEN_DIAGNOSTIC_LOGFILE(dir) global_open_diagnostic_log_file(dir)

#define ISLE_LOG_TIMER(msg)          global_log_timer(msg, __FILE__, __func__, __LINE__)
#define ISLE_OPEN_TIMER_LOGFILE(dir) global_open_timer_log_file(dir)

#define NAME(var) (#var)

#define ISLE_LOG_SET_TRACE_FUNC(func_)	global_set_trace_func	(func_)
#define ISLE_LOG_SET_INFO_FUNC(func_)	global_set_info_func	(func_)
#define ISLE_LOG_SET_WARNING_FUNC(func_) global_set_warning_func (func_)
#define ISLE_LOG_SET_ERROR_FUNC(func_)	global_set_error_func	(func_)

  void global_log_info(const std::string& msg, const std::string& fileName, const std::string& fnName, int lineNo);
  void global_log_trace(const std::string& msg, const std::string& fileName, const std::string& fnName, int lineNo);
  void global_log_warning(const std::string& msg, const std::string& fileName, const std::string& fnName, int lineNo);
  void global_log_error(const std::string& msg, const std::string& fileName, const std::string& fnName, int lineNo);

  void global_log_timer(const std::string& msg, const std::string& fileName, const std::string& fnName, int lineNo);
  bool global_open_timer_log_file(const std::string& outDir);

  void global_log_diagnostic(const std::string& msg, const std::string& fileName, const std::string& fnName, int lineNo);
  void global_log_diagnostic(const int& num, const std::string& num_name, const std::string& fileName, int lineNo);
  bool global_open_diagnostic_log_file(const std::string& outDir);

  void global_set_trace_func(ChannelFunc func_);
  void global_set_info_func(ChannelFunc func_);
  void global_set_warning_func(ChannelFunc func_);
  void global_set_error_func(ChannelFunc func_);
}