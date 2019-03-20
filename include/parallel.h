// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

//
// Turn on to switch off parallelism
//
//#define NO_PAR

#define NUM_OMP_THR 28

#if defined(NO_PAR)
#define pfor			    for
#define pfor_static_256		for
#define pfor_static_1024	for
#define pfor_static_131072	for
#define pfor_dynamic_1		for
#define pfor_dynamic_16		for
#define pfor_dynamic_256	for
#define pfor_dynamic_512	for
#define pfor_dynamic_1024	for
#define pfor_dynamic_8192	for
#define pfor_dynamic_65536	for
#define pfor_dynamic_65536	for
#define pfor_dynamic_131072 for
#define cilk_spawn
#define cilk_sync

#define parallel_sort std::sort

// Parallelism is enabled.
// User Cilk Plus on Linux and OpenMP on Windows

#elif defined(LINUX)


#ifdef CILK

#include <cilk/cilk.h>

#define pfor			cilk_for
#define pfor_static_256		cilk_for
#define pfor_static_1024	cilk_for
#define pfor_static_131072	cilk_for
#define pfor_dynamic_1		cilk_for
#define pfor_dynamic_16		cilk_for
#define pfor_dynamic_256	cilk_for
#define pfor_dynamic_512	cilk_for
#define pfor_dynamic_1024	cilk_for
#define pfor_dynamic_8192	cilk_for
#define pfor_dynamic_65536	cilk_for
#define pfor_dynamic_131072	cilk_for

#endif

#ifndef CILK
#include <omp.h>

#define pfor			    _Pragma("omp parallel for schedule(static, 1) num_threads(NUM_OMP_THR)")		for
#define pfor_static_256		_Pragma("omp parallel for schedule(static, 256) num_threads(NUM_OMP_THR)")	    for
#define pfor_static_1024	_Pragma("omp parallel for schedule(static, 1024) num_threads(NUM_OMP_THR)")	    for
#define pfor_static_131072	_Pragma("omp parallel for schedule(static, 131072) num_threads(NUM_OMP_THR)")	for
#define pfor_dynamic_1		_Pragma("omp parallel for schedule(dynamic, 1) num_threads(NUM_OMP_THR)")		for
#define pfor_dynamic_16		_Pragma("omp parallel for schedule(dynamic, 16) num_threads(NUM_OMP_THR)")	    for
#define pfor_dynamic_256	_Pragma("omp parallel for schedule(dynamic, 256) num_threads(NUM_OMP_THR)")	    for
#define pfor_dynamic_512	_Pragma("omp parallel for schedule(dynamic, 512) num_threads(NUM_OMP_THR)")	    for
#define pfor_dynamic_1024	_Pragma("omp parallel for schedule(dynamic, 1024) num_threads(NUM_OMP_THR)")	for
#define pfor_dynamic_8192	_Pragma("omp parallel for schedule(dynamic, 8292) num_threads(NUM_OMP_THR)")	for
#define pfor_dynamic_65536	_Pragma("omp parallel for schedule(dynamic, 65536) num_threads(NUM_OMP_THR)")	for
#define pfor_dynamic_131072	_Pragma("omp parallel for schedule(dynamic, 131072) num_threads(NUM_OMP_THR)")  for
#define cilk_spawn
#define cilk_sync

#endif

#include <parallel/algorithm>

#define parallel_sort __gnu_parallel::sort

#elif defined(_MSC_VER)
#include <omp.h>
#define pfor				__pragma(omp parallel for schedule(static, 1) num_threads(NUM_OMP_THR))		    for
#define pfor_static_256		__pragma(omp parallel for schedule(static, 256) num_threads(NUM_OMP_THR))       for
#define pfor_static_1024	__pragma(omp parallel for schedule(static, 1024) num_threads(NUM_OMP_THR))	    for
#define pfor_static_131072	__pragma(omp parallel for schedule(static, 131072) num_threads(NUM_OMP_THR))    for
#define pfor_dynamic_1		__pragma(omp parallel for schedule(dynamic, 1) num_threads(NUM_OMP_THR))		for
#define pfor_dynamic_16		__pragma(omp parallel for schedule(dynamic, 16) num_threads(NUM_OMP_THR))	    for
#define pfor_dynamic_256	__pragma(omp parallel for schedule(dynamic, 256) num_threads(NUM_OMP_THR))	    for
#define pfor_dynamic_512	__pragma(omp parallel for schedule(dynamic, 512) num_threads(NUM_OMP_THR))	    for
#define pfor_dynamic_1024	__pragma(omp parallel for schedule(dynamic, 1024) num_threads(NUM_OMP_THR))	    for
#define pfor_dynamic_8192	__pragma(omp parallel for schedule(dynamic, 8292) num_threads(NUM_OMP_THR))	    for
#define pfor_dynamic_65536	__pragma(omp parallel for schedule(dynamic, 65536) num_threads(NUM_OMP_THR))	for
#define pfor_dynamic_131072	__pragma(omp parallel for schedule(dynamic, 131072) num_threads(NUM_OMP_THR))   for
#define cilk_spawn
#define cilk_sync

#define parallel_sort std::sort

#else
static_assert(false);
#endif
