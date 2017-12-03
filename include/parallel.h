#pragma once

//
// Turn on to switch off parallelism
//
//#define NO_PAR

#if defined(NO_PAR)
#define pfor			    for
#define pfor_static_256		for
#define pfor_static_1024	for
#define pfor_static_131072	for
#define pfor_dynamic_1		for
#define pfor_dynamic_16		for
#define pfor_dynamic_512	for
#define pfor_dynamic_1024	for
#define pfor_dynamic_8192	for
#define pfor_dynamic_65536	for
#define pfor_dynamic_65536	for
#define pfor_dynamic_131072 for
#define cilk_spawn
#define cilk_sync

#else

// Parallelism is enabled.
// User Cilk Plus on Linux and OpenMP on Windows

#if defined(LINUX)
#include <cilk/cilk.h>
#define pfor			cilk_for
#define pfor_static_256		cilk_for
#define pfor_static_1024	cilk_for
#define pfor_static_131072	cilk_for
#define pfor_dynamic_1		cilk_for
#define pfor_dynamic_16		cilk_for
#define pfor_dynamic_512	cilk_for
#define pfor_dynamic_1024	cilk_for
#define pfor_dynamic_8192	cilk_for
#define pfor_dynamic_65536	cilk_for
#define pfor_dynamic_131072	cilk_for

#elif defined(_MSC_VER)
#include <omp.h>
#define pfor				__pragma(omp parallel for schedule(static, 1))		for
#define pfor_static_256		__pragma(omp parallel for schedule(static, 256))	for
#define pfor_static_1024	__pragma(omp parallel for schedule(static, 1024))	for
#define pfor_static_131072	__pragma(omp parallel for schedule(static, 131072))	for
#define pfor_dynamic_1		__pragma(omp parallel for schedule(dynamic, 1))		for
#define pfor_dynamic_16		__pragma(omp parallel for schedule(dynamic, 16))	for
#define pfor_dynamic_512	__pragma(omp parallel for schedule(dynamic, 512))	for
#define pfor_dynamic_1024	__pragma(omp parallel for schedule(dynamic, 1024))	for
#define pfor_dynamic_8192	__pragma(omp parallel for schedule(dynamic, 8292))	for
#define pfor_dynamic_65536	__pragma(omp parallel for schedule(dynamic, 65536))	for
#define pfor_dynamic_131072	__pragma(omp parallel for schedule(dynamic, 131072)) for
#define cilk_spawn
#define cilk_sync

#else
assert(false);
#endif

#endif
