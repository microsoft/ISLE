# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


CONFIG_FLAGS = -DSINGLE #-DNDEBUG

MKL_EIGEN_FLAGS = -DMKL_ILP64 #-DEIGEN_USE_MKL_ALL

LDFLAGS= -lm -ldl

INTEL_ROOT=/opt/intel/compilers_and_libraries/linux
MKL_ROOT=$(INTEL_ROOT)/mkl

MKL_COMMON_LDFLAGS=-L$(INTEL_ROOT)/lib/intel64 -L$(MKL_ROOT)/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 
MKL_SEQ_LDFLAGS = $(MKL_COMMON_LDFLAGS) -lmkl_sequential -lmkl_core
MKL_PAR_LDFLAGS = $(MKL_COMMON_LDFLAGS) -lmkl_intel_thread -lmkl_core -liomp5 -lpthread
MKL_PAR_STATIC_LDFLAGS = -Wl,--start-group $(MKL_ROOT)/lib/intel64/libmkl_intel_lp64.a $(MKL_ROOT)/lib/intel64/libmkl_intel_thread.a $(MKL_ROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl

CILK_LDFLAGS = -lcilkrts
CILK_FLAGS = -fcilkplus -DCILK

INCLUDE_DIR = include
SRC_DIR = src
DRIVER_DIR = drivers
ARMA_DIR = armadillo-8.400.0
BLOCK_KS_DIR = block-ks

CC=g++
OPTIM_FLAGS = -Ofast -march=native -mtune=native -fno-math-errno #-ffast-math - use at your own discretion
# DEBUG_FLAGS = -ggdb -DDEBUG
IFLAGS= -I . -I spectra-master -I $(INCLUDE_DIR) -I$(MKL_ROOT)/include -I$(ARMA_DIR)/include -I$(BLOCK_KS_DIR)
CFLAGS= -std=c++14 -DLINUX $(OPTIM_FLAGS) $(DEBUG_FLAGS) $(CONFIG_FLAGS) $(MKL_EIGEN_FLAGS) $(CILK_FLAGS)

INCLUDES =  $(INCLUDE_DIR)/timer.h $(INCLUDE_DIR)/log_utils.h \
	    $(INCLUDE_DIR)/mat_utils.h $(INCLUDE_DIR)/hyperparams.h \
	    $(INCLUDE_DIR)/sparse_matrix.h $(INCLUDE_DIR)/dense_matrix.h \
	    $(INCLUDE_DIR)/utils.h $(INCLUDE_DIR)/types.h \
	    $(INCLUDE_DIR)/trainer.h $(INCLUDE_DIR)/parallel.h \
	    $(INCLUDE_DIR)/infer.h $(INCLUDE_DIR)/logger.h
 
all: train infer format

trainer.o: $(SRC_DIR)/trainer.cpp $(INCLUDES)
	mkdir -p bin
	$(CC) -c -o bin/$@ $< $(IFLAGS) $(CFLAGS) $(MKL_PAR_LDFLAGS) $(CILK_LDFLAGS)

infer.o: $(SRC_DIR)/infer.cpp $(INCLUDES)
	mkdir -p bin
	$(CC) -c -o bin/$@ $< $(IFLAGS) $(CFLAGS) $(MKL_SEQ_LDFLAGS) $(CILK_LDFLAGS)

utils.o: $(SRC_DIR)/utils.cpp $(INCLUDES)
	mkdir -p bin
	$(CC) -c -o bin/$@ $< $(IFLAGS) $(CFLAGS) $(MKL_PAR_LDFLAGS) $(CILK_LDFLAGS)

dense_matrix.o : $(SRC_DIR)/dense_matrix.cpp $(INCLUDES)
	mkdir -p bin
	$(CC) -c -o bin/$@ $< $(IFLAGS) $(CFLAGS) $(MKL_PAR_LDFLAGS) $(CILK_LDFLAGS)

sparse_matrix.o : $(SRC_DIR)/sparse_matrix.cpp $(INCLUDES)
	mkdir -p bin
	$(CC) -c -o bin/$@ $< $(IFLAGS) $(CFLAGS) $(MKL_PAR_LDFLAGS) $(CILK_LDFLAGS)

logger.o : $(SRC_DIR)/logger.cpp $(INCLUDES)
	mkdir -p bin
	$(CC) -c -o bin/$@ $< $(IFLAGS) $(CFLAGS) $(MKL_PAR_LDFLAGS) $(CILK_LDFLAGS)

train:  $(DRIVER_DIR)/train.cpp trainer.o utils.o dense_matrix.o sparse_matrix.o logger.o $(INCLUDES)
	$(CC) -o bin/$@ $< bin/trainer.o bin/utils.o bin/dense_matrix.o bin/sparse_matrix.o bin/logger.o $(IFLAGS) $(CFLAGS) $(MKL_PAR_LDFLAGS) $(LDFLAGS) $(CILK_LDFLAGS)

infer: $(DRIVER_DIR)/infer.cpp infer.o utils.o dense_matrix.o sparse_matrix.o logger.o $(INCLUDES)
	$(CC) -o bin/$@ $< bin/infer.o bin/utils.o bin/dense_matrix.o bin/sparse_matrix.o bin/logger.o $(IFLAGS) $(CFLAGS) $(MKL_SEQ_LDFLAGS) $(LDFLAGS) $(CILK_LDFLAGS)

format:
	clang-format-4.0 -i include/*.h src/*.cpp drivers/*.cpp block-ks/*.h  

.PHONY: clean

clean: 
	rm -f bin/train bin/infer bin/*.o
