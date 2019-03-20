# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

DEBUGGING_FLAGS= #-DNDEBUG
MKL_EIGEN_FLAGS =  #-DEIGEN_USE_MKL_ALL
CILK_FLAGS = #-fcilkplus -DCILK
# Use -DCILK above if you are certain, Mixing Cilk + Intel MKL/OpenMP causes problems
CONFIG_FLAGS = -DLINUX -DMKL_ILP64 -DSINGLE $(DEBUGGING_FLAGS) $(MKL_EIGEN_FLAGS) $(CILK_FLAGS)

INTEL_ROOT=/opt/intel/compilers_and_libraries/linux
MKL_ROOT=$(INTEL_ROOT)/mkl

MKL_COMMON_LDFLAGS=-L$(INTEL_ROOT)/lib/intel64 -L$(MKL_ROOT)/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 
MKL_SEQ_LDFLAGS = $(MKL_COMMON_LDFLAGS) -lmkl_sequential -lmkl_core -lm -ldl
MKL_PAR_LDFLAGS = $(MKL_COMMON_LDFLAGS) -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
MKL_PAR_STATIC_LDFLAGS = -L$(INTEL_ROOT)/lib/intel64 -Wl,--start-group $(MKL_ROOT)/lib/intel64/libmkl_intel_ilp64.a $(MKL_ROOT)/lib/intel64/libmkl_intel_thread.a $(MKL_ROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl


CILK_LDFLAGS = #-lcilkrts
# Use Cilk above if you are certain, Mixing Cilk + Intel MKL/OpenMP causes problems

INCLUDE_DIR = include
SRC_DIR = src
DRIVER_DIR = drivers
SPECTRA_DIR = spectra-master
ARMA_DIR = armadillo-8.400.0
BLOCK_KS_DIR = block-ks

CC=g++
IFLAGS= -I. -I$(SPECTRA_DIR) -I$(ARMA_DIR)/include  -I$(BLOCK_KS_DIR)  -I$(MKL_ROOT)/include  -I $(INCLUDE_DIR)
CFLAGS= -g -w -O3 -fopenmp  -std=c++14

INCLUDES =  $(INCLUDE_DIR)/timer.h $(INCLUDE_DIR)/logUtils.h \
	    $(INCLUDE_DIR)/matUtils.h $(INCLUDE_DIR)/hyperparams.h \
	    $(INCLUDE_DIR)/sparseMatrix.h $(INCLUDE_DIR)/denseMatrix.h \
	    $(INCLUDE_DIR)/utils.h $(INCLUDE_DIR)/types.h \
	    $(INCLUDE_DIR)/trainer.h $(INCLUDE_DIR)/parallel.h \
	    $(INCLUDE_DIR)/infer.h $(INCLUDE_DIR)/logger.h
 
all: ISLETrain ISLEInfer

trainer.o: $(SRC_DIR)/trainer.cpp $(INCLUDES)
	$(CC) $(CONFIG_FLAGS) $(IFLAGS) $(CFLAGS) -c -o $@ $< 

infer.o: $(SRC_DIR)/infer.cpp $(INCLUDES)
	$(CC) $(CONFIG_FLAGS)  $(IFLAGS) $(CFLAGS) -c -o $@ $< 

utils.o: $(SRC_DIR)/utils.cpp $(INCLUDES)
	$(CC) $(CONFIG_FLAGS) $(IFLAGS) $(CFLAGS) -c -o $@ $< 

denseMatrix.o : $(SRC_DIR)/denseMatrix.cpp $(INCLUDES)
	$(CC) $(CONFIG_FLAGS) $(IFLAGS) $(CFLAGS) -c -o $@ $< 

sparseMatrix.o : $(SRC_DIR)/sparseMatrix.cpp $(INCLUDES)
	$(CC) $(CONFIG_FLAGS) $(IFLAGS) $(CFLAGS) -c -o $@ $< 

logger.o : $(SRC_DIR)/logger.cpp $(INCLUDES)
	$(CC) $(CONFIG_FLAGS) $(IFLAGS) $(CFLAGS) -c -o $@ $< 

ISLETrain:  $(DRIVER_DIR)/ISLETrain.cpp trainer.o utils.o denseMatrix.o sparseMatrix.o logger.o $(INCLUDES)
	$(CC) $(CONFIG_FLAGS) $(IFLAGS) $(CFLAGS) -o $@ $< trainer.o utils.o denseMatrix.o sparseMatrix.o logger.o $(MKL_PAR_LDFLAGS) $(CILK_LDFLAGS)

ISLEInfer: $(DRIVER_DIR)/ISLEInfer.cpp infer.o utils.o denseMatrix.o sparseMatrix.o logger.o $(INCLUDES)
	$(CC) $(CONFIG_FLAGS) $(IFLAGS) $(CFLAGS) -o $@ $< infer.o utils.o denseMatrix.o sparseMatrix.o logger.o $(MKL_SEQ_LDFLAGS) $(CILK_LDFLAGS)

.PHONY: clean cleanest

clean: 
	rm -f *.o

cleanest: clean
	rm -f ISLETrain ISLEInfer *.o
