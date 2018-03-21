# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


CONFIG_FLAGS = -DSINGLE #-DNDEBUG

MKL_EIGEN_FLAGS = -DMKL_ILP64 #-DEIGEN_USE_MKL_ALL
#MKL_EIGEN_FLAGS = -DMKL_ILP64 -DEIGEN_USE_MKL_ALL

LDFLAGS= -lm -ldl

MKL_ROOT=/opt/intel/mkl
#MKL_ROOT=/opt/intel/compilers_and_libraries_2018/linux/mkl/

MKL_COMMON_LDFLAGS=-L $(MKL_ROOT)/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_core
MKL_SEQ_LDFLAGS = $(MKL_COMMON_LDFLAGS) -lmkl_sequential
MKL_PAR_LDFLAGS = $(MKL_COMMON_LDFLAGS) -lmkl_gnu_thread -lgomp -lpthread
MKL_PAR_STATIC_LDFLAGS = -Wl,--start-group /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a /opt/intel/mkl/lib/intel64/libmkl_gnu_thread.a /opt/intel/mkl/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl

CILK_LDFLAGS = -lcilkrts
CILK_FLAGS = -fcilkplus -DCILK

INCLUDE_DIR = include
SRC_DIR = src
DRIVER_DIR = drivers

CC=g++-5
IFLAGS= -I . -I spectra-master -I $(INCLUDE_DIR) -I$(MKL_ROOT)/include
CFLAGS= -ggdb -O3 -std=c++14 -DLINUX $(DEBUGGING_FLAGS) $(CONFIG_FLAGS) $(MKL_EIGEN_FLAGS) $(CILK_FLAGS)

INCLUDES =  $(INCLUDE_DIR)/timer.h $(INCLUDE_DIR)/logUtils.h \
	    $(INCLUDE_DIR)/matUtils.h $(INCLUDE_DIR)/hyperparams.h \
	    $(INCLUDE_DIR)/sparseMatrix.h $(INCLUDE_DIR)/denseMatrix.h \
	    $(INCLUDE_DIR)/utils.h $(INCLUDE_DIR)/types.h \
	    $(INCLUDE_DIR)/trainer.h $(INCLUDE_DIR)/parallel.h \
	    $(INCLUDE_DIR)/infer.h $(INCLUDE_DIR)/logger.h
 
all: ISLETrain ISLEInfer

trainer.o: $(SRC_DIR)/trainer.cpp $(INCLUDES)
	$(CC) -c -o $@ $< $(IFLAGS) $(CFLAGS) $(MKL_PAR_LDFLAGS) $(CILK_LDFLAGS)

infer.o: $(SRC_DIR)/infer.cpp $(INCLUDES)
	$(CC) -c -o $@ $< $(IFLAGS) $(CFLAGS) $(MKL_SEQ_LDFLAGS) $(CILK_LDFLAGS)

utils.o: $(SRC_DIR)/utils.cpp $(INCLUDES)
	$(CC) -c -o $@ $< $(IFLAGS) $(CFLAGS) $(MKL_PAR_LDFLAGS) $(CILK_LDFLAGS)

denseMatrix.o : $(SRC_DIR)/denseMatrix.cpp $(INCLUDES)
	$(CC) -c -o $@ $< $(IFLAGS) $(CFLAGS) $(MKL_PAR_LDFLAGS) $(CILK_LDFLAGS)

sparseMatrix.o : $(SRC_DIR)/sparseMatrix.cpp $(INCLUDES)
	$(CC) -c -o $@ $< $(IFLAGS) $(CFLAGS) $(MKL_PAR_LDFLAGS) $(CILK_LDFLAGS)

logger.o : $(SRC_DIR)/logger.cpp $(INCLUDES)
	$(CC) -c -o $@ $< $(IFLAGS) $(CFLAGS) $(MKL_PAR_LDFLAGS) $(CILK_LDFLAGS)

ISLETrain:  $(DRIVER_DIR)/ISLETrain.cpp trainer.o utils.o denseMatrix.o sparseMatrix.o logger.o $(INCLUDES)
	$(CC) -o $@ $< trainer.o utils.o denseMatrix.o sparseMatrix.o logger.o $(IFLAGS) $(CFLAGS) $(MKL_PAR_LDFLAGS) $(CILK_LDFLAGS)

ISLEInfer: $(DRIVER_DIR)/ISLEInfer.cpp infer.o utils.o denseMatrix.o sparseMatrix.o logger.o $(INCLUDES)
	$(CC) -o $@ $< infer.o utils.o denseMatrix.o sparseMatrix.o logger.o $(IFLAGS) $(CFLAGS) $(MKL_SEQ_LDFLAGS) $(CILK_LDFLAGS)

.PHONY: clean cleanest

clean: 
	rm -f *.o

cleanest: clean
	rm -f ISLETrain ISLEInfer *.o
