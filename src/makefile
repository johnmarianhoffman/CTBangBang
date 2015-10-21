NVCC=nvcc
CC=gcc

ODIR=obj

IDIR_RECON=../include/
IDIR_READER=/home/john/Public/reader_cuda/
INCLUDES=-I$(IDIR_RECON) -I$(IDIR_READER)

LIBS=$(IDIR_READER)read_raw_file.o -lfftw3 -lm

CFLAGS=$(INCLUDES) -L$(IDIR_READER) -g

_OBJ=main.o setup.o interp.o rebin_filter.o rebin_filter_cpu.o backproject.o backproject_cpu.o
OBJ=$(patsubst %,$(ODIR)/%,$(_OBJ));

all: ../ctbangbang

../ctbangbang: $(OBJ)
	(cd obj && $(NVCC) -L$(IDIR_READER) -o ../../ctbangbang $(IDIR_READER)read_raw_file.o *.o -lfftw3)

obj/main.o: main.cu $(IDIR_RECON)recon_structs.h $(IDIR_RECON)setup.h $(IDIR_RECON)rebin_filter.h $(IDIR_RECON)backproject.h
	$(NVCC) $(CFLAGS) -c -o $@ $<

obj/setup.o: setup.cu $(IDIR_READER)read_raw_file.h $(IDIR_RECON)setup.h
	$(NVCC) $(CFLAGS) -c -o $@ $<

obj/interp.o: interp.cu $(IDIR_RECON)interp.h
	$(NVCC) $(CFLAGS) -c -o $@ $<

obj/rebin_filter.o: rebin_filter.cu $(IDIR_RECON)rebin_filter.h $(IDIR_RECON)rebin_filter.cuh 
	$(NVCC) $(CFLAGS) -c -o $@ $<

obj/rebin_filter_cpu.o: rebin_filter_cpu.cu $(IDIR_RECON)rebin_filter_cpu.h 
	$(NVCC) $(CFLAGS) -c -o $@ $<

obj/backproject.o: backproject.cu $(IDIR_RECON)backproject.cuh
	$(NVCC) $(CFLAGS) -c -o $@ $<

obj/backproject_cpu.o: backproject_cpu.cu $(IDIR_RECON)backproject_cpu.h
	$(NVCC) $(CFLAGS) -c -o $@ $<

.PHONY: all test clean

clean:
	rm obj/*.o; rm -rf ../*~

