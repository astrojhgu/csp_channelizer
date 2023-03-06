all: main scan_freq test_filter test

NVCC=nvcc
HEADERS=channelizer.hpp kernels.h transpose.hpp types.hpp utils.hpp fir_coeffs.h
CFLAGS=-O3 -Xcompiler -fopenmp

test_filter: test_filter.o utils.o channelizer.o kernels.o fir_coeffs.o
	nvcc $^ -o $@ -lcufft -lfftw3f $(CFLAGS)

test_filter.o: test_filter.cu $(HEADERS)
	nvcc -c -o $@ $< $(CFLAGS)

test: test.o utils.o channelizer.o kernels.o fir_coeffs.o
	nvcc $^ -o $@ -lcufft -lfftw3f $(CFLAGS)

test.o: test.cu $(HEADERS)
	nvcc -c -o $@ $< $(CFLAGS)


main: main.o utils.o channelizer.o kernels.o fir_coeffs.o
	nvcc $^ -o $@ -lcufft -lfftw3f $(CFLAGS)

main.o: main.cu $(HEADERS)
	nvcc -c -o $@ $< $(CFLAGS)

scan_freq: scan_freq.o utils.o channelizer.o kernels.o fir_coeffs.o
	nvcc $^ -o $@ -lcufft -lfftw3f $(CFLAGS)

scan_freq.o: scan_freq.cu $(HEADERS)
	nvcc -c -o $@ $< $(CFLAGS)	

utils.o: utils.cu $(HEADERS)
	nvcc -c -o $@ $< $(CFLAGS)


channelizer.o: channelizer.cu $(HEADERS)
	nvcc -c -o $@ $< $(CFLAGS)

kernels.o: kernels.cu $(HEADERS)
	nvcc -c -o $@ $< $(CFLAGS)

fir_coeffs.o: fir_coeffs.cpp $(HEADERS)
	g++ -c -o $@ $< -O3

clean:
	rm -f *.o main
