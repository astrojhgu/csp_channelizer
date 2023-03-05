all: main scan_freq

NVCC=nvcc
HEADERS=channelizer.hpp kernels.h transpose.hpp types.hpp utils.hpp fir_coeffs.h
CFLAGS=-O3

main: main.o utils.o channelizer.o kernels.o fir_coeffs.o
	nvcc $^ -o $@ -lcufft -lfftw3f

scan_freq: scan_freq.o utils.o channelizer.o kernels.o fir_coeffs.o
	nvcc $^ -o $@ -lcufft -lfftw3f

main.o: main.cu $(HEADERS)
	nvcc -c -o $@ $< $(CFLAGS)

scan_freq.o: scan_freq.cu $(HEADERS)
	nvcc -c -o $@ $< $(CFLAGS)	

utils.o: utils.cu $(HEADERS)
	nvcc -c -o $@ $< $(CFLAGS)


channelizer.o: channelizer.cu $(HEADERS)
	nvcc -c -o $@ $< $(CFLAGS)

kernels.o: kernels.cu $(HEADERS)
	nvcc -c -o $@ $< $(CFLAGS)

fir_coeffs.o: fir_coeffs.cpp $(HEADERS)
	g++ -c -o $@ $< $(CFLAGS)

clean:
	rm -f *.o main