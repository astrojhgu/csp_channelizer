all: main scan_freq test_filter test

NVCC=nvcc
HEADERS=channelizer.hpp kernels.h transpose.hpp types.hpp utils.hpp fir_coeffs.h
NVFLAGS=-O3 -rdc=true
CFLAGS=-I /usr/local/cuda/include -O3
OBJS=utils.o channelizer.o kernels.o fir_coeffs.o 
NVLIBS=-lcufft -lfftw3f -lcudart
LIBS=-L . -L /usr/local/cuda/lib64 -lcspch  -lfftw3f -lcudart -lcufft

cspch.o: $(OBJS)
	nvcc -dlink -o $@ $^ $(NVLIBS)

libcspch.a: cspch.o $(OBJS)
	ar crv $@ $^
	ranlib $@

test_filter: test_filter.o $(OBJS)
	nvcc $^ -o $@ $(NVLIBS)

test_filter.o: test_filter.cu $(HEADERS)
	nvcc -c -o $@ $< $(NVLIBS) $(NVFLAGS)

test: test.o $(OBJS)
	nvcc $^ -o $@ $(NVLIBS)

test.o: test.cu $(HEADERS)
	nvcc -c -o $@ $< $(NVFLAGS)


main: main.o libcspch.a
	g++ -o $@ $< $(LIBS)

main.o: main.cpp $(HEADERS)
	g++ -c $< -c -o $@ $(CFLAGS)

scan_freq: scan_freq.o libcspch.a
	g++ -o $@ $< $(LIBS)

scan_freq.o: scan_freq.cpp $(HEADERS)
	g++ -c $< -c -o $@ $(CFLAGS)

utils.o: utils.cu $(HEADERS)
	nvcc -c -o $@ $< $(NVFLAGS)


channelizer.o: channelizer.cu $(HEADERS)
	nvcc -c -o $@ $< $(NVFLAGS)

kernels.o: kernels.cu $(HEADERS)
	nvcc -c -o $@ $< $(NVFLAGS)

fir_coeffs.o: fir_coeffs.cpp $(HEADERS)
	g++ -c -o $@ $< -O3

clean:
	rm -f *.o main
