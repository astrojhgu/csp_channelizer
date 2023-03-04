#include "utils.hpp"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace std;

#include "transpose.hpp"
int main() {
  using T = float;
  int nrows = 88;
  int ncols = 937500;
  auto d_data_in = gpu_alloc<T>(ncols * nrows);
  auto d_data_out = gpu_alloc<T>(ncols * nrows);
  vector<T> data_in(nrows * ncols);
  vector<T> data_out(nrows * ncols);

  std::cerr << "filling" << std::endl;
  for (int i = 0; i < nrows * ncols; ++i) {
    data_in[i] = i;
  }

  /*
  for (int i = 0; i < nrows; ++i) {
    for (int j = 0; j < ncols; ++j) {
      cout << data_in[i * ncols + j] << " ";
    }
    cout << endl;
  }*/

  // while (1)
  for (int i = 0; i < 100; ++i)
  {
    std::cout << i << std::endl;
    transpose(data_in.data(), data_out.data(), nrows, ncols);
  }

  std::cerr << "validating" << std::endl;
  for (int j = 0; j < nrows; ++j) {
    for (int i = 0; i < ncols; ++i) {
      assert(data_out[i * nrows + j] == data_in[j * ncols + i]);
    }
  }
}