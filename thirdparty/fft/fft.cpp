/* * Free-to-use C++ FFT library
 * Copyright (c) 2016 Project Nayuki. (MIT License)
 * https://www.nayuki.io/page/free-small-fft-in-multiple-languages
 */

 #include <algorithm>
 #include <cmath>
 #include "fft.hpp"
 
 using std::complex;
 using std::size_t;
 using std::vector;
 
 static size_t reverseBits(size_t x, int n) {
     size_t result = 0;
     for (int i = 0; i < n; i++, x >>= 1)
         result = (result << 1) | (x & 1);
     return result;
 }
 
 void Fft::transform(vector<complex<double> > &vec) {
     size_t n = vec.size();
     if (n == 0)
         return;
 
     int levels = 0;
     while (static_cast<size_t>(1U) << levels < n)
         levels++;
     if (static_cast<size_t>(1U) << levels != n)
         throw "Length is not a power of 2";
 
     vector<complex<double> > temp(n);
     for (size_t i = 0; i < n; i++)
         temp[i] = vec[reverseBits(i, levels)];
     vec = temp;
     
     for (size_t size = 2; size <= n; size *= 2) {
         for (size_t i = 0; i < n; i += size) {
             for (size_t j = 0; j < size / 2; j++) {
                 double t = 2 * M_PI * j / size;
                 complex<double> exp(cos(t), -sin(t));
                 complex<double> a = vec[i + j];
                 complex<double> b = vec[i + j + size / 2] * exp;
                 vec[i + j] = a + b;
                 vec[i + j + size / 2] = a - b;
             }
         }
     }
 }
 
 void Fft::inverseTransform(vector<complex<double> > &vec) {
     for (complex<double> &x : vec)
         x = conj(x);
     transform(vec);
     for (complex<double> &x : vec)
         x = conj(x) / static_cast<double>(vec.size());
 }
 