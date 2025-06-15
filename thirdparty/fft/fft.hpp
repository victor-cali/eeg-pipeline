/* * Free-to-use C++ FFT library
 * Copyright (c) 2016 Project Nayuki. (MIT License)
 * https://www.nayuki.io/page/free-small-fft-in-multiple-languages
 */

 #pragma once

 #include <complex>
 #include <vector>
 
 namespace Fft {
     void transform(std::vector<std::complex<double> > &vec);
     void inverseTransform(std::vector<std::complex<double> > &vec);
 }
 