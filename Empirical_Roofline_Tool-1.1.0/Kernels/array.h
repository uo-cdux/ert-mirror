#ifndef ARRAY_H
#define ARRAY_H

#include <string>

#ifdef ERT_GPU
extern int gpu_blocks;
extern int gpu_threads;
#endif

#define KERNEL1(a,b,c)   ((a) = (b) + (c))
#define KERNEL2(a,b,c)   ((a) = (a)*(b) + (c))

namespace Info {
  void Error (std::string message, const char* str = __builtin_FUNCTION()) {
    std::string info = "Error: ";
    info += message;
    info += " in ";
    info += std::to_string(str);
    std::cout << info << std::endl;
    exit(1);
  }
}

template< typename T >
class Array {
private:
#ifdef ERT_GPU
  T *fBuffer;
#else
  T * __restrict__ fBuffer;
#endif

  void Alloc(uint64_t size);
  void Free();


public:
  Array (uint64_t size);
  ~Array ();
};

#endif
