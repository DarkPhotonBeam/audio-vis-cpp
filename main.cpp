#include <unsupported/Eigen/FFT>
#include <iostream>
#define DEBUG
#include "wave_io.h"

#define BAR_HEIGHT 32

int main(const int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
    return EXIT_FAILURE;
  }
  std::string path{argv[1]};
  try {
    WaveIO wave{path, 1<<10};

    auto callback = [](const WaveIO *obj) {

      const Eigen::ArrayXf *left_norm = obj->fft();
      const std::size_t width = static_cast<std::size_t>(left_norm->size());
      std::vector<std::vector<char>> grid{width};
      for (std::size_t i = 0; i < width; ++i) {
        for (std::size_t j = 0; j < BAR_HEIGHT; ++j) {
          const float ratio = static_cast<float>(j) / BAR_HEIGHT;
          grid[i].push_back(ratio <= left_norm->operator()(i) ? 'x' : ' ');
        }
      }
      std::cout << "\033[H";
      for (std::size_t row = 0; row < BAR_HEIGHT; ++row) {
        for (std::size_t col = 0; col < width; ++col) {
          std::cout << grid[col][BAR_HEIGHT-row-1];
        }
        std::cout << std::endl;
      }
      std::cout.flush();

    };

    std::cout << "\033[2J\033[H";
    wave.play(callback);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return 0;
}
