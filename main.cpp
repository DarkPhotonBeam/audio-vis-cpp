#include <unsupported/Eigen/FFT>
#include <iostream>
#define DEBUG
#include "wave_io.h"
#include <unistd.h>
#define BAR_HEIGHT 64

int main(const int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
    return EXIT_FAILURE;
  }
  std::string path{argv[1]};
  try {
    WaveIO wave{path, 1<<12};
    wave.set_smoothing_factor(9.0f);

    auto callback = [](WaveIO *obj) {

      const Eigen::ArrayXf *fft_arr = obj->fft();
      Eigen::Index s = 120;
      Eigen::ArrayXf compressed_arr(2 * s);
      compressed_arr << fft_arr->head(s), fft_arr->tail(s);

      const std::size_t width = static_cast<std::size_t>(compressed_arr.size());
      std::vector<std::vector<char>> grid{width};
      for (std::size_t i = 0; i < width; ++i) {
        for (std::size_t j = 0; j < BAR_HEIGHT; ++j) {
          const float ratio = static_cast<float>(j) / BAR_HEIGHT;
          grid[i].push_back(ratio <= compressed_arr(i) ? '|' : ' ');
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
