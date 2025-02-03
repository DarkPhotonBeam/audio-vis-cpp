//
// Created by Daniel Leuthold on 2/2/25.
//

#ifndef WAVE_IO_H
#define WAVE_IO_H
#define DEBUG
#ifdef DEBUG
#define DBG_LOG(x) std::cout << x << std::endl
#else
#define DBG_LOG(x) do {} while (0)
#endif

#include <vector>
#include <cstdint>
#include <string>
#include <fstream>
#include <functional>
#include <Eigen/Dense>
#include <chrono>

#define AUDIO_FORMAT_PCM 1

class WaveIO {
private:
    std::vector<uint8_t> raw_bytes{};

    std::string &file_path;

    std::ifstream stream;

    uint16_t audio_format{};
    uint32_t sample_rate{};
    uint32_t byte_rate{};
    uint16_t block_align{};
    uint16_t num_channels{};
    uint16_t bits_per_sample{};
    uint32_t data_size{};
    std::size_t buffer_size{1<<10};
    bool fmt_read{false};

    std::chrono::time_point<std::chrono::system_clock> prev_time{};
    // Smaller value -> Less Smoothing
    float smoothing_factor{12.0f};

    bool fft_has_prev{false};



    Eigen::ArrayXf normalized_fft;
    Eigen::ArrayXf normalize_fft(const Eigen::VectorXcf &) const;
    void process_fmt_body();
    void process_fft(const void*, int);


public:
    explicit WaveIO(std::string &file_path, std::size_t buffer_size = 1024);

    void play();
    void play(std::function<void(const WaveIO*)> &&fn);

    // Returns normalized fft of current buffer
    const Eigen::ArrayXf *fft() const;

    [[nodiscard]] uint32_t get_sample_rate() const;
    [[nodiscard]] uint32_t get_byte_rate() const;
    [[nodiscard]] uint16_t get_block_align() const;
    [[nodiscard]] uint16_t get_num_channels() const;
    [[nodiscard]] uint16_t get_bits_per_sample() const;
    [[nodiscard]] uint32_t get_data_size() const;
    [[nodiscard]] std::size_t get_buffer_size() const;

    ~WaveIO();
};

#endif //WAVE_IO_H
