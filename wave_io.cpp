//
// Created by Daniel Leuthold on 2/2/25.
//

#include "wave_io.h"
#include <cstdint>
#include <iostream>
#include <cmath>
#include <matplot/matplot.h>
#include <alsa/asoundlib.h>
#include <unsupported/Eigen/FFT>

WaveIO::WaveIO(std::string &file_path, const std::size_t buffer_size): file_path(file_path), buffer_size(buffer_size) {
    if ((buffer_size & (buffer_size - 1))) {
        throw std::invalid_argument("buffer_size must be a power of 2");
    }

    stream = std::ifstream(file_path, std::ios::binary);
    if (!stream.is_open()) {
        throw std::runtime_error("Failed to open file " + file_path);
    }

    // Read Initial RIFF CHUNK to check if this is indeed a RIFF file...
    stream.seekg(0, std::ios::beg);
    std::string head_id(4, '\0');
    stream.read(&head_id[0], 4);

    DBG_LOG(head_id);

    if (head_id != "RIFF") {
        throw std::runtime_error("Invalid file type. Must be a RIFF file.");
    }

    uint32_t file_size;
    stream.read(reinterpret_cast<char *>(&file_size), sizeof(file_size));
    DBG_LOG("File size: " << (file_size/1024) << "kB");

    std::string riff_type_id(4, '\0');
    stream.read(&riff_type_id[0], 4);
    DBG_LOG("RIFF FILE TYPE: " << riff_type_id);

    std::string chunk_id(4, '\0');
    while (chunk_id != "DATA" && chunk_id != "data" && chunk_id != "END ") {
        stream.read(&chunk_id[0], 4);
        DBG_LOG(chunk_id);
        uint32_t chunk_size;
        stream.read(reinterpret_cast<char *>(&chunk_size), sizeof(chunk_size));
        DBG_LOG("Chunk size: " << (chunk_size) << " bytes");
        if (chunk_id == "fmt ") process_fmt_body();
        else if (chunk_id == "data") {
            data_size = chunk_size;
            raw_bytes.resize(data_size);
            stream.read(reinterpret_cast<char *>(raw_bytes.data()), data_size);
        }
        else stream.seekg(chunk_size, std::ios::cur);
    }
}

void WaveIO::play() {
    play([](const WaveIO *obj){});
}

void WaveIO::play(std::function<void(const WaveIO*)> &&fn) {
    snd_pcm_t *pcm_handle;
    snd_pcm_hw_params_t *params;
    snd_pcm_open(&pcm_handle, "default", SND_PCM_STREAM_PLAYBACK, 0);
    snd_pcm_hw_params_malloc(&params);
    snd_pcm_hw_params_any(pcm_handle, params);

    snd_pcm_hw_params_set_access(pcm_handle, params, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_format_t format;

    if (bits_per_sample == 8) {
        format = SND_PCM_FORMAT_U8;
        DBG_LOG("Format: 8-bit unsigned integer");
    } else if (bits_per_sample == 16) {
        format = SND_PCM_FORMAT_S16_LE;
        DBG_LOG("Format: 16-bit signed integer");
    } else if (bits_per_sample == 24) {
        format = SND_PCM_FORMAT_S24_3LE;
        DBG_LOG("Format: 24-bit unsigned integer");
    } else if (bits_per_sample == 32) {
        format = SND_PCM_FORMAT_S32_LE;
        DBG_LOG("Format: 32-bit signed integer");
    } else {
        throw std::runtime_error("Unsupported bits_per_sample");
    }

    snd_pcm_hw_params_set_format(pcm_handle, params, format);
    snd_pcm_hw_params_set_channels(pcm_handle, params, num_channels);
    snd_pcm_hw_params_set_rate(pcm_handle, params, sample_rate, 0);
    snd_pcm_hw_params(pcm_handle, params);
    snd_pcm_hw_params_free(params);
    snd_pcm_prepare(pcm_handle);

    size_t offset = 0;

    while (offset < raw_bytes.size()) {
        const size_t chunk_size = std::min(buffer_size, raw_bytes.size() - offset);
        const int frames = chunk_size / block_align;

        // FFT PROCESSING
        process_fft(raw_bytes.data() + offset, frames);

        fn(this);

        if (const ssize_t result = snd_pcm_writei(pcm_handle, raw_bytes.data() + offset, frames); result == -EPIPE) {
            snd_pcm_prepare(pcm_handle);
        } else if (result == -EAGAIN) {
            snd_pcm_wait(pcm_handle, 1000);
        }

        offset += chunk_size;
    }

    snd_pcm_drain(pcm_handle);
    snd_pcm_close(pcm_handle);
}

Eigen::ArrayXf WaveIO::normalize_fft(const Eigen::VectorXcf &v) const {
    Eigen::VectorXf v_log(v.size());
    for (Eigen::Index i = 0; i < v.size(); ++i) {
        v_log(i) = 20. * std::log(std::abs(v(i)) + 1e-10);
    }

    float min_abs = std::numeric_limits<float>::max();
    float max_abs = std::numeric_limits<float>::min();

    for (Eigen::Index i = 0; i < v_log.size(); i++) {
        float abs = std::abs(v_log(i));
        min_abs = std::min(min_abs, abs);
        max_abs = std::max(max_abs, abs);
    }

    const float range = max_abs - min_abs;
    Eigen::ArrayXf out(v_log.size());
    out = ((v_log.array() - Eigen::ArrayXf::Constant(v_log.size(), min_abs)) / range).abs();

    return out;
}

const Eigen::ArrayXf *WaveIO::fft() const {
    return &normalized_fft;
}

void WaveIO::process_fmt_body() {
    if (fmt_read) {
        throw std::runtime_error("process_fmt_body called twice");
    }
    stream.read(reinterpret_cast<char *>(&audio_format), sizeof(audio_format));
    stream.read(reinterpret_cast<char *>(&num_channels), sizeof(num_channels));
    stream.read(reinterpret_cast<char *>(&sample_rate), sizeof(sample_rate));
    stream.read(reinterpret_cast<char *>(&byte_rate), sizeof(byte_rate));
    stream.read(reinterpret_cast<char *>(&block_align), sizeof(block_align));
    stream.read(reinterpret_cast<char *>(&bits_per_sample), sizeof(bits_per_sample));
    DBG_LOG("\tAudio Format: " << (audio_format == AUDIO_FORMAT_PCM ? "PCM" : "Other"));
    DBG_LOG("\tNum Channels: " << (num_channels));
    DBG_LOG("\tSample Rate: " << (sample_rate));
    DBG_LOG("\tByte Rate: " << (byte_rate));
    DBG_LOG("\tBlock Align: " << (block_align));
    DBG_LOG("\tBits Per Sample: " << (bits_per_sample));
    if (audio_format != AUDIO_FORMAT_PCM) {
        throw std::runtime_error("Non-PCM audio format not supported.");
    }
    if (bits_per_sample != 16) {
        throw std::runtime_error("Unsupported bits_per_sample (only 16-bit supported)");
    }
    if (num_channels != 2) {
        throw std::runtime_error("Audio file must be stereo");
    }
    fmt_read = true;
}

void WaveIO::process_fft(const void *ptr, int frames) {
    Eigen::FFT<float> fft;
    Eigen::VectorXf left_samples(frames);
    Eigen::VectorXf right_samples(frames);
    const short *s_ptr = static_cast<const short *>(ptr);
    for (int i = 0; i < frames; ++i) {
        const short left = s_ptr[2*i];
        const short right = s_ptr[2*i+1];
        const float left_d = left;
        const float right_d = right;
        left_samples(i) = left_d;
        right_samples(i) = right_d;
        //printf("%d, %d -> %f, %f\n", left, right, left_d, right_d);
    }
    Eigen::VectorXcf left_fft = fft.fwd(left_samples);
    const Eigen::VectorXcf right_fft = fft.fwd(right_samples);
    const Eigen::Index half = left_fft.size() / 2;
    left_fft.tail(half) = right_fft.tail(half);

    // remove some high frequencies
    Eigen::Index amount = 100;
    Eigen::VectorXcf stripped_fft(left_fft.size() - amount);
    stripped_fft << left_fft.head(half - amount/2), left_fft.tail(half - amount / 2);

    Eigen::ArrayXf normed_fft = normalize_fft(stripped_fft);
    auto now = std::chrono::high_resolution_clock::now();
    if (!fft_has_prev) {
        normalized_fft = normed_fft;
        fft_has_prev = true;
    } else {
        const std::chrono::duration<float> diff = now - prev_time;
        const float t = 1.0f - std::exp(-smoothing_factor * diff.count());
        const Eigen::ArrayXf old = normalized_fft;
        for (Eigen::Index i = 0; i < normalized_fft.size(); ++i) {
            normalized_fft(i) = old(i) + t * (normed_fft(i) - old(i));
        }
    }
    prev_time = now;
}

uint32_t WaveIO::get_sample_rate() const {
    return sample_rate;
}

uint32_t WaveIO::get_byte_rate() const {
    return byte_rate;
}

uint16_t WaveIO::get_block_align() const {
    return block_align;
}

uint16_t WaveIO::get_num_channels() const {
    return num_channels;
}

uint16_t WaveIO::get_bits_per_sample() const {
    return bits_per_sample;
}

uint32_t WaveIO::get_data_size() const {
    return data_size;
}

std::size_t WaveIO::get_buffer_size() const {
    return buffer_size;
}

WaveIO::~WaveIO() {
    if (stream.is_open()) stream.close();
}
