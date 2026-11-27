//
// Created by Damien Ronssin on 09.03.23.
//

#include "AudioUtils.h"

#include <algorithm>
#include <cmath>
#include <limits>

// Declarations only: the implementation lives in the minimp3 target, so its
// warnings do not land on this file. See ThirdParty/minimp3/CMakeLists.txt.
#include "minimp3.h"
#include "minimp3_ex.h"

namespace AudioUtils
{

namespace
{
constexpr int MAX_AUDIO_CHANNELS = 64;
constexpr double MIN_AUDIO_SAMPLE_RATE = 8000.0;
constexpr double MAX_AUDIO_SAMPLE_RATE = 768000.0;
// 128M float samples (~512 MB, ~23 minutes of stereo 48 kHz audio) and 30 minutes max duration:
// prevents integer truncation, upsampling overflow, and multi-GB allocations from malformed or huge files.
constexpr double MAX_AUDIO_DURATION_SECONDS = 1800.0;
constexpr juce::int64 MAX_TOTAL_AUDIO_SAMPLES = 128LL * 1024LL * 1024LL;
constexpr juce::int64 MAX_MP3_FILE_BYTES = 256LL * 1024LL * 1024LL;
} // namespace

bool loadAudioFile(const juce::File& inFile, AudioBuffer<float>& outBuffer, double& outSampleRate)
{
    if (inFile.hasFileExtension(".mp3")) {
        return _loadMP3File(inFile, outBuffer, outSampleRate);
    }

    // Register different audio formats
    auto audio_format_manager = createAudioFormatManager();

    std::unique_ptr<juce::AudioFormatReader> format_reader;
    format_reader.reset(audio_format_manager->createReaderFor(inFile));

    // Verify format reader is not null
    if (!format_reader)
        return false;

    const double sample_rate = format_reader->sampleRate;
    const unsigned int raw_channels = format_reader->numChannels;
    const juce::int64 raw_samples = format_reader->lengthInSamples;

    if (!std::isfinite(sample_rate) || sample_rate < MIN_AUDIO_SAMPLE_RATE || sample_rate > MAX_AUDIO_SAMPLE_RATE) {
        return false;
    }

    if (raw_channels == 0 || raw_channels > static_cast<unsigned int>(MAX_AUDIO_CHANNELS)) {
        return false;
    }

    if (raw_samples <= 0 || raw_samples > std::numeric_limits<int>::max()
        || raw_samples > MAX_TOTAL_AUDIO_SAMPLES / static_cast<juce::int64>(raw_channels)
        || static_cast<double>(raw_samples) / sample_rate > MAX_AUDIO_DURATION_SECONDS) {
        return false;
    }

    // Get properties of input audio
    outSampleRate = sample_rate;
    const int num_source_samples = static_cast<int>(raw_samples);
    const int num_channels = static_cast<int>(raw_channels);

    outBuffer.setSize(num_channels, num_source_samples);

    // Read source file. If not successful, return false
    if (!format_reader->read(&outBuffer, 0, num_source_samples, 0, true, true)) {
        outBuffer.setSize(0, 0);
        return false;
    }

    return true;
}

StringArray getSupportedAudioFileExtensions()
{
    StringArray supported_extensions;
    supported_extensions.add(".mp3");

    auto audio_format_manager = createAudioFormatManager();
    for (auto& format: *audio_format_manager) {
        StringArray file_extensions = format->getFileExtensions();
        for (auto& extension: file_extensions) {
            supported_extensions.add(extension);
        }
    }

    return supported_extensions;
}

std::unique_ptr<AudioFormatManager> createAudioFormatManager()
{
    auto audio_format_manager = std::make_unique<AudioFormatManager>();
    audio_format_manager->registerFormat(new juce::WavAudioFormat, true);
    audio_format_manager->registerFormat(new juce::AiffAudioFormat, false);
    audio_format_manager->registerFormat(new juce::FlacAudioFormat, false);
    audio_format_manager->registerFormat(new juce::OggVorbisAudioFormat, false);
#if JUCE_MAC
    audio_format_manager->registerFormat(new juce::CoreAudioFormat(juce::CoreAudioFormat::StreamKind::kM4a), false);
#endif

    return audio_format_manager;
}

void resampleBuffer(const AudioBuffer<float>& inBuffer,
                    AudioBuffer<float>& outBuffer,
                    double inSourceSampleRate,
                    double inTargetSampleRate)
{
    const int num_channels = inBuffer.getNumChannels();
    const int num_samples = inBuffer.getNumSamples();

    if (num_channels <= 0 || num_samples <= 0 || !std::isfinite(inSourceSampleRate) || inSourceSampleRate <= 0.0
        || !std::isfinite(inTargetSampleRate) || inTargetSampleRate <= 0.0) {
        outBuffer.setSize(0, 0);
        return;
    }

    if (juce::exactlyEqual(inSourceSampleRate, inTargetSampleRate)) {
        outBuffer.makeCopyOf(inBuffer);
        return;
    }

    const double speed_ratio = inSourceSampleRate / inTargetSampleRate;
    const double estimated_out =
        (static_cast<double>(num_samples) + static_cast<double>(LagrangeInterpolator::getBaseLatency())) / speed_ratio;

    if (!std::isfinite(estimated_out) || estimated_out <= 0.0
        || estimated_out > static_cast<double>(std::numeric_limits<int>::max())
        || estimated_out * static_cast<double>(num_channels) > static_cast<double>(MAX_TOTAL_AUDIO_SAMPLES)) {
        outBuffer.setSize(0, 0);
        return;
    }

    Resampler resampler;
    // Prepare resampler
    resampler.prepareToPlay(inSourceSampleRate, num_samples, inTargetSampleRate);
    auto num_expected_samples_after_resample = resampler.getNumOutSamplesOnNextProcessBlock(num_samples);

    if (num_expected_samples_after_resample <= 0) {
        outBuffer.setSize(0, 0);
        return;
    }

    outBuffer.setSize(num_channels, num_expected_samples_after_resample);

    for (int ch = 0; ch < num_channels; ch++) {
        resampler.reset();
        int num_samples_after_resample =
            resampler.processBlock(inBuffer.getReadPointer(ch), outBuffer.getWritePointer(ch), num_samples);
        jassertquiet(num_samples_after_resample == num_expected_samples_after_resample);
    }
}

void resampleBufferToMono(const AudioBuffer<float>& inBuffer,
                          AudioBuffer<float>& outBuffer,
                          double inSourceSampleRate,
                          double inTargetSampleRate)
{
    const int num_channels = inBuffer.getNumChannels();
    const int num_samples = inBuffer.getNumSamples();

    if (num_channels < 1 || num_samples <= 0 || !std::isfinite(inSourceSampleRate) || inSourceSampleRate <= 0.0
        || !std::isfinite(inTargetSampleRate) || inTargetSampleRate <= 0.0) {
        // Nothing to average. Caught here rather than left to divide by zero downstream.
        jassertfalse;
        outBuffer.setSize(1, 0);
        return;
    }

    if (juce::exactlyEqual(inSourceSampleRate, inTargetSampleRate)) {
        outBuffer.setSize(1, num_samples);
        outBuffer.copyFrom(0, 0, inBuffer, 0, 0, num_samples);

        for (int ch = 1; ch < num_channels; ch++) {
            outBuffer.addFrom(0, 0, inBuffer, ch, 0, num_samples);
        }

        outBuffer.applyGain(1.0f / static_cast<float>(num_channels));
        return;
    }

    const double speed_ratio = inSourceSampleRate / inTargetSampleRate;
    const double estimated_out =
        (static_cast<double>(num_samples) + static_cast<double>(LagrangeInterpolator::getBaseLatency())) / speed_ratio;

    if (!std::isfinite(estimated_out) || estimated_out <= 0.0
        || estimated_out > static_cast<double>(std::numeric_limits<int>::max())
        || estimated_out > static_cast<double>(MAX_TOTAL_AUDIO_SAMPLES)) {
        outBuffer.setSize(1, 0);
        return;
    }

    Resampler resampler;
    resampler.prepareToPlay(inSourceSampleRate, num_samples, inTargetSampleRate);
    const auto num_expected_samples_after_resample = resampler.getNumOutSamplesOnNextProcessBlock(num_samples);

    if (num_expected_samples_after_resample <= 0) {
        outBuffer.setSize(1, 0);
        return;
    }

    outBuffer.setSize(1, num_expected_samples_after_resample);

    // One pass over the whole thing, channels folded in as they are read.
    const int num_samples_after_resample = resampler.processBlock(
        inBuffer.getArrayOfReadPointers(), num_channels, outBuffer.getWritePointer(0), num_samples);
    jassertquiet(num_samples_after_resample == num_expected_samples_after_resample);
}

bool _loadMP3File(const juce::File& inFile, juce::AudioBuffer<float>& outBuffer, double& outSampleRate)
{
    const juce::int64 file_size = inFile.getSize();
    if (file_size <= 0 || file_size > MAX_MP3_FILE_BYTES) {
        return false;
    }

    // Read through JUCE rather than mp3dec_load: minimp3 opens paths with CreateFileA on Windows,
    // which misreads a UTF-8 path containing non-ASCII characters.
    juce::MemoryBlock data;
    if (!inFile.loadFileAsData(data)) {
        return false;
    }

    mp3dec_t mp3d;
    mp3dec_file_info_t info {};
    int loadResult =
        mp3dec_load_buf(&mp3d, static_cast<const uint8_t*>(data.getData()), data.getSize(), &info, nullptr, nullptr);

    // minimp3 returns success with no channels when it finds no mp3 frame.
    if (loadResult || info.channels <= 0 || info.channels > MAX_AUDIO_CHANNELS || info.samples == 0
        || info.samples > static_cast<size_t>(MAX_TOTAL_AUDIO_SAMPLES)
        || info.hz < static_cast<int>(MIN_AUDIO_SAMPLE_RATE) || info.hz > static_cast<int>(MAX_AUDIO_SAMPLE_RATE)) {
        free(info.buffer);
        return false;
    }

    const auto num_channels = static_cast<size_t>(info.channels);
    const size_t num_frames = info.samples / num_channels;

    if (num_frames == 0 || num_frames > static_cast<size_t>(std::numeric_limits<int>::max())
        || static_cast<double>(num_frames) / static_cast<double>(info.hz) > MAX_AUDIO_DURATION_SECONDS) {
        free(info.buffer);
        return false;
    }

    outBuffer.setSize(info.channels, static_cast<int>(num_frames));

    for (size_t i = 0; i < num_frames * num_channels; ++i) {
        size_t channel = i % num_channels;
        outBuffer.setSample((int) channel, static_cast<int>(i / num_channels), (float) info.buffer[i] / 32768.0f);
    }

    outSampleRate = static_cast<double>(info.hz);
    free(info.buffer);
    return true;
}

} // namespace AudioUtils