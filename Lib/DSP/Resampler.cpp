//
// Created by Damien Ronssin on 05.03.23.
//

#include "Resampler.h"

#include <algorithm>
#include <cmath>

void Resampler::prepareToPlay(double inSourceSampleRate, int inMaxBlockSize, double inTargetSampleRate)
{
    mSourceSampleRate = inSourceSampleRate > 0.0 ? inSourceSampleRate : 44100.0;
    mTargetSampleRate = inTargetSampleRate > 0.0 ? inTargetSampleRate : 44100.0;
    mSpeedRatio = mSourceSampleRate / mTargetSampleRate;
    mMaxBlockSize = std::max(1, inMaxBlockSize);

    mInternalBuffer.setSize(1, mMaxBlockSize + 2 * static_cast<int>(LagrangeInterpolator::getBaseLatency()) + 1);

    // Lowpass filter stuffs
    if (mTargetSampleRate < mSourceSampleRate) {
        auto filter_coeffs = juce::dsp::FilterDesign<float>::designIIRLowpassHighOrderButterworthMethod(
            static_cast<float>(mTargetSampleRate / 2.0), mSourceSampleRate, 4);

        mLowpassFilters.resize(static_cast<size_t>(filter_coeffs.size()));

        for (size_t i = 0; i < mLowpassFilters.size(); i++) {
            mLowpassFilters[i].coefficients = filter_coeffs[(int) i];
        }
    } else {
        // Remove elements from vector (set size to 0)
        mLowpassFilters.clear();
    }

    reset();
}

void Resampler::reset()
{
    mInternalBuffer.clear();
    mNumInputSamplesAvailable = mInitPadding;
    mInterpolator.reset();

    for (auto& lowpass_filter: mLowpassFilters)
        lowpass_filter.reset();
}

int Resampler::processBlock(const float* inBuffer, float* outBuffer, int inNumSamples)
{
    return processBlock(&inBuffer, 1, outBuffer, inNumSamples);
}

int Resampler::processBlock(const float* const* inChannels, int inNumChannels, float* outBuffer, int inNumSamples)
{
    if (inChannels == nullptr || outBuffer == nullptr || inNumChannels < 1 || inNumSamples <= 0) {
        return 0;
    }

    const int max_writable = mInternalBuffer.getNumSamples() - mNumInputSamplesAvailable;

    if (max_writable <= 0) {
        jassertfalse;
        return 0;
    }

    jassert(inNumSamples <= max_writable);
    const int samples_to_process = std::min(inNumSamples, max_writable);

    float* internal_buffer_ptr = mInternalBuffer.getWritePointer(0);

    // Averaged, not summed: a downmix that can exceed the input's peak would clip a hot master, and
    // the model is sensitive to level.
    const float scale = 1.0f / static_cast<float>(inNumChannels);

    for (int i = 0; i < samples_to_process; i++) {
        float sum = 0.0f;

        for (int ch = 0; ch < inNumChannels; ch++) {
            sum += inChannels[ch][i];
        }

        internal_buffer_ptr[mNumInputSamplesAvailable + i] = sum * scale;
    }

    // Lowpass filter if necessary
    if (mTargetSampleRate < mSourceSampleRate) {
        for (int i = 0; i < samples_to_process; i++) {
            for (auto& lowpass_filter: mLowpassFilters) {
                internal_buffer_ptr[mNumInputSamplesAvailable + i] =
                    lowpass_filter.processSample(internal_buffer_ptr[mNumInputSamplesAvailable + i]);
            }
        }
    }

    mNumInputSamplesAvailable += samples_to_process;

    int num_out_samples_to_produce = static_cast<int>(std::floor(mNumInputSamplesAvailable / mSpeedRatio));

    int num_input_samples_used =
        mInterpolator.process(mSpeedRatio, mInternalBuffer.getReadPointer(0), outBuffer, num_out_samples_to_produce);

    jassert(num_input_samples_used <= mNumInputSamplesAvailable);

    mNumInputSamplesAvailable = std::max(0, mNumInputSamplesAvailable - num_input_samples_used);

    for (int i = 0; i < mNumInputSamplesAvailable; i++) {
        mInternalBuffer.setSample(0, i, mInternalBuffer.getSample(0, i + num_input_samples_used));
    }

    return num_out_samples_to_produce;
}

int Resampler::getNumOutSamplesOnNextProcessBlock(int inNumSamples) const
{
    return static_cast<int>(std::floor((mNumInputSamplesAvailable + inNumSamples) / mSpeedRatio));
}
