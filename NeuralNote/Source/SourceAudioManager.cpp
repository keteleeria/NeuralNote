//
// Created by Damien Ronssin on 19.06.23.
//

#include <algorithm>
#include <array>
#include <memory>

#include "SourceAudioManager.h"
#include "PluginProcessor.h"
#include "NnLogger.h"

SourceAudioManager::SourceAudioManager(NeuralNoteAudioProcessor* inProcessor)
    : mProcessor(inProcessor)
{
    mProcessor->addListenerToStateValueTree(this);
    jassert(mProcessor->getValueTree().hasProperty(NnId::SourceAudioNativeSrPathId));
}

SourceAudioManager::~SourceAudioManager()
{
    mProcessor->removeListenerFromStateValueTree(this);
}

void SourceAudioManager::prepareToPlay(double inSampleRate, int inSamplesPerBlock)
{
    if (mIsRecording.load()) {
        stopRecording();
    }

    mSampleRate = inSampleRate > 0.0 ? inSampleRate : 44100.0;
    mMaxBlockSize = std::max(1, inSamplesPerBlock);
    mDownSampler.prepareToPlay(mSampleRate, mMaxBlockSize, TRANSCRIPTION_SAMPLE_RATE);

    // Sized for the maximum input samples Resampler's internal buffer can hold (including base
    // latency padding), so even host sample rates below TRANSCRIPTION_SAMPLE_RATE cannot overrun it.
    const int max_resampler_input = mMaxBlockSize + 2 * static_cast<int>(LagrangeInterpolator::getBaseLatency()) + 1;
    mInternalDownsampledBuffer.setSize(
        1, static_cast<int>(std::ceil(TRANSCRIPTION_SAMPLE_RATE / mSampleRate * max_resampler_input)) + 5);

    // canPlay(), not a list of states: the question is whether there is source audio to keep
    // playable, which is true from the moment a file is loaded and before anything is transcribed.
    if (mProcessor->canPlay() && !juce::exactlyEqual(mSampleRate, mSourceAudioSampleRate)
        && mSourceAudio.getNumSamples() > 0 && mSourceAudio.getNumChannels() > 0) {
        AudioBuffer<float> tmp_buffer;
        AudioUtils::resampleBuffer(mSourceAudio, tmp_buffer, mSourceAudioSampleRate, mSampleRate);

        if (tmp_buffer.getNumSamples() > 0 && tmp_buffer.getNumChannels() > 0) {
            const ScopedLock sl(mProcessor->getCallbackLock());
            mSourceAudio = std::move(tmp_buffer);
            mSourceAudioSampleRate = mSampleRate;
        }
    }
}

namespace
{
// Logic reports a stale playhead position for the first blocks after the transport starts.
constexpr int NUM_SETTLING_BLOCKS = 2;
} // namespace

void SourceAudioManager::_tryCaptureHostTimeline()
{
    auto* playhead = mProcessor->getPlayHead();

    if (playhead == nullptr) {
        return;
    }

    const auto position = playhead->getPosition();

    if (!position.hasValue() || !position->getIsPlaying()) {
        return;
    }

    if (mNumPlayingBlocks++ < NUM_SETTLING_BLOCKS) {
        return;
    }

    const auto bpm = position->getBpm();

    if (!bpm.hasValue() || *bpm <= 0.0) {
        return;
    }

    const auto time_signature = position->getTimeSignature();
    const int numerator = time_signature.hasValue() ? time_signature->numerator : 4;
    const int denominator = time_signature.hasValue() ? time_signature->denominator : 4;

    const auto ppq = position->getPpqPosition();
    const auto ppq_last_bar = position->getPpqPositionOfLastBarStart();

    double offset_seconds = 0.0;

    if (ppq.hasValue() && ppq_last_bar.hasValue() && numerator > 0 && denominator > 0) {
        const double bar_seconds = (numerator * 4.0 / denominator) * 60.0 / *bpm;
        const double seconds_into_take = static_cast<double>(mNumSamplesAcquired.load()) / mSampleRate;

        // Where this bar started, in take time. Positive if the transport started after the take
        // did, so step back whole bars until the line is at or before the take's own zero.
        const double bar_start = seconds_into_take - (*ppq - *ppq_last_bar) * 60.0 / *bpm;

        offset_seconds = -(bar_start - std::ceil(bar_start / bar_seconds) * bar_seconds);
    }

    mExportStartOffsetSeconds.store(offset_seconds);
    mHostBpm.store(*bpm);
    mCapturedHostTimeline = true;
}

void SourceAudioManager::processBlock(const AudioBuffer<float>& inBuffer)
{
    if (!mIsRecording.load()) {
        return;
    }

    const ScopedTryLock sl(mWriterLock);

    if (!sl.isLocked() || !mIsRecording.load()) {
        return;
    }

    const int total_samples = inBuffer.getNumSamples();
    const int num_channels = inBuffer.getNumChannels();

    if (total_samples <= 0 || num_channels <= 0 || mMaxBlockSize <= 0 || mInternalDownsampledBuffer.getNumChannels() <= 0
        || mInternalDownsampledBuffer.getNumSamples() <= 0 || mThreadedWriter == nullptr
        || mThreadedWriterDown == nullptr) {
        return;
    }

    if (!mCapturedHostTimeline) {
        // Before the count moves on, so the reference is where the take stands entering here.
        _tryCaptureHostTimeline();
    }

    // Write incoming audio to file at native sample rate
    if (mWriterNumChannels > 0 && num_channels >= mWriterNumChannels) {
        bool result = mThreadedWriter->write(inBuffer.getArrayOfReadPointers(), total_samples);
        jassertquiet(result);
        mNumSamplesAcquired += static_cast<unsigned long long>(total_samples);
    }

    // Downmix to mono and downsample in chunks bounded by the prepared block size so variable-block
    // hosts cannot overrun the resampler or internal output buffer.
    const int max_chunk = mMaxBlockSize;
    const int channels_to_mix = std::min(num_channels, 64);
    std::array<const float*, 64> chunk_ptrs {};

    for (int offset = 0; offset < total_samples; offset += max_chunk) {
        const int chunk_samples = std::min(max_chunk, total_samples - offset);

        for (int ch = 0; ch < channels_to_mix; ++ch) {
            chunk_ptrs[static_cast<size_t>(ch)] = inBuffer.getReadPointer(ch) + offset;
        }

        int num_samples_down = mDownSampler.processBlock(chunk_ptrs.data(),
                                                         channels_to_mix,
                                                         mInternalDownsampledBuffer.getWritePointer(0),
                                                         chunk_samples);
        num_samples_down = std::clamp(num_samples_down, 0, mInternalDownsampledBuffer.getNumSamples());

        if (num_samples_down > 0) {
            bool result_down =
                mThreadedWriterDown->write(mInternalDownsampledBuffer.getArrayOfReadPointers(), num_samples_down);
            jassertquiet(result_down);

            mNumSamplesAcquiredDown += static_cast<unsigned long long>(num_samples_down);
        }
    }

    mDuration = static_cast<double>(mNumSamplesAcquiredDown) / TRANSCRIPTION_SAMPLE_RATE;
}

void SourceAudioManager::startRecording()
{
    // If already recording (should not happen but if it does, simply return)
    if (mIsRecording.load()) {
        jassertfalse;
        return;
    }

    // Prepare files to be written
    bool dir_ready = NNFileUtils::ensureDirectoryExists(mRecordingsDir);
    jassertquiet(dir_ready);

    mDroppedFilename = "";

    const String prefix = NNFileUtils::RECORDING_FILENAME_PREFIX;
    String timestamp = Time::getCurrentTime().formatted("%Y-%m-%d_%H-%M-%S");

    mSourceFile = mRecordingsDir.getChildFile(prefix + timestamp + ".wav");
    mRecordedFileDown = mRecordingsDir.getChildFile(prefix + timestamp + "_downsampled.wav");

    size_t i = 1;

    while (mSourceFile.existsAsFile() || mRecordedFileDown.existsAsFile()) {
        mSourceFile = mRecordingsDir.getChildFile(prefix + timestamp + "_" + String(i) + ".wav");
        mRecordedFileDown = mRecordingsDir.getChildFile(prefix + timestamp + "_" + String(i) + "_downsampled.wav");
        i += 1;
    }

    Result file_creation_result = mSourceFile.create();
    Result file_creation_result_down = mRecordedFileDown.create();

    if (!file_creation_result.wasOk() || !file_creation_result_down.wasOk()) {
        mProcessor->clear();
        NnLogger::write(NnLogger::Level::Error, "SourceAudioManager: File creation for recording failed.");
        NativeMessageBox::showMessageBoxAsync(
            MessageBoxIconType::NoIcon, "Error", "File creation for recording failed.");
        return;
    }

    mFilesToDelete.push_back(mSourceFile);
    mFilesToDelete.push_back(mRecordedFileDown);

    const int num_input_channels = std::min(mProcessor->getTotalNumInputChannels(), 2);

    if (num_input_channels <= 0) {
        mProcessor->clear();
        NnLogger::write(NnLogger::Level::Error, "SourceAudioManager: Cannot record with zero input channels.");
        NativeMessageBox::showMessageBoxAsync(
            MessageBoxIconType::NoIcon, "Error", "No audio input channels are available for recording.");
        return;
    }

    // Init first writer at native sample rate (stereo)
    WavAudioFormat format;

    std::unique_ptr<OutputStream> stream = std::make_unique<FileOutputStream>(mSourceFile);
    auto wav_writer = format.createWriterFor(stream,
                                             AudioFormatWriterOptions()
                                                 .withSampleRate(mSampleRate)
                                                 .withNumChannels(num_input_channels)
                                                 .withBitsPerSample(16));

    // Init second writer at basic pitch sample rate (mono)
    WavAudioFormat format_down;

    std::unique_ptr<OutputStream> stream_down = std::make_unique<FileOutputStream>(mRecordedFileDown);
    auto wav_writer_down = format_down.createWriterFor(
        stream_down,
        AudioFormatWriterOptions().withSampleRate(TRANSCRIPTION_SAMPLE_RATE).withNumChannels(1).withBitsPerSample(16));

    if (wav_writer == nullptr || wav_writer_down == nullptr) {
        wav_writer.reset();
        wav_writer_down.reset();
        stream.reset();
        stream_down.reset();
        mProcessor->clear();
        NnLogger::write(NnLogger::Level::Error, "SourceAudioManager: Could not initialize WAV writers for recording.");
        NativeMessageBox::showMessageBoxAsync(
            MessageBoxIconType::NoIcon, "Error", "Could not initialize audio writers for recording.");
        return;
    }

    mWriterNumChannels = num_input_channels;

    mWriterThread.startThread();
    mThreadedWriter = std::make_unique<AudioFormatWriter::ThreadedWriter>(wav_writer.release(), mWriterThread, 32768);

    mWriterThreadDown.startThread();
    mThreadedWriterDown =
        std::make_unique<AudioFormatWriter::ThreadedWriter>(wav_writer_down.release(), mWriterThreadDown, 32768);
    mDownSampler.reset();

    // setDataReceiver resets the receiver for us, which is what clears the peaks from the last take.
    mThreadedWriterDown->setDataReceiver(&mPeaksReceiver);

    mNumSamplesAcquired = 0;
    mNumSamplesAcquiredDown = 0;

    mExportStartOffsetSeconds = 0.0;
    mHostBpm = 0.0;
    mCapturedHostTimeline = false;
    mNumPlayingBlocks = 0;

    mIsRecording.store(true);
    mProcessor->setStateToRecording();

    NnLogger::write(NnLogger::Level::Info, "SourceAudioManager: recording started at " + String(mSampleRate) + " Hz");
}

void SourceAudioManager::stopRecording()
{
    if (!mIsRecording.load()) {
        return;
    }

    {
        ScopedLock sl(mWriterLock);
        mIsRecording.store(false);
    }

    mThreadedWriter.reset();
    mThreadedWriterDown.reset();

    mWriterThread.stopThread(1000);
    mWriterThreadDown.stopThread(1000);

    NnLogger::write(NnLogger::Level::Info,
                    "SourceAudioManager: recording stopped after "
                        + String(static_cast<double>(mNumSamplesAcquiredDown) / TRANSCRIPTION_SAMPLE_RATE, 1) + " s");

    // Record then stop before a block came in: there is no audio to load, and going to AudioLoaded
    // would offer transport and transcription over an empty buffer.
    if (mNumSamplesAcquired == 0 || mNumSamplesAcquiredDown == 0) {
        mProcessor->clear();
        return;
    }

    AudioBuffer<float> loaded_source_audio;
    double loaded_source_sr = mSampleRate;
    bool success = AudioUtils::loadAudioFile(mSourceFile, loaded_source_audio, loaded_source_sr);
    jassert(juce::exactlyEqual(loaded_source_sr, mSampleRate));

    // Should def not happen
    if (!success) {
        mProcessor->clear();
        NnLogger::write(NnLogger::Level::Error, "SourceAudioManager: Could not load the recorded audio sample.");
        NativeMessageBox::showMessageBoxAsync(
            MessageBoxIconType::NoIcon, "Could not load the recorded audio sample.", "");
        return;
    }

    {
        const ScopedLock sl(mProcessor->getCallbackLock());
        mSourceAudio = std::move(loaded_source_audio);
        mSourceAudioSampleRate = loaded_source_sr;
    }

    double dummy_sr;
    success = AudioUtils::loadAudioFile(mRecordedFileDown, mDownsampledSourceAudio, dummy_sr);
    jassert(juce::exactlyEqual(dummy_sr, TRANSCRIPTION_SAMPLE_RATE));

    // Should def not happen
    if (!success) {
        mProcessor->clear();
        NnLogger::write(NnLogger::Level::Error, "SourceAudioManager: Could not load the recorded audio sample.");
        NativeMessageBox::showMessageBoxAsync(
            MessageBoxIconType::NoIcon, "Could not load the recorded audio sample.", "");
        return;
    }

    // Rebuild from the buffer that will actually be played and transcribed, rather than keeping the
    // peaks accumulated live off the FIFO: this went to disk as 16-bit and came back, so the two
    // are no longer quite the same audio.
    // The tempo the take was played to is the one to export it at, so the toolbar field follows it.
    if (const double host_bpm = mHostBpm.load(); host_bpm > 0.0) {
        mProcessor->getValueTree().setProperty(NnId::ExportTempoId, host_bpm, nullptr);
    }

    mWaveformPeaks.buildFrom(
        {mDownsampledSourceAudio.getReadPointer(0), static_cast<size_t>(mDownsampledSourceAudio.getNumSamples())});
    sendChangeMessage();

    auto& tree = mProcessor->getValueTree();
    tree.setPropertyExcludingListener(this, NnId::SourceAudioNativeSrPathId, mSourceFile.getFullPathName(), nullptr);

    mProcessor->setStateToAudioLoaded();
}

bool SourceAudioManager::onFileDrop(const File& inFile)
{
    const State state = mProcessor->getState();

    if (state == EmptyAudioAndMidiRegions || state == AudioLoaded || state == Paused
        || state == PopulatedAudioAndMidiRegions) {
        mProcessor->clear();

        AudioBuffer<float> loaded_source_audio;
        double loaded_source_sr = mSampleRate;
        bool success = AudioUtils::loadAudioFile(inFile, loaded_source_audio, loaded_source_sr);

        if (!success) {
            mProcessor->clear();
            NnLogger::write(NnLogger::Level::Warning,
                            "SourceAudioManager: Could not load the audio file (" + inFile.getFileExtension() + ").");
            NativeMessageBox::showMessageBoxAsync(
                MessageBoxIconType::NoIcon,
                "Could not load the audio file.",
                "Check your file format (Accepted formats: "
                    + AudioUtils::getSupportedAudioFileExtensions().joinIntoString(", ") + ").");
            return false;
        }

        NnLogger::write(NnLogger::Level::Info,
                        "SourceAudioManager: loaded a " + inFile.getFileExtension() + " file: "
                            + String(loaded_source_sr) + " Hz, " + String(loaded_source_audio.getNumChannels())
                            + " channels, " + String(loaded_source_audio.getNumSamples() / loaded_source_sr, 1) + " s");

        // Downmix and downsample to the rate the model expects, in one pass
        AudioBuffer<float> downsampled_audio;
        AudioUtils::resampleBufferToMono(
            loaded_source_audio, downsampled_audio, loaded_source_sr, TRANSCRIPTION_SAMPLE_RATE);

        if (downsampled_audio.getNumChannels() <= 0 || downsampled_audio.getNumSamples() <= 0) {
            mProcessor->clear();
            NnLogger::write(NnLogger::Level::Warning, "SourceAudioManager: Could not resample loaded audio for transcription.");
            return false;
        }

        // Resample to current plugin sample rate for playback
        if (!juce::exactlyEqual(loaded_source_sr, mSampleRate)) {
            AudioBuffer<float> tmp_buffer;
            AudioUtils::resampleBuffer(loaded_source_audio, tmp_buffer, loaded_source_sr, mSampleRate);

            if (tmp_buffer.getNumChannels() <= 0 || tmp_buffer.getNumSamples() <= 0) {
                mProcessor->clear();
                NnLogger::write(NnLogger::Level::Warning, "SourceAudioManager: Could not resample loaded audio for playback.");
                return false;
            }

            loaded_source_audio = std::move(tmp_buffer);
            loaded_source_sr = mSampleRate;
        }

        mDownsampledSourceAudio = std::move(downsampled_audio);

        {
            const ScopedLock sl(mProcessor->getCallbackLock());
            mSourceAudio = std::move(loaded_source_audio);
            mSourceAudioSampleRate = loaded_source_sr;
        }

        mNumSamplesAcquiredDown = static_cast<unsigned long long>(mDownsampledSourceAudio.getNumSamples());
        mNumSamplesAcquired = static_cast<unsigned long long>(mSourceAudio.getNumSamples());
        mDuration = static_cast<double>(mNumSamplesAcquiredDown) / TRANSCRIPTION_SAMPLE_RATE;

        mDroppedFilename = inFile.getFileNameWithoutExtension();
        mSourceFile = inFile;

        mExportStartOffsetSeconds = 0.0;
        mHostBpm = 0.0;

        auto& tree = mProcessor->getValueTree();
        tree.setPropertyExcludingListener(this, NnId::SourceAudioNativeSrPathId, inFile.getFullPathName(), nullptr);

        mWaveformPeaks.buildFrom(
            {mDownsampledSourceAudio.getReadPointer(0), static_cast<size_t>(mDownsampledSourceAudio.getNumSamples())});
        sendChangeMessage();

        // No transcription yet: the user starts one from the piano roll, once they have had the
        // chance to pick which instruments it should look for.
        mProcessor->setStateToAudioLoaded();

    } else {
        jassertfalse;
    }

    return true;
}

void SourceAudioManager::clear()
{
    if (mIsRecording) {
        stopRecording();
    }

    // Before the buffers go: the peaks borrow mDownsampledSourceAudio rather than copying it.
    mWaveformPeaks.clear();

    {
        const ScopedLock sl(mProcessor->getCallbackLock());
        mSourceAudio = {};
    }

    mDownsampledSourceAudio = {};

    mNumSamplesAcquiredDown = 0;
    mNumSamplesAcquired = 0;
    mDuration = 0.0;

    mExportStartOffsetSeconds = 0.0;
    mHostBpm = 0.0;

    mDownSampler.reset();

    _deleteFilesToDelete();

    mProcessor->getValueTree().setPropertyExcludingListener(this, NnId::SourceAudioNativeSrPathId, String(), nullptr);

    mSourceFile = File();
    mRecordedFileDown = File();
    mDroppedFilename = "";

    sendChangeMessage();
}

void SourceAudioManager::PeaksReceiver::reset(int numChannels, double sampleRate, int64 totalSamplesInSource)
{
    ignoreUnused(numChannels, sampleRate, totalSamplesInSource);

    mOwner.mWaveformPeaks.clear();
}

void SourceAudioManager::PeaksReceiver::addBlock(int64 sampleNumberInSource,
                                                 const AudioBuffer<float>& newData,
                                                 int startOffsetInBuffer,
                                                 int numSamples)
{
    ignoreUnused(sampleNumberInSource);

    // Mono: this is the writer that runs at the transcription rate.
    mOwner.mWaveformPeaks.append({newData.getReadPointer(0, startOffsetInBuffer), static_cast<size_t>(numSamples)});

    mOwner.sendChangeMessage();
}

AudioBuffer<float>& SourceAudioManager::getDownsampledSourceAudioForTranscription()
{
    // The caller reads channel 0 and nothing else, so a second channel here would be half the mix
    // silently thrown away rather than a harmless extra.
    jassert(mDownsampledSourceAudio.getNumChannels() <= 1);

    return mDownsampledSourceAudio;
}

AudioBuffer<float>& SourceAudioManager::getSourceAudioForPlayback()
{
    return mSourceAudio;
}

String SourceAudioManager::getDroppedFilename() const
{
    return mDroppedFilename;
}

int SourceAudioManager::getNumSamplesDownAcquired() const
{
    return static_cast<int>(mNumSamplesAcquiredDown);
}

double SourceAudioManager::getAudioSampleDuration() const
{
    return mDuration;
}

const WaveformPeaks& SourceAudioManager::getWaveformPeaks() const
{
    return mWaveformPeaks;
}

void SourceAudioManager::valueTreePropertyChanged(ValueTree& treeWhosePropertyHasChanged, const Identifier& property)
{
    if (property == NnId::SourceAudioNativeSrPathId) {
        auto path = treeWhosePropertyHasChanged.getProperty(property).toString();

        if (path != mSourceFile.getFullPathName()) {
            if (path.isEmpty()) {
                clear();
                return;
            }

            const auto trimmed_path = path.trim();
            const auto normalized_prefix = trimmed_path.substring(0, 2).replaceCharacter('/', '\\');

            if (normalized_prefix == "\\\\" || !File::isAbsolutePath(trimmed_path)) {
                NnLogger::write(NnLogger::Level::Warning,
                                "SourceAudioManager: refusing to auto-load network/UNC or relative path from saved state");
                return;
            }

            onFileDrop(File(trimmed_path));

            // If loading state from recorded audio, add to files to delete on clear.
            if (NNFileUtils::isRecordingFile(mSourceFile)) {
                mRecordedFileDown =
                    mRecordingsDir.getChildFile(mSourceFile.getFileNameWithoutExtension() + "_downsampled.wav");
                mFilesToDelete.push_back(mSourceFile);
                mFilesToDelete.push_back(mRecordedFileDown);
                mDroppedFilename = "";
            }
        }
    }
}

void SourceAudioManager::_deleteFilesToDelete()
{
    for (auto& file: mFilesToDelete) {
        // Make sure we only ever delete files that have been recorded, not loaded from disk.
        if (NNFileUtils::isRecordingFile(file)) {
            bool res = file.deleteFile();
            jassertquiet(res);
        } else {
            jassertfalse;
        }
    }

    mFilesToDelete.clear();
}
