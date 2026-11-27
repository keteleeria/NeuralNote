//
// Created by Damien Ronssin on 11.06.23.
//

#include "Player.h"

#include <algorithm>
#include <cmath>

#include "PluginProcessor.h"

Player::Player(NeuralNoteAudioProcessor* inProcessor)
    : mProcessor(inProcessor)
{
    mProcessor->addListenerToStateValueTree(this);

    mSynth = std::make_unique<InstrumentSynth>();
    mSynthController = std::make_unique<SynthController>(inProcessor);

    setPlayheadPositionSeconds(mProcessor->getValueTree().getProperty(NnId::PlayheadPositionSecId, 0.0));

    mShouldOutputMidi.store(static_cast<bool>(mProcessor->getValueTree().getProperty(NnId::MidiOut, false)),
                            std::memory_order_relaxed);
    mShouldLoop = static_cast<bool>(mProcessor->getValueTree().getProperty(NnId::LoopEnabledId, false));
}

Player::~Player()
{
    mProcessor->removeListenerFromStateValueTree(this);
}

void Player::prepareToPlay(double inSampleRate, int inSamplesPerBlock)
{
    const double safe_sample_rate = inSampleRate > 0.0 ? inSampleRate : 44100.0;
    const int safe_block_size = std::max(1, inSamplesPerBlock);

    mSynth->prepareToPlay(safe_sample_rate, safe_block_size);
    mSynthController->setSampleRate(safe_sample_rate);
    mSampleRate = safe_sample_rate;

    // Sized to what actually goes out, not always to two: the synth renders real stereo now -- the
    // soundfont pans instruments across the field -- and a mono host would otherwise be handed only
    // the left channel and quietly lose whatever sits on the right. At one channel the synth folds.
    mInternalBuffer.setSize(jlimit(1, 2, mProcessor->getTotalNumOutputChannels()), safe_block_size);

    mMasterMeter.prepare(safe_sample_rate, METER_WINDOW_SECONDS);
    mMasterMeanSquare.store(0.0f, std::memory_order_relaxed);
}

void Player::processBlock(AudioBuffer<float>& inAudioBuffer, MidiBuffer& outMidiBuffer)
{
    const int total_samples = inAudioBuffer.getNumSamples();
    const int max_chunk = mInternalBuffer.getNumSamples();

    // Clamped to what mInternalBuffer and inAudioBuffer actually hold: a host that reports a
    // different channel count now must not send these loops past the end of either buffer.
    const int num_out_channels = std::min(
        std::min(mProcessor->getTotalNumOutputChannels(), mInternalBuffer.getNumChannels()),
        inAudioBuffer.getNumChannels());

    if (total_samples <= 0 || max_chunk <= 0 || num_out_channels <= 0) {
        return;
    }

    jassert(num_out_channels > 0 && num_out_channels <= 2);

    auto old_audio_gain = mGainSourceAudio;
    auto old_master_gain = mMasterGain;

    _setGains(mProcessor->getParameterValue(ParameterHelpers::MixId),
              mProcessor->getParameterValue(ParameterHelpers::MasterGainId));

    const bool should_output_midi = mShouldOutputMidi.load(std::memory_order_relaxed);

    // Before the block is generated, not after: what the host is still holding was started in an
    // earlier block, so it is exactly what is active right now. A note ending inside this block
    // would be gone from that set by the end of it, with its note-off reaching only the synth.
    if (!should_output_midi && mWasOutputtingMidi) {
        // The host stops seeing this buffer from here on, so whatever it is holding has to be
        // released now. The synth keeps its notes; only this consumer is being disconnected.
        mSynthController->emitActiveNotesOffTo(outMidiBuffer);
        mWasOutputtingMidi = false;
    }

    const bool should_loop = mShouldLoop.load(std::memory_order_relaxed);

    // Process in chunks bounded by the prepared internal buffer size so variable-block hosts
    // (such as FL Studio) cannot overrun mInternalBuffer or InstrumentSynth's scratch buffers.
    for (int sample_offset = 0; sample_offset < total_samples; sample_offset += max_chunk) {
        const int chunk_samples = std::min(max_chunk, total_samples - sample_offset);
        const bool is_playing = mIsPlaying.load(std::memory_order_relaxed);

        // Consume any pending seek inside the chunk loop right beside SynthController so audio and
        // MIDI playheads stay locked on the same chunk even under variable-block chunking.
        const double pending_playhead = mPendingPlayheadTime.exchange(-1.0, std::memory_order_acq_rel);
        if (pending_playhead >= 0.0) {
            mPlayheadTime.store(pending_playhead, std::memory_order_relaxed);
        }

        mInternalBuffer.clear(0, chunk_samples);

        // Every block, playing or not: a stopped transport, a seek and a swapped note list all leave
        // note-offs to deliver, and this is the only thing that delivers them.
        auto& midi_buffer = mSynthController->generateNextMidiBuffer(chunk_samples, is_playing, should_loop);

        if (should_output_midi) {
            outMidiBuffer.addEvents(midi_buffer, 0, chunk_samples, sample_offset);
            mWasOutputtingMidi = true;
        }

        if (mShouldSilenceSynth.exchange(false)) {
            // The scheduler's note-offs are not enough on their own: the synth deliberately ignores
            // them for drums, because a General MIDI kit is one-shot and honouring a note-off 10 ms
            // after the hit would choke every cymbal. So a stop or a seek has to say so directly, or a
            // crash rings on through it.
            mSynth->allNotesOff();
        }

        // The synth is driven by the scheduler's own event list rather than by midi_buffer: a
        // transcription can name 35 instruments and MIDI has 16 channels, so the buffer the host sees
        // is the flattened rendering of the same decisions, not the full one.
        //
        // The gain ramp lives inside the synth, which needs a per-instrument one anyway.
        mSynth->processBlock(mInternalBuffer, mSynthController->getSynthEvents(), chunk_samples, mGainSynth);

        if (is_playing && mProcessor->canPlay()) {
            const auto& source_buffer = mProcessor->getSourceAudioManager()->getSourceAudioForPlayback();
            const int num_source_samples = source_buffer.getNumSamples();
            const int num_source_channels = source_buffer.getNumChannels();

            if (num_source_samples > 0 && num_source_channels > 0) {
                int playhead_index =
                    static_cast<int>(std::round(mPlayheadTime.load(std::memory_order_relaxed) * mSampleRate));

                if (playhead_index < 0 || playhead_index >= num_source_samples) {
                    playhead_index = 0;
                }

                const int num_samples = std::min(chunk_samples, num_source_samples - playhead_index);

                if (num_samples > 0) {
                    for (int ch = 0; ch < num_out_channels; ch++) {
                        const int source_channel = std::min(ch, num_source_channels - 1);
                        mInternalBuffer.addFromWithRamp(ch,
                                                        0,
                                                        source_buffer.getReadPointer(source_channel) + playhead_index,
                                                        num_samples,
                                                        old_audio_gain,
                                                        mGainSourceAudio);
                    }

                    playhead_index += num_samples;

                    if (playhead_index >= num_source_samples) {
                        playhead_index = 0;
                    }

                    mPlayheadTime.store(static_cast<double>(playhead_index) / mSampleRate, std::memory_order_relaxed);
                }
            }
        }

        // Ramped across the block, like the source gain above.
        for (int ch = 0; ch < num_out_channels; ch++) {
            mInternalBuffer.applyGainRamp(ch, 0, chunk_samples, old_master_gain, mMasterGain);
        }

        // Metered here rather than on the host's buffer, and after the fader: this is what comes out
        // of NeuralNote. Whatever the host sent in is passing through and is not ours to show.
        {
            const float* const left = mInternalBuffer.getReadPointer(0);
            const float* const right = mInternalBuffer.getNumChannels() > 1 ? mInternalBuffer.getReadPointer(1) : left;

            for (int i = 0; i < chunk_samples; i++) {
                mMasterMeter.push((left[i] + right[i]) * 0.5f);
            }
        }

        for (int ch = 0; ch < num_out_channels; ch++) {
            inAudioBuffer.addFrom(ch, sample_offset, mInternalBuffer, ch, 0, chunk_samples);
        }

        old_audio_gain = mGainSourceAudio;
        old_master_gain = mMasterGain;
    }

    mMasterMeanSquare.store(static_cast<float>(mMasterMeter.getMeanSquare()), std::memory_order_relaxed);
    mMeterFrame.fetch_add(1, std::memory_order_relaxed);
}

bool Player::isPlaying() const
{
    return mIsPlaying.load();
}

void Player::setPlayingState(bool inIsPlaying)
{
    mIsPlaying.store(inIsPlaying);

    if (!inIsPlaying) {
        // Through the scheduler, so the same note-offs also reach the MIDI output port.
        mSynthController->stopAllNotes();
        mShouldSilenceSynth = true;
    }
}

void Player::returnToStart()
{
    setPlayingState(false);
    setPlayheadPositionSeconds(0.0);
}

void Player::reset()
{
    mSynthController->reset();
    setPlayingState(false);

    {
        const ScopedLock sl(mProcessor->getCallbackLock());
        mPendingPlayheadTime.store(-1.0, std::memory_order_release);
        mPlayheadTime.store(0.0, std::memory_order_relaxed);
    }

    // The published value only, not the window behind it: this runs without the callback lock, so
    // the audio thread may be inside push(). The window drains itself over the next few blocks.
    mMasterMeanSquare.store(0.0f, std::memory_order_relaxed);
}

InstrumentSynth* Player::getInstrumentSynth() const
{
    return mSynth.get();
}

double Player::getPlayheadPositionSeconds() const
{
    return mSynthController->getCurrentTimeSeconds();
}

void Player::setPlayheadPositionSeconds(double inNewPosition)
{
    if (inNewPosition >= 0 && inNewPosition < mProcessor->getSourceAudioManager()->getAudioSampleDuration()) {
        mSynthController->setNewTimeSeconds(inNewPosition);
        mPendingPlayheadTime.store(inNewPosition, std::memory_order_release);

        // A drum hit that started before the seek has no note-off the synth will act on, so it
        // would otherwise carry across the jump.
        mShouldSilenceSynth = true;
    }
}

SynthController* Player::getSynthController() const
{
    return mSynthController.get();
}

void Player::saveStateToValueTree()
{
    const double pending_playhead = mPendingPlayheadTime.load(std::memory_order_acquire);
    const double playhead_to_save =
        pending_playhead >= 0.0 ? pending_playhead : mPlayheadTime.load(std::memory_order_relaxed);

    mProcessor->getValueTree().setPropertyExcludingListener(
        this, NnId::PlayheadPositionSecId, playhead_to_save, nullptr);
}

void Player::valueTreePropertyChanged(ValueTree& treeWhosePropertyHasChanged, const Identifier& property)
{
    if (property == NnId::PlayheadPositionSecId) {
        double new_position = treeWhosePropertyHasChanged.getProperty(property);

        if (mProcessor->getState() != EmptyAudioAndMidiRegions) {
            new_position = std::clamp(new_position, 0.0, mProcessor->getSourceAudioManager()->getAudioSampleDuration());
        } else {
            new_position = 0.0;
        }

        setPlayheadPositionSeconds(new_position);
    }

    if (property == NnId::MidiOut) {
        mShouldOutputMidi.store(static_cast<bool>(treeWhosePropertyHasChanged.getProperty(property)),
                                std::memory_order_relaxed);
    }

    if (property == NnId::LoopEnabledId) {
        mShouldLoop = static_cast<bool>(treeWhosePropertyHasChanged.getProperty(property));
    }
}

void Player::_setGains(float inMix, float inMasterGainDb)
{
    // With no notes the synth side is silence, so a mix towards MIDI would only fade the source
    // out for nothing.
    const float mix = mSynthController->hasNotes() ? inMix : 0.0f;

    // Equal power, so the two are both 3 dB down at the midpoint rather than summing to a bump.
    const float angle = mix * MathConstants<float>::halfPi;

    mGainSourceAudio = std::cos(angle);
    mGainSynth = std::sin(angle);
    mMasterGain = Decibels::decibelsToGain(inMasterGainDb, MIN_INF_GAIN_DB);
}
