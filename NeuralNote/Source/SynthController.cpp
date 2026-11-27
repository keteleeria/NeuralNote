//
// Created by Damien Ronssin on 10.06.23.
//

#include "SynthController.h"

#include <algorithm>

#include "PluginProcessor.h"

SynthController::SynthController(NeuralNoteAudioProcessor* inProcessor)
    : mProcessor(inProcessor)
{
    // Sized for the worst block: every active note released and a fresh set started, none of which
    // may allocate on the audio thread. Generous on purpose -- a few KB against a resize in the
    // callback is not a trade worth thinking about twice.
    mMidiBuffer.ensureSize(16 * (2 * NoteScheduler::MAX_ACTIVE_NOTES + 128));
}

void SynthController::setSampleRate(double inSampleRate)
{
    mScheduler.setSampleRate(inSampleRate);
}

void SynthController::setNotes(std::vector<NoteEvent>& ioNotes)
{
    const ScopedLock sl(mProcessor->getCallbackLock());

    mScheduler.setNotes(ioNotes);
}

const MidiBuffer& SynthController::generateNextMidiBuffer(int inNumSamples, bool inIsPlaying, bool inShouldLoop)
{
    mMidiBuffer.clear();

    const double pending_seek = mPendingSeekTime.exchange(-1.0, std::memory_order_acq_rel);
    if (pending_seek >= 0.0) {
        mScheduler.setTimeSeconds(pending_seek);
    }

    mScheduler.renderNextBlock(mMidiBuffer, inNumSamples, inIsPlaying);

    // Only while playing: this is called every block now, and with no audio loaded the duration is
    // zero, so an unguarded check would rewind the transport on every callback forever.
    if (inIsPlaying && mScheduler.getTimeSeconds() >= mProcessor->getSourceAudioManager()->getAudioSampleDuration()) {
        if (!inShouldLoop) {
            mProcessor->getPlayer()->setPlayingState(false);
        }

        mScheduler.setTimeSeconds(0.0);
    }

    return mMidiBuffer;
}

void SynthController::stopAllNotes()
{
    mScheduler.stopAllNotes();
}

void SynthController::emitActiveNotesOffTo(MidiBuffer& outMidiBuffer) const
{
    mScheduler.emitActiveNotesOffTo(outMidiBuffer);
}

void SynthController::reset()
{
    const ScopedLock sl(mProcessor->getCallbackLock());

    mPendingSeekTime.store(-1.0, std::memory_order_release);
    mScheduler.reset();
}

void SynthController::setNewTimeSeconds(double inNewTime)
{
    jassert(inNewTime >= 0.0);
    mPendingSeekTime.store(std::max(0.0, inNewTime), std::memory_order_release);
}

double SynthController::getCurrentTimeSeconds() const
{
    const double pending_seek = mPendingSeekTime.load(std::memory_order_acquire);
    return pending_seek >= 0.0 ? pending_seek : mScheduler.getTimeSeconds();
}
