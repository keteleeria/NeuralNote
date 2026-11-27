//
// Created by Damien Ronssin on 11/11/2024.
//

#include "UpdateCheck.h"

#include <algorithm>
#include <atomic>

#include "NnFonts.h"
#include "NnIcons.h"
#include "NnLook.h"

namespace
{
/** Between the message, the button and the cross. */
constexpr int CONTENT_GAP = 9;

constexpr int BUTTON_HEIGHT = 24;

constexpr int CONNECTION_TIMEOUT_MS = 4000;
constexpr int MAX_REDIRECTS = 5;
constexpr size_t MAX_RESPONSE_BYTES = 256 * 1024;
constexpr int READ_CHUNK_BYTES = 4096;

/** Compares dotted versions numerically, ignoring a leading "v"; missing components count as 0. */
bool isNewerVersion(const juce::String& inCandidate, const juce::String& inCurrent)
{
    const auto components = [](const juce::String& inVersion) {
        return juce::StringArray::fromTokens(inVersion.trim().trimCharactersAtStart("vV"), ".", "");
    };

    const auto candidate = components(inCandidate);
    const auto current = components(inCurrent);

    for (int i = 0; i < juce::jmax(candidate.size(), current.size()); ++i) {
        const int a = candidate[i].getIntValue();
        const int b = current[i].getIntValue();

        if (a != b) {
            return a > b;
        }
    }

    return false;
}
} // namespace

class UpdateCheck::Worker : public juce::Thread
{
public:
    explicit Worker(UpdateCheck& inOwner)
        : juce::Thread("NeuralNote update check")
        , mOwner(&inOwner)
    {
    }

    ~Worker() override
    {
        requestStop();
        stopThread(-1);
    }

    void startCheck(bool inShowNotificationOnLatestVersion)
    {
        if (isThreadRunning()) {
            if (inShowNotificationOnLatestVersion) {
                mShowOnLatest.store(true, std::memory_order_relaxed);
            }

            return;
        }

        mShowOnLatest.store(inShowNotificationOnLatestVersion, std::memory_order_relaxed);
        startThread();
    }

    void requestStop()
    {
        signalThreadShouldExit();

        const juce::ScopedLock lock(mStreamLock);

        if (mStream != nullptr) {
            mStream->cancel();
        }
    }

private:
    void run() override
    {
        const juce::URL url("https://api.github.com/repos/DamRsn/NeuralNote/releases/latest");

        juce::WebInputStream stream(url, false);
        stream.withConnectionTimeout(CONNECTION_TIMEOUT_MS).withNumRedirectsToFollow(MAX_REDIRECTS);

        {
            const juce::ScopedLock lock(mStreamLock);
            mStream = &stream;
        }

        const juce::ScopeGuard forget_stream {[this] {
            const juce::ScopedLock lock(mStreamLock);
            mStream = nullptr;
        }};

        if (threadShouldExit() || !stream.connect(nullptr)) {
            return;
        }

        if (stream.getStatusCode() != 200) {
            return;
        }

        juce::MemoryOutputStream body;
        char buffer[READ_CHUNK_BYTES];

        while (!threadShouldExit() && body.getDataSize() < MAX_RESPONSE_BYTES) {
            const int max_read = static_cast<int>(
                std::min<size_t>(static_cast<size_t>(READ_CHUNK_BYTES), MAX_RESPONSE_BYTES - body.getDataSize()));
            const int num_read = stream.read(buffer, max_read);

            if (num_read <= 0) {
                break;
            }

            body.write(buffer, static_cast<size_t>(num_read));
        }

        if (threadShouldExit()) {
            return;
        }

        const juce::String result = body.toUTF8();

        if (result.isEmpty()) {
            return;
        }

        const auto json = juce::JSON::parse(result);

        if (!json.isObject()) {
            return;
        }

        const auto current_version = juce::String(JucePlugin_VersionString);
        const auto latest_version = json.getProperty("tag_name", {}).toString();
        const bool show_on_latest = mShowOnLatest.load(std::memory_order_relaxed);
        if (threadShouldExit()) {
            return;
        }

        const juce::Component::SafePointer<UpdateCheck> safe_this = mOwner;

        juce::MessageManager::callAsync([current_version, latest_version, show_on_latest, safe_this] {
            if (safe_this == nullptr) {
                return;
            }

            if (isNewerVersion(latest_version, current_version)) {
                safe_this->_showNewVersionAvailableNotification();
            } else if (show_on_latest) {
                safe_this->_showOnLatestVersionNotification();
            }
        });
    }

    juce::Component::SafePointer<UpdateCheck> mOwner;
    std::atomic<bool> mShowOnLatest {false};

    juce::CriticalSection mStreamLock;
    juce::WebInputStream* mStream = nullptr;
};

UpdateCheck::UpdateCheck()
    : mWorker(std::make_unique<Worker>(*this))
{
    mSeeUpdateButton.setLabel("See update", nn::fonts::buttonLabel());
    mSeeUpdateButton.setPadding(12, 12, 0);

    // Accent-outlined, like Drag MIDI out: it is the one thing this notification is asking for.
    mSeeUpdateButton.setColour(NnFlatButton::backgroundColourId, nn::colours::accentFillButton());
    mSeeUpdateButton.setColour(NnFlatButton::outlineColourId, nn::colours::accent);
    mSeeUpdateButton.setColour(NnFlatButton::textColourId, nn::colours::accentText);
    mSeeUpdateButton.setWantsKeyboardFocus(false);
    mSeeUpdateButton.onClick = [this] { mWebsiteUrl.launchInDefaultBrowser(); };
    addChildComponent(mSeeUpdateButton);

    // The same cross the status bar cancels a transcription with, for the same reason: the ten
    // second timer is generous, and there is no reason to wait it out.
    mDismissButton.setIcon(nn::icons::crossStroked, NnFlatButton::IconStyle::stroked, nn::metrics::cancelGlyphSize);
    mDismissButton.setCornerRadius(4.0f);
    mDismissButton.setTooltip("Dismiss");
    mDismissButton.setWantsKeyboardFocus(false);
    mDismissButton.onClick = [this] { _hideNotification(); };
    addAndMakeVisible(mDismissButton);
}

UpdateCheck::~UpdateCheck()
{
    stopTimer();
    mWorker.reset();
}

juce::String UpdateCheck::_message() const
{
    return mUpdateAvailable ? "A new version of NeuralNote is available"
                            : "You are on the latest version of NeuralNote";
}

juce::Rectangle<int> UpdateCheck::_panelBounds() const
{
    int width = 2 * nn::metrics::menuPadX + juce::GlyphArrangement::getStringWidthInt(nn::fonts::menuItem(), _message())
                + CONTENT_GAP + nn::metrics::cancelHitSize;

    if (mUpdateAvailable) {
        width += CONTENT_GAP + mSeeUpdateButton.getIdealWidth();
    }

    return getLocalBounds().removeFromRight(juce::jmin(width, getWidth()));
}

void UpdateCheck::resized()
{
    mPanel = _panelBounds();

    auto row = mPanel.reduced(nn::metrics::menuPadX, 0);

    mDismissButton.setBounds(
        row.removeFromRight(nn::metrics::cancelHitSize)
            .withSizeKeepingCentre(nn::metrics::cancelHitSize, nn::metrics::cancelHitSize));

    if (mUpdateAvailable) {
        row.removeFromRight(CONTENT_GAP);

        const int button_width = mSeeUpdateButton.getIdealWidth();
        mSeeUpdateButton.setBounds(
            row.removeFromRight(button_width).withSizeKeepingCentre(button_width, BUTTON_HEIGHT));
    }

    // What the buttons left behind, so a long message ellipsises rather than running under them.
    mTextArea = row.withTrimmedRight(CONTENT_GAP);
}

bool UpdateCheck::hitTest(int inX, int inY)
{
    // The component is given more room than the panel fills, so that the panel can size itself to
    // its message. Without this the empty part of it would swallow clicks meant for the piano roll.
    return mPanel.contains(inX, inY);
}

void UpdateCheck::paint(juce::Graphics& g)
{
    nn::drawPopupSurface(g, mPanel.toFloat());

    g.setColour(nn::colours::popupItem);
    g.setFont(nn::fonts::menuItem());
    g.drawText(_message(), mTextArea, juce::Justification::centredLeft, true);
}

void UpdateCheck::timerCallback()
{
    auto current_time = juce::Time::getCurrentTime();
    auto mouse_over = isMouseOver(true);

    if (mouse_over) {
        mHideTime = std::max(current_time + juce::RelativeTime::seconds(mTimeIncrementOnMouseOverSeconds), mHideTime);
    }

    if (current_time >= mHideTime) {
        _hideNotification();
    }
}

void UpdateCheck::checkForUpdate(bool inShowNotificationOnLatestVersion)
{
    const juce::Component::SafePointer<UpdateCheck> safe_this(this);

    // Call async because of issue on Windows with spinning cursor when started inside editor construction.
    juce::MessageManager::callAsync([safe_this, inShowNotificationOnLatestVersion] {
        if (safe_this != nullptr && safe_this->mWorker != nullptr) {
            safe_this->mWorker->startCheck(inShowNotificationOnLatestVersion);
        }
    });
}

void UpdateCheck::_showNewVersionAvailableNotification()
{
    mUpdateAvailable = true;
    setVisible(true);
    mSeeUpdateButton.setVisible(true);
    mHideTime = juce::Time::getCurrentTime() + juce::RelativeTime::seconds(mNotificationDurationSeconds);

    // The panel is sized to its contents, and it just gained a button.
    resized();

    startTimerHz(5);
}

void UpdateCheck::_showOnLatestVersionNotification()
{
    mUpdateAvailable = false;
    setVisible(true);
    mSeeUpdateButton.setVisible(false);
    mHideTime = juce::Time::getCurrentTime() + juce::RelativeTime::seconds(mNotificationDurationSeconds);

    resized();

    startTimerHz(5);
}

void UpdateCheck::_hideNotification()
{
    stopTimer();
    mSeeUpdateButton.setVisible(false);
    setVisible(false);
}
