#pragma once

// Count received mapping frames, not optimization iterations or keyframes.
inline bool saveDiagnosticImages(int frame_index)
{
    return frame_index >= 0 && (frame_index + 1) % 50 == 0;
}
