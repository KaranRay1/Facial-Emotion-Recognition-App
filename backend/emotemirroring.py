# emotemirroring.py
#%%writefile emotemirroring.py
import cv2
import os
import numpy as np
import glob # To find image sequences
import time # ADDED THIS IMPORT FOR time.sleep

# Define the relative path to your emotes folder
EMOTE_ASSET_DIR_RELATIVE = '/content/drive/MyDrive/Project/Face Emotion Recoginization /face/assets/emotes'

# Global dictionary to store loaded emote animations (sequences of frames)
_EMOTE_ANIMATIONS = {}

# Global variable for animation speed control (frames to skip per update)
_ANIMATION_SKIP_FRAMES = 2 # Show every 2nd frame of the emote animation

class EmoteAnimator:
    def __init__(self, base_project_path, emotion_labels, target_size=(100, 100)):
        self.base_project_path = base_project_path
        self.emotion_labels = emotion_labels
        self.target_size = target_size
        self._load_all_emote_animations()
        self.frame_counter = 0 # A global counter for animation progression across all emotes

    def _load_all_emote_animations(self):
        """
        Loads all emote animation frames from subfolders.
        Expected structure: assets/emotes/emotion_label/emotion_label_XX.png
        """
        emotes_base_folder = os.path.join(self.base_project_path, EMOTE_ASSET_DIR_RELATIVE)
        print(f"Loading animated emotes from: {emotes_base_folder}")

        if not os.path.exists(emotes_base_folder):
            print(f"Error: Emote assets directory not found at {emotes_base_folder}")
            print("Please ensure you have created 'assets/emotes' folder with subfolders for each emotion and uploaded PNG sequences.")
            return

        for label in self.emotion_labels:
            emotion_folder = os.path.join(emotes_base_folder, label)
            if os.path.exists(emotion_folder):
                # Use glob to find all PNG files in the folder, sorted numerically
                # Updated glob pattern to match potential naming like happy_01.png or happy1.png
                frame_paths = sorted(glob.glob(os.path.join(emotion_folder, f"{label}_*.png")) +
                                     glob.glob(os.path.join(emotion_folder, f"{label}*.png")))
                
                # Remove duplicates if both patterns match
                frame_paths = list(dict.fromkeys(frame_paths))
                frame_paths.sort() # Re-sort after deduping

                if not frame_paths:
                    print(f"Warning: No animation frames found for '{label}' in {emotion_folder}. Skipping.")
                    continue

                frames = []
                for frame_path in frame_paths:
                    emote_img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
                    if emote_img is not None:
                        emote_img_resized = cv2.resize(emote_img, self.target_size, interpolation=cv2.INTER_AREA)
                        frames.append(emote_img_resized)
                    else:
                        print(f"Warning: Could not load frame {frame_path} for '{label}'. Check file integrity.")
                
                if frames:
                    _EMOTE_ANIMATIONS[label] = frames
                else:
                    print(f"Warning: No valid frames loaded for '{label}'. Skipping this animation.")
            else:
                print(f"Warning: Folder for '{label}' not found at {emotion_folder}. Skipping this animation.")

        if not _EMOTE_ANIMATIONS:
            print("No animated emotes loaded. Emote mirroring will not function.")
        else:
            total_frames = sum(len(frames) for frames in _EMOTE_ANIMATIONS.values())
            print(f"Loaded animations for {len(_EMOTE_ANIMATIONS)} emotions, total frames: {total_frames}.")

    def get_current_emote_frame(self, emotion_label):
        """
        Retrieves the current animation frame for a given emotion.
        Cycles through frames based on a global counter.
        """
        animation_frames = _EMOTE_ANIMATIONS.get(emotion_label)
        if not animation_frames:
            # Fallback to neutral if specific emote not found, or return None if neutral also not found
            fallback_animation = _EMOTE_ANIMATIONS.get('neutral')
            if fallback_animation:
                animation_frames = fallback_animation
                # print(f"Using fallback 'neutral' emote for '{emotion_label}'.")
            else:
                return None # No emote available

        # Update the global frame counter
        self.frame_counter = (self.frame_counter + 1) % (len(animation_frames) * _ANIMATION_SKIP_FRAMES)
        current_index_in_animation = (self.frame_counter // _ANIMATION_SKIP_FRAMES) % len(animation_frames)

        return animation_frames[current_index_in_animation]

def overlay_transparent_image(background, overlay_image, x_offset, y_offset, scale=1.0):
    """
    Overlays a transparent PNG image onto a background image.
    Args:
        background (np.array): The background image (typically 3 channels BGR).
        overlay_image (np.array): The foreground image to overlay (4 channels BGRA).
        x_offset (int): X-coordinate of the top-left corner where the overlay starts.
        y_offset (int): Y-coordinate of the top-left corner where the overlay starts.
        scale (float): Scale factor for the overlay image (1.0 means no scaling beyond initial target_size).
    Returns:
        np.array: The background image with the overlay applied.
    """
    if overlay_image is None or background is None:
        return background

    # Ensure background is BGR, not BGRA
    if background.shape[2] == 4:
        background = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)

    # Resize overlay if scale is not 1.0
    if scale != 1.0:
        new_width = int(overlay_image.shape[1] * scale)
        new_height = int(overlay_image.shape[0] * scale)
        overlay_image = cv2.resize(overlay_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    h, w, _ = background.shape
    h_overlay, w_overlay = overlay_image.shape[0], overlay_image.shape[1]

    # Extract the alpha mask and the BGR channels from the overlay image
    if overlay_image.shape[2] == 4:
        b, g, r, a = cv2.split(overlay_image)
        overlay_rgb = cv2.merge((b, g, r))
        alpha_mask = a / 255.0 # Normalize alpha to 0-1
    else: # Assume no alpha, treat as opaque
        overlay_rgb = overlay_image
        alpha_mask = np.ones((h_overlay, w_overlay), dtype=np.float32)

    # Calculate ROI coordinates, ensuring they stay within background bounds
    x1, x2 = max(0, x_offset), min(w, x_offset + w_overlay)
    y1, y2 = max(0, y_offset), min(h, y_offset + h_overlay)

    # Calculate overlay dimensions within ROI
    w_effective = x2 - x1
    h_effective = y2 - y1

    if w_effective <= 0 or h_effective <= 0:
        return background # Overlay completely outside bounds

    # Adjust overlay and alpha_mask if clipped
    overlay_rgb_cropped = overlay_rgb[0:h_effective, 0:w_effective]
    alpha_mask_cropped = alpha_mask[0:h_effective, 0:w_effective]

    # Create inverse alpha mask
    alpha_inv = 1.0 - alpha_mask_cropped

    # Apply alpha blending
    background_roi = background[y1:y2, x1:x2]

    # Ensure background_roi and overlay_rgb_cropped have the same depth (number of channels)
    # This loop assumes 3 channels for background_roi and overlay_rgb_cropped
    for c in range(0, 3):
        background_roi[:, :, c] = (background_roi[:, :, c] * alpha_inv +
                                   overlay_rgb_cropped[:, :, c] * alpha_mask_cropped)

    background[y1:y2, x1:x2] = background_roi

    return background


if __name__ == '__main__':
    # Test loading and overlaying (requires dummy assets/emotes structure)
    print("Running emotemirroring.py directly for demonstration of animation loading.")
    MAIN_PROJECT_PATH_TEST = '/content/drive/MyDrive/Project/Face Emotion Recoginization /face' # UPDATED PATH
    
    # Create dummy background image
    dummy_bg = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_bg, "Background", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Create dummy assets/emotes directory and dummy frames for 'happy'
    dummy_emotes_dir = os.path.join(MAIN_PROJECT_PATH_TEST, EMOTE_ASSET_DIR_RELATIVE)
    os.makedirs(os.path.join(dummy_emotes_dir, 'happy'), exist_ok=True)
    
    # Create 3 dummy frames for 'happy'
    for i in range(1, 4):
        dummy_happy_path = os.path.join(dummy_emotes_dir, 'happy', f'happy_0{i}.png')
        if not os.path.exists(dummy_happy_path):
            dummy_emote = np.zeros((100, 100, 4), dtype=np.uint8)
            cv2.circle(dummy_emote, (50, 50), 40, (0, 255 // i, 0, 200), -1) # Vary color slightly
            cv2.putText(dummy_emote, str(i), (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 2)
            cv2.imwrite(dummy_happy_path, dummy_emote)
    
    # Ensure there's a 'neutral' folder and dummy frame for fallback
    os.makedirs(os.path.join(dummy_emotes_dir, 'neutral'), exist_ok=True)
    dummy_neutral_path = os.path.join(dummy_emotes_dir, 'neutral', 'neutral_01.png')
    if not os.path.exists(dummy_neutral_path):
        dummy_neutral_emote = np.zeros((100, 100, 4), dtype=np.uint8)
        cv2.putText(dummy_neutral_emote, "N", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 2)
        cv2.imwrite(dummy_neutral_path, dummy_neutral_emote)


    # Initialize animator
    animator = EmoteAnimator(MAIN_PROJECT_PATH_TEST, ['happy', 'neutral'], target_size=(100, 100)) # Only happy and neutral for test
    
    # Simulate animation
    print("\nSimulating emote animation (check output for frame progression):")
    from google.colab.patches import cv2_imshow # For displaying in Colab
    for _ in range(10): # Show 10 frames
        current_emote_frame = animator.get_current_emote_frame('happy')
        if current_emote_frame is not None:
            output_frame = overlay_transparent_image(dummy_bg.copy(), current_emote_frame, x_offset=50, y_offset=50)
            cv2_imshow(output_frame)
            time.sleep(0.5) # Pause to see frames

    print("Animation simulation complete.")