import numpy as np
import json

class EpisodicLoader:
    def __init__(self, motion_file_path):
        """
        Loads episodic motion data from a JSON file and provides frame access by index.
        
        Args:
            motion_file_path: Path to a JSON reference motion file
        """
        self.fps = None
        self.frame_offsets = None
        self.motion_data = None
        self.slices = {}  # Dictionary to store slices for different motion components
    
        # Load the motion data from JSON
        self._load_from_json(motion_file_path)
        self.n_frames = self.motion_data.shape[0]
        
        print(f"[EpisodicLoader] Loaded motion data with {self.n_frames} frames")
    
    def _load_from_json(self, file_path):
        """Load motion data directly from a JSON reference motion file"""
        data = json.load(open(file_path))
        _Y = np.array(data["Frames"])
        self.fps = data["FPS"]
        
        frame_offsets = data["Frame_offset"][0]
        frame_sizes = data["Frame_size"][0]
            
        # Create slices
        for key in frame_offsets.keys():
            self.slices[key] = slice(frame_offsets[key], frame_offsets[key] + frame_sizes[key])
        
        print("[EpisodicLoader] Created slices:", self.slices)
        
        # Store the full motion data
        self.motion_data = np.array(_Y)
    
    def get_reference_motion(self, frame_index):
        """
        Get the reference motion data for a specific frame index.
        If the index exceeds the available frames, it will wrap around.
        
        Args:
            frame_index: Integer index of the desired frame
        
        Returns:
            JAX array containing the motion data for the requested frame
        """
        # Ensure the frame index is within bounds using modulo
        valid_index = frame_index % self.n_frames
        return self.motion_data[valid_index]
        
    def get_motion_steps_in_period(self):
        """Returns the total number of frames in the motion"""
        return self.n_frames


if __name__ == "__main__":
    # Example usage
    loader = EpisodicLoader("path/to/reference_motion.json")
    
    # Get a specific frame
    frame_data = loader.get_reference_motion(10)
    print(f"Frame data shape: {frame_data.shape}")
    
    # Plot a specific dimension over time
    import matplotlib.pyplot as plt
    
    select_dim = 0  # Choose which dimension to plot
    frames = []
    for i in range(loader.get_motion_steps_in_period()):
        frames.append(loader.get_reference_motion(i)[select_dim])
    
    plt.plot(range(loader.get_motion_steps_in_period()), frames)
    plt.title(f"Motion data dimension {select_dim}")
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.show()

