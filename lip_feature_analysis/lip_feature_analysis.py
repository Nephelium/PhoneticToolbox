import sys
import os
import traceback

# Setup logging to file for debugging startup issues
log_file = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "startup_log.txt")
def log(msg):
    with open(log_file, "a", encoding='utf-8') as f:
        f.write(msg + "\n")

log("Starting application...")
log(f"Python version: {sys.version}")
log(f"Executable: {sys.executable}")
log(f"CWD: {os.getcwd()}")

# Fix for PyInstaller and MediaPipe DLL load error
if getattr(sys, 'frozen', False):
    log("Running in frozen mode")
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
        log(f"MEIPASS: {base_path}")
    else:
        base_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        log(f"Base path (onedir): {base_path}")
    
    if os.name == 'nt':
        # Add mediapipe/python to DLL search path
        # Try multiple potential paths for robustness
        paths_to_try = [
            os.path.join(base_path, 'mediapipe', 'python'),
            os.path.join(base_path, 'mediapipe'),
            os.path.join(base_path, 'cv2'),  # If opencv DLLs are here
            base_path # Root directory
        ]
        
        # FIX: Path separator normalization for MediaPipe resource loading
        # MediaPipe sometimes fails if paths have mixed slashes or if the resource path logic 
        # doesn't handle the frozen environment correctly.
        # We can try to monkey patch resource loading if needed, but first let's ensure
        # the environment variables are set correctly.
        
        # Ensure PATH includes these directories
        os.environ['PATH'] = os.pathsep.join(paths_to_try) + os.pathsep + os.environ['PATH']
        
        # MONKEY PATCH: Fix MediaPipe's resource loading in Windows Frozen environment with Chinese paths
        # The C++ layer of MediaPipe cannot handle paths with non-ASCII characters on Windows.
        # We need to copy the model files to a temporary directory (ASCII path) and redirect MediaPipe to look there.
        try:
            import mediapipe.python.solution_base as mp_solution_base
            import mediapipe.python.solutions.face_mesh as mp_face_mesh_module
            import mediapipe.python.solutions.face_detection as mp_face_detection_module
            import shutil
            import tempfile
            
            def fix_mediapipe_resources():
                if not getattr(sys, 'frozen', False):
                    return

                log("Attempting to fix MediaPipe resources for frozen environment...")
                
                # Determine the base directory where 'mediapipe' folder is located
                if hasattr(sys, '_MEIPASS'):
                    base_dir = sys._MEIPASS
                else:
                    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
                    if os.path.exists(os.path.join(base_dir, '_internal')):
                        base_dir = os.path.join(base_dir, '_internal')
                
                mp_source_modules = os.path.join(base_dir, 'mediapipe', 'modules')
                
                if not os.path.exists(mp_source_modules):
                    log(f"WARNING: MediaPipe modules source not found at {mp_source_modules}")
                    return

                # Create a temp directory with a short, ASCII-only name
                temp_base = os.path.join(tempfile.gettempdir(), 'mp_res_fix')
                mp_target_modules = os.path.join(temp_base, 'mediapipe', 'modules')
                
                log(f"Target temp resource dir: {temp_base}")
                
                # Copy modules if they don't exist
                if not os.path.exists(mp_target_modules):
                    log(f"Copying modules from {mp_source_modules} to {mp_target_modules}")
                    try:
                        shutil.copytree(mp_source_modules, mp_target_modules, dirs_exist_ok=True)
                        log("Copy completed successfully")
                    except Exception as e:
                        log(f"Failed to copy modules: {e}")
                        return
                else:
                    log("Modules already exist in temp, skipping copy")

                # 1. Monkey patch set_resource_dir to point to our temp base
                # This ensures that internal resources (like .tflite files referenced in graphs) are found
                original_set_resource_dir = mp_solution_base.resource_util.set_resource_dir
                
                def patched_set_resource_dir(path):
                    log(f"Intercepted set_resource_dir. Redirecting to: {temp_base}")
                    return original_set_resource_dir(temp_base)
                
                mp_solution_base.resource_util.set_resource_dir = patched_set_resource_dir
                log("Successfully patched resource_util.set_resource_dir")
                
                # 2. Update binary graph paths to be absolute paths in the temp directory
                # This ensures that SolutionBase loads the main graph from the ASCII path
                
                # Fix Face Mesh path
                # Original is relative: mediapipe/modules/face_landmark/face_landmark_front_cpu.binarypb
                if hasattr(mp_face_mesh_module, '_BINARYPB_FILE_PATH'):
                    rel_path = mp_face_mesh_module._BINARYPB_FILE_PATH
                    # Remove 'mediapipe/modules/' prefix if present to join correctly with our target
                    # Actually, our structure in temp is temp_base/mediapipe/modules/...
                    # And _BINARYPB_FILE_PATH includes 'mediapipe/modules/...'
                    # So we just join temp_base with the relative path
                    abs_path = os.path.join(temp_base, rel_path)
                    abs_path = os.path.normpath(abs_path)
                    mp_face_mesh_module._BINARYPB_FILE_PATH = abs_path
                    log(f"Patched FaceMesh binary path to: {abs_path}")
                
                # Fix Face Detection paths (used internally by FaceMesh)
                if hasattr(mp_face_detection_module, '_SHORT_RANGE_GRAPH_FILE_PATH'):
                    rel_path = mp_face_detection_module._SHORT_RANGE_GRAPH_FILE_PATH
                    abs_path = os.path.join(temp_base, rel_path)
                    abs_path = os.path.normpath(abs_path)
                    mp_face_detection_module._SHORT_RANGE_GRAPH_FILE_PATH = abs_path
                    log(f"Patched FaceDetection short range path to: {abs_path}")
                    
                if hasattr(mp_face_detection_module, '_FULL_RANGE_GRAPH_FILE_PATH'):
                    rel_path = mp_face_detection_module._FULL_RANGE_GRAPH_FILE_PATH
                    abs_path = os.path.join(temp_base, rel_path)
                    abs_path = os.path.normpath(abs_path)
                    mp_face_detection_module._FULL_RANGE_GRAPH_FILE_PATH = abs_path
                    log(f"Patched FaceDetection full range path to: {abs_path}")

            fix_mediapipe_resources()
            
        except Exception as e:
            log(f"Failed to apply MediaPipe resource fix: {e}")
            log(traceback.format_exc())
            
        except Exception as e:
            log(f"Failed to patch MediaPipe path: {e}")
        
        for p in paths_to_try:
            if os.path.exists(p):
                try:
                    os.add_dll_directory(p)
                    log(f"Added DLL directory: {p}")
                    # Also list files in this directory to verify existence of key DLLs
                    files = os.listdir(p)
                    dlls = [f for f in files if f.lower().endswith('.dll') or f.lower().endswith('.pyd')]
                    log(f"Files in {p}: {dlls}")
                except Exception as e:
                    log(f"Failed to add DLL directory {p}: {e}")
            else:
                log(f"Directory not found: {p}")
        
        # Also add to PATH environment variable as fallback
        # os.environ['PATH'] = os.pathsep.join(paths_to_try) + os.pathsep + os.environ['PATH'] # Already done above

try:
    log("Attempting to import mediapipe...")
    import mediapipe as mp
    log("MediaPipe imported successfully")
except Exception as e:
    log(f"CRITICAL ERROR importing mediapipe: {e}")
    log(traceback.format_exc())
    # Don't exit yet, let the error propagate or show in GUI if possible, 
    # but for now we want to catch it to log it.
    # We re-raise it so the app crashes 'normally' if we can't recover,
    # but at least we have the log.
    raise e

import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import wave
import threading
import time
import os
from datetime import datetime
import pickle
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2

# Parse command line arguments
parser = argparse.ArgumentParser(description='Lip feature tracking application')
parser.add_argument('--load', action='store_true', help='Load existing data instead of recording new data')
args = parser.parse_args()

# Function to select a directory containing data or create a new one
def select_or_create_directory():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True) # Ensure dialog is on top
    
    if args.load:
        # If loading data, prompt user to select a directory
        load_dir = filedialog.askdirectory(title="Select directory containing data")
        root.destroy()
        
        if not load_dir:
            print("No directory selected. Exiting...")
            exit()
        
        return load_dir
    else:
        # If recording new data, ask user for the parent directory to save data
        parent_dir = filedialog.askdirectory(title="请选择保存数据的文件夹 (Select Save Folder)")
        root.destroy()
        
        if not parent_dir:
            print("No directory selected.")
            return None
            
        base_dir_name = f"lip_tracking_data_{datetime.now().strftime('%Y%m%d')}"
        save_dir = os.path.join(parent_dir, base_dir_name)
        
        # 检查是否存在同名文件夹，如果存在则添加序号
        counter = 1
        original_save_dir = save_dir
        while os.path.exists(save_dir):
            save_dir = f"{original_save_dir}_{counter}"
            counter += 1
            
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

# Get the data directory
if args.load:
    data_dir = select_or_create_directory()
else:
    data_dir = None


# Setup MediaPipe Face Mesh - MOVED TO main() to catch initialization errors
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = None # Will be initialized in main

# Lip landmarks indices based on MediaPipe Face Mesh
# Outer lip landmarks
OUTER_LIP_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0]
# Inner lip landmarks
INNER_LIP_LANDMARKS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13]
# Face contour landmarks (for face area calculation)
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

# Reference points for face width calculation
LEFT_FACE = 234  # Left side of face
RIGHT_FACE = 454  # Right side of face
TOP_FACE = 10  # Top of face
BOTTOM_FACE = 152  # Bottom of face

# Audio recording settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Initialize variables for tracking
frame_times = []  # Absolute timestamps for each frame
relative_times = []  # Relative times from start of recording
area_values = []
height_values = []
width_values = []
inner_width_values = []
total_width_values = []
open_values = []
length_values = []
circularity_values = []
face_width_values = []
face_height_values = []
normalized_height_values = []
normalized_width_values = []
normalized_inner_width_values = []
normalized_total_width_values = []
normalized_open_values = []
landmark_data = []

# Global flag for recording control
is_recording = False

# Function to calculate polygon area
def polygon_area(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# Function to calculate polygon perimeter
def polygon_perimeter(vertices):
    perimeter = 0
    for i in range(len(vertices)):
        perimeter += np.linalg.norm(vertices[i] - vertices[(i + 1) % len(vertices)])
    return perimeter

# Audio recording class to control the recording process
class AudioRecorder:
    def __init__(self, output_file):
        self.output_file = output_file
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.thread = None
        self.start_time = None
        self.frame_timestamps = []  # To store timestamps for each audio chunk
    
    def start_recording(self):
        self.is_recording = True
        self.frames = []
        self.frame_timestamps = []
        self.start_time = time.time()
        self.stream = self.audio.open(
            format=FORMAT, 
            channels=CHANNELS,
            rate=RATE, 
            input=True,
            frames_per_buffer=CHUNK
        )
        self.thread = threading.Thread(target=self._record)
        self.thread.start()
        print("Started audio recording")
    
    def _record(self):
        while self.is_recording:
            try:
                data = self.stream.read(CHUNK)
            except (IOError, AttributeError) as e:
                print(f"读取音频数据时出错: {e}")
                data = b''  # 返回空字节作为后备
            timestamp = time.time()
            self.frames.append(data)
            self.frame_timestamps.append(timestamp)
    
    def stop_recording(self):
        if not self.is_recording:
            return
            
        self.is_recording = False
        if self.thread:
            self.thread.join()
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        # Save the audio file
        wf = wave.open(self.output_file, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        
        # Save audio timestamps
        timestamps_file = os.path.splitext(self.output_file)[0] + "_timestamps.pkl"
        with open(timestamps_file, 'wb') as f:
            pickle.dump({
                'start_time': self.start_time,
                'frame_timestamps': self.frame_timestamps,
                'sample_rate': RATE,
                'chunk_size': CHUNK
            }, f)
        
        print(f"Audio saved to {self.output_file}")
        print(f"Audio timestamps saved to {timestamps_file}")
    
    def __del__(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

# Function to save all collected data in a format that's easy to segment
def save_data(directory):
    data = {
        'absolute_timestamps': frame_times,  # Absolute timestamps for sync
        'relative_times': relative_times,    # Relative times from start
        'area': area_values,
        'face_width': face_width_values,
        'face_height': face_height_values,
        # Raw pixel values
        'height_px': height_values,
        'outer_width_px': width_values,
        'inner_width_px': inner_width_values,
        'total_width_px': total_width_values,
        'open_px': open_values,
        'length': length_values,
        # Normalized values
        'height': normalized_height_values,
        'outer_width': normalized_width_values,
        'inner_width': normalized_inner_width_values,
        'total_width': normalized_total_width_values,
        'open': normalized_open_values,
        'circularity': circularity_values,
        'landmarks': landmark_data,
        'metadata': {
            'recording_start_time': frame_times[0] if frame_times else None,
            'recording_duration': relative_times[-1] if relative_times else None,
            'fps': len(frame_times) / relative_times[-1] if frame_times and relative_times else None,
            'created_at': datetime.now().isoformat()
        }
    }
    
    data_file = os.path.join(directory, "audio_recording.pkl")
    with open(data_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Metric data saved to {data_file}")

# Function to load saved data
def load_data(directory):
    global frame_times, relative_times, area_values, height_values, width_values, inner_width_values
    global total_width_values, open_values, length_values, circularity_values
    global face_width_values, face_height_values, normalized_height_values
    global normalized_width_values, normalized_inner_width_values
    global normalized_total_width_values, normalized_open_values, landmark_data
    
    data_file = os.path.join(directory, "audio_recording.pkl")
    
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded data from {data_file}")
        
        # Extract data from the loaded dictionary
        frame_times = data.get('absolute_timestamps', [])
        relative_times = data.get('relative_times', [])
        area_values = data.get('area', [])
        face_width_values = data.get('face_width', [])
        face_height_values = data.get('face_height', [])
        height_values = data.get('height_px', [])
        width_values = data.get('outer_width_px', [])
        inner_width_values = data.get('inner_width_px', [])
        total_width_values = data.get('total_width_px', [])
        open_values = data.get('open_px', [])
        length_values = data.get('length', [])
        normalized_height_values = data.get('height', [])
        normalized_width_values = data.get('outer_width', [])
        normalized_inner_width_values = data.get('inner_width', [])
        normalized_total_width_values = data.get('total_width', [])
        normalized_open_values = data.get('open', [])
        circularity_values = data.get('circularity', [])
        landmark_data = data.get('landmarks', [])
        
        # Print metadata if available
        if 'metadata' in data:
            metadata = data['metadata']
            print(f"Recording start time: {metadata.get('recording_start_time')}")
            print(f"Recording duration: {metadata.get('recording_duration')} seconds")
            print(f"Approximate FPS: {metadata.get('fps')}")
        
        return True
    else:
        print(f"Data file {data_file} not found")
        return False

# Function to plot all metrics
def plot_metrics():
    # Plot normalized metrics
    plt.figure(figsize=(15, 9))
    
    plt.subplot(3, 2, 1)
    plt.plot(relative_times, area_values)
    plt.title('Lip Area Ratio')
    plt.xlabel('Time (s)')
    plt.ylabel('Area Ratio')
    
    plt.subplot(3, 2, 2)
    plt.plot(relative_times, normalized_height_values)
    plt.title('Normalized Lip Height')
    plt.xlabel('Time (s)')
    plt.ylabel('Height (ratio to face height)')
    
    plt.subplot(3, 2, 3)
    plt.plot(relative_times, normalized_width_values)
    plt.title('Normalized Outer Lip Width')
    plt.xlabel('Time (s)')
    plt.ylabel('Width (ratio to face width)')
    
    plt.subplot(3, 2, 4)
    plt.plot(relative_times, normalized_open_values)
    plt.title('Normalized Lip Openness')
    plt.xlabel('Time (s)')
    plt.ylabel('Opening (ratio to face height)')
    
    plt.subplot(3, 2, 5)
    plt.plot(relative_times, normalized_total_width_values)
    plt.title('Normalized Inner + Outer Lip Width')
    plt.xlabel('Time (s)')
    plt.ylabel('Total Width (ratio to face width)')
    
    plt.subplot(3, 2, 6)
    plt.plot(relative_times, circularity_values)
    plt.title('Lip Circularity')
    plt.xlabel('Time (s)')
    plt.ylabel('Circularity')
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "lip_metrics_plot.png"))
    # plt.show()  # Disabled showing plot
    plt.close()
    
    # Also plot raw pixel values for comparison
    plt.figure(figsize=(15, 9))
    
    plt.subplot(3, 2, 1)
    plt.plot(relative_times, face_width_values)
    plt.title('Face Width (pixels)')
    plt.xlabel('Time (s)')
    plt.ylabel('Pixels')
    
    plt.subplot(3, 2, 2)
    plt.plot(relative_times, height_values)
    plt.title('Lip Height (pixels)')
    plt.xlabel('Time (s)')
    plt.ylabel('Pixels')
    
    plt.subplot(3, 2, 3)
    plt.plot(relative_times, width_values)
    plt.title('Outer Lip Width (pixels)')
    plt.xlabel('Time (s)')
    plt.ylabel('Pixels')
    
    plt.subplot(3, 2, 4)
    plt.plot(relative_times, open_values)
    plt.title('Lip Openness (pixels)')
    plt.xlabel('Time (s)')
    plt.ylabel('Pixels')
    
    plt.subplot(3, 2, 5)
    plt.plot(relative_times, total_width_values)
    plt.title('Inner + Outer Lip Width (pixels)')
    plt.xlabel('Time (s)')
    plt.ylabel('Pixels')
    
    plt.subplot(3, 2, 6)
    plt.plot(relative_times, face_height_values)
    plt.title('Face Height (pixels)')
    plt.xlabel('Time (s)')
    plt.ylabel('Pixels')
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "raw_metrics_plot.png"))
    # plt.show()  # Disabled showing plot
    plt.close()

# Main function to handle recording or loading
def main():
    global face_mesh
    
    try:
        log("Initializing FaceMesh...")
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        log("FaceMesh initialized successfully")
    except Exception as e:
        log(f"Error initializing FaceMesh: {e}")
        log(traceback.format_exc())
        
        # Ensure we have a root window for messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", f"Failed to initialize MediaPipe FaceMesh.\n\n{e}")
        root.destroy()
        return

    if args.load:
        # Load existing data
        if not load_data(data_dir):
            print("Failed to load data")
            exit(1)
            
        # Plot the loaded data
        plot_metrics()
    else:
        # Record new data
        record_new_data()

# Function to record new data
def record_new_data():
    global is_recording, frame_times, relative_times, area_values, height_values, width_values, inner_width_values
    global total_width_values, open_values, length_values, circularity_values
    global face_width_values, face_height_values, normalized_height_values
    global normalized_width_values, normalized_inner_width_values
    global normalized_total_width_values, normalized_open_values, landmark_data
    global data_dir  # 添加全局变量data_dir
    
    # Initialize lists for this recording session
    recording_frame_times = []
    recording_relative_times = []
    recording_area_values = []
    recording_height_values = []
    recording_width_values = []
    recording_inner_width_values = []
    recording_total_width_values = []
    recording_open_values = []
    recording_length_values = []
    recording_circularity_values = []
    recording_face_width_values = []
    recording_face_height_values = []
    recording_normalized_height_values = []
    recording_normalized_width_values = []
    recording_normalized_inner_width_values = []
    recording_normalized_total_width_values = []
    recording_normalized_open_values = []
    recording_landmark_data = []
    
    # Set up audio recorder
    audio_recorder = None
    if data_dir is not None:
        audio_file = os.path.join(data_dir, "audio_recording.wav")
        audio_recorder = AudioRecorder(audio_file)
    
    # Video capture
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera")
        return
        
    h, w, _ = frame.shape
    
    # Create a control window
    cv2.namedWindow('MediaPipe Lip Tracking')
    print("Press 'r' to start/stop recording, 'q' to quit")
    
    start_time = None
    is_recording = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Display current status on frame
            status_text = "RECORDING" if is_recording else "STANDBY - Press 'r' to Record, 'q' to Quit"
            cv2.putText(frame, status_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (0, 0, 255) if is_recording else (255, 255, 255), 2)
            
            if is_recording:
                # Get the absolute timestamp for this frame
                current_absolute_time = time.time()
                
                if start_time is None:
                    start_time = current_absolute_time
                
                recording_frame_times.append(current_absolute_time)
                
                # Calculate relative time from start
                current_relative_time = current_absolute_time - start_time
                recording_relative_times.append(current_relative_time)
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                
                # Draw facial landmarks
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
                        
                        # Convert normalized landmarks to pixel coordinates
                        landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
                        
                        # Store all landmarks for future use
                        recording_landmark_data.append(landmarks)
                        
                        # Calculate face width and height for normalization
                        face_width = abs(landmarks[RIGHT_FACE][0] - landmarks[LEFT_FACE][0])
                        face_height = abs(landmarks[BOTTOM_FACE][1] - landmarks[TOP_FACE][1])
                        recording_face_width_values.append(face_width)
                        recording_face_height_values.append(face_height)
                        
                        # Extract lip landmarks
                        outer_lip = np.array([landmarks[i] for i in OUTER_LIP_LANDMARKS])
                        inner_lip = np.array([landmarks[i] for i in INNER_LIP_LANDMARKS])
                        face_oval = np.array([landmarks[i] for i in FACE_OVAL])
                        
                        # Calculate metrics
                        face_area = polygon_area(face_oval)
                        outer_lip_area = polygon_area(outer_lip)
                        inner_lip_area = polygon_area(inner_lip)
                        lip_area = outer_lip_area - inner_lip_area
                        
                        # Area ratio (already normalized by face area)
                        area_ratio = lip_area / face_area
                        recording_area_values.append(area_ratio)
                        
                        # Lip metrics in pixels
                        outer_lip_y = outer_lip[:, 1]
                        outer_lip_x = outer_lip[:, 0]
                        lip_height = max(outer_lip_y) - min(outer_lip_y)
                        outer_lip_width = max(outer_lip_x) - min(outer_lip_x)
                        
                        inner_lip_y = inner_lip[:, 1]
                        inner_lip_x = inner_lip[:, 0]
                        inner_lip_width = max(inner_lip_x) - min(inner_lip_x)
                        
                        total_width = outer_lip_width + inner_lip_width
                        
                        # Lip openness
                        top_lip_bottom = landmarks[13][1]  # Bottom point of top lip
                        bottom_lip_top = landmarks[14][1]  # Top point of bottom lip
                        lip_openness = bottom_lip_top - top_lip_bottom
                        
                        # Store raw pixel values
                        recording_height_values.append(lip_height)
                        recording_width_values.append(outer_lip_width)
                        recording_inner_width_values.append(inner_lip_width)
                        recording_total_width_values.append(total_width)
                        recording_open_values.append(lip_openness)
                        
                        # Normalize metrics by face dimensions
                        norm_height = lip_height / face_height
                        norm_outer_width = outer_lip_width / face_width
                        norm_inner_width = inner_lip_width / face_width
                        norm_total_width = total_width / face_width
                        norm_openness = lip_openness / face_height
                        
                        # Store normalized values
                        recording_normalized_height_values.append(norm_height)
                        recording_normalized_width_values.append(norm_outer_width)
                        recording_normalized_inner_width_values.append(norm_inner_width)
                        recording_normalized_total_width_values.append(norm_total_width)
                        recording_normalized_open_values.append(norm_openness)
                        
                        # Lip perimeter (calculating but not displaying in graph)
                        outer_lip_perimeter = polygon_perimeter(outer_lip)
                        recording_length_values.append(outer_lip_perimeter)
                        
                        # Circularity (4π * area / perimeter²) - a ratio, so no need to normalize
                        circularity = (4 * np.pi * lip_area) / (outer_lip_perimeter ** 2)
                        recording_circularity_values.append(circularity)
                        
                        # Display raw and normalized metrics on frame
                        metrics_text = f"Height: {norm_height:.3f}, Width: {norm_outer_width:.3f}, Area: {area_ratio:.5f}"
                        metrics_text2 = f"Open: {norm_openness:.3f}, Inner Width: {norm_inner_width:.3f}, Total Width: {norm_total_width:.3f}"
                        cv2.putText(frame, metrics_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, metrics_text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Face Width: {face_width:.0f}, Face Height: {face_height:.0f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Time: {current_relative_time:.2f}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # If no face detected, append NaN values to maintain time alignment
                    recording_area_values.append(np.nan)
                    recording_height_values.append(np.nan)
                    recording_width_values.append(np.nan)
                    recording_inner_width_values.append(np.nan)
                    recording_total_width_values.append(np.nan)
                    recording_open_values.append(np.nan)
                    recording_length_values.append(np.nan)
                    recording_circularity_values.append(np.nan)
                    recording_face_width_values.append(np.nan)
                    recording_face_height_values.append(np.nan)
                    recording_normalized_height_values.append(np.nan)
                    recording_normalized_width_values.append(np.nan)
                    recording_normalized_inner_width_values.append(np.nan)
                    recording_normalized_total_width_values.append(np.nan)
                    recording_normalized_open_values.append(np.nan)
                    recording_landmark_data.append(None)
            
            # Display the frame
            cv2.imshow('MediaPipe Lip Tracking', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):  # Toggle recording
                if not is_recording:
                    # Start recording
                    # 每次开始新录音时创建新文件夹
                    new_dir = select_or_create_directory()
                    
                    if new_dir:
                        data_dir = new_dir
                        is_recording = True
                        print(f"Recording to new directory: {data_dir}")
                        
                        # 更新音频文件路径
                        audio_file = os.path.join(data_dir, "audio_recording.wav")
                        audio_recorder = AudioRecorder(audio_file)
                        
                        print("Started recording")
                        start_time = None
                        audio_recorder.start_recording()
                        
                        # Reset recording lists
                        recording_frame_times = []
                        recording_relative_times = []
                        recording_area_values = []
                        recording_height_values = []
                        recording_width_values = []
                        recording_inner_width_values = []
                        recording_total_width_values = []
                        recording_open_values = []
                        recording_length_values = []
                        recording_circularity_values = []
                        recording_face_width_values = []
                        recording_face_height_values = []
                        recording_normalized_height_values = []
                        recording_normalized_width_values = []
                        recording_normalized_inner_width_values = []
                        recording_normalized_total_width_values = []
                        recording_normalized_open_values = []
                        recording_landmark_data = []
                    else:
                        print("Recording cancelled by user")
                else:
                    # Stop recording
                    is_recording = False
                    print("Stopped recording")
                    audio_recorder.stop_recording()
                    
                    # Copy recording data to global variables
                    frame_times = recording_frame_times
                    relative_times = recording_relative_times
                    area_values = recording_area_values
                    height_values = recording_height_values
                    width_values = recording_width_values
                    inner_width_values = recording_inner_width_values
                    total_width_values = recording_total_width_values
                    open_values = recording_open_values
                    length_values = recording_length_values
                    circularity_values = recording_circularity_values
                    face_width_values = recording_face_width_values
                    face_height_values = recording_face_height_values
                    normalized_height_values = recording_normalized_height_values
                    normalized_width_values = recording_normalized_width_values
                    normalized_inner_width_values = recording_normalized_inner_width_values
                    normalized_total_width_values = recording_normalized_total_width_values
                    normalized_open_values = recording_normalized_open_values
                    landmark_data = recording_landmark_data
                    
                    # Save the data
                    save_data(data_dir)
                    # Plot the data
                    if len(relative_times) > 0:
                        plot_metrics()
            
            elif key == ord('q'):  # Quit
                if is_recording:
                    print("Stopping recording before quit...")
                    if audio_recorder:
                        audio_recorder.stop_recording()
                    
                    # Copy recording data to global variables
                    frame_times = recording_frame_times
                    relative_times = recording_relative_times
                    area_values = recording_area_values
                    height_values = recording_height_values
                    width_values = recording_width_values
                    inner_width_values = recording_inner_width_values
                    total_width_values = recording_total_width_values
                    open_values = recording_open_values
                    length_values = recording_length_values
                    circularity_values = recording_circularity_values
                    face_width_values = recording_face_width_values
                    face_height_values = recording_face_height_values
                    normalized_height_values = recording_normalized_height_values
                    normalized_width_values = recording_normalized_width_values
                    normalized_inner_width_values = recording_normalized_inner_width_values
                    normalized_total_width_values = recording_normalized_total_width_values
                    normalized_open_values = recording_normalized_open_values
                    landmark_data = recording_landmark_data
                    
                    # Save the data
                    save_data(data_dir)
                    
                    # Turn off recording flag so finally block doesn't try to stop again
                    is_recording = False
                    
                break
    
    finally:
        # Clean up
        if is_recording and audio_recorder:
            audio_recorder.stop_recording()
            
        cap.release()
        cv2.destroyAllWindows()
        print("Recording session ended")
        

# Utility function to segment data based on time range
def segment_data(start_time, end_time, data_dir=None):
    """
    Extract a segment of the lip tracking data between start_time and end_time.
    
    Parameters:
    - start_time: Start time in seconds (relative to recording start)
    - end_time: End time in seconds (relative to recording start)
    - data_dir: Directory containing the data files (defaults to current data_dir)
    
    Returns:
    - Dictionary with the segmented data
    """
    if data_dir is None:
        data_dir = data_dir
        
    # Load the data if not already loaded
    if not relative_times:
        if not load_data(data_dir):
            print("Failed to load data for segmentation")
            return None
    
    # Find indices for the segment
    start_idx = next((i for i, t in enumerate(relative_times) if t >= start_time), 0)
    end_idx = next((i for i, t in enumerate(relative_times) if t > end_time), len(relative_times))
    
    # Create a new dictionary with the segmented data
    segmented_data = {
        'absolute_timestamps': frame_times[start_idx:end_idx],
        'relative_times': [t - relative_times[start_idx] for t in relative_times[start_idx:end_idx]],
        'area': area_values[start_idx:end_idx],
        'face_width': face_width_values[start_idx:end_idx],
        'face_height': face_height_values[start_idx:end_idx],
        'height_px': height_values[start_idx:end_idx],
        'outer_width_px': width_values[start_idx:end_idx],
        'inner_width_px': inner_width_values[start_idx:end_idx],
        'total_width_px': total_width_values[start_idx:end_idx],
        'open_px': open_values[start_idx:end_idx],
        'length': length_values[start_idx:end_idx],
        'height': normalized_height_values[start_idx:end_idx],
        'outer_width': normalized_width_values[start_idx:end_idx],
        'inner_width': normalized_inner_width_values[start_idx:end_idx],
        'total_width': normalized_total_width_values[start_idx:end_idx],
        'open': normalized_open_values[start_idx:end_idx],
        'circularity': circularity_values[start_idx:end_idx],
        'landmarks': landmark_data[start_idx:end_idx],
        'metadata': {
            'original_start_time': frame_times[0],
            'segment_start_time': frame_times[start_idx],
            'segment_end_time': frame_times[end_idx-1] if end_idx > 0 and end_idx <= len(frame_times) else None,
            'segment_duration': relative_times[end_idx-1] - relative_times[start_idx] if end_idx > 0 and end_idx <= len(relative_times) else None,
            'created_at': datetime.now().isoformat()
        }
    }
    
    return segmented_data

# Function to save a segment to a file
def save_segment(segmented_data, output_dir, filename="lip_metrics_segment.pkl"):
    """
    Save a segmented data dictionary to a file
    

    Parameters:
    - segmented_data: The dictionary with segmented data
    - output_dir: Directory to save the data
    - filename: Filename for the segmented data
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, filename)
    
    with open(output_file, 'wb') as f:
        pickle.dump(segmented_data, f)
    
    print(f"Segmented data saved to {output_file}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"Fatal error in main loop: {e}")
        log(traceback.format_exc())
        
        # Ensure we have a root window for messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Fatal Error", f"An unexpected error occurred:\n\n{e}")
        root.destroy()