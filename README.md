# Person Counter with Custom Line Drawing

An OpenCV-based person counter that allows you to draw a custom counting line on your video. Track people crossing the line with real-time IN/OUT counts.

## Features

- Upload any video file for analysis
- Draw custom counting line by clicking two points
- Works with any line orientation (horizontal, vertical, diagonal)
- Real-time person detection using background subtraction
- Object tracking to prevent duplicate counts
- Visual display of:
  - IN count (green) - people crossing one direction
  - OUT count (red) - people crossing opposite direction
  - Total people inside (IN - OUT)
  - Yellow counting line
  - Tracked objects with IDs
  - Bounding boxes around detected persons

## Installation

Install the required dependencies:
```bash
pip install opencv-python numpy
```

## Usage

### Running the Application

```bash
python person_counter.py path/to/your/video.mp4
```

### Examples:

```bash
# Windows path
python person_counter.py C:\Users\Videos\people_walking.mp4

# Relative path
python person_counter.py video.mp4

# Linux/Mac path
python person_counter.py /home/user/videos/crowd.mp4
```

### Steps to Use:

1. **Start the application** with your video file
2. **Click TWO points** on the video window to define your counting line
   - Click anywhere on the video for the first point
   - Click another location for the second point
   - The yellow line will appear connecting these points
3. **Watch the counts** update in real-time at the top-left corner
4. Use the controls to manage the session

## Controls

| Key | Action |
|-----|--------|
| **SPACE** | Pause/Resume video playback |
| **R** | Redraw counting line (click two new points) |
| **S** | Reset counters to zero |
| **Q** | Quit the application |

## How It Works

### Detection & Tracking
1. **Background Subtraction**: Uses MOG2 algorithm to detect moving objects
2. **Contour Detection**: Identifies shapes and filters by area/aspect ratio
3. **Object Tracking**: Tracks centroids of detected persons across frames
4. **Line Crossing Detection**: Monitors when tracked objects cross your custom line

### Counting Logic
- **IN count**: Increases when a person crosses from one side to the other
- **OUT count**: Increases when a person crosses in the opposite direction
- **Duplicate Prevention**: Each tracked object is only counted once per crossing

### Line Drawing
- Draw the line in **any direction**: horizontal, vertical, or diagonal
- The algorithm calculates which side of the line a person is on
- Crossing detection works regardless of line orientation

## Customization

You can adjust these parameters in the code ([person_counter.py](person_counter.py)):

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `self.line_offset` | Line 35 | 15 | Detection zone around the line (pixels) |
| `self.min_area` | Line 48 | 800 | Minimum contour area to be considered a person |
| `self.max_disappeared` | Line 47 | 30 | Max frames an object can disappear before removal |
| `aspect_ratio` threshold | Line 207 | 2.0 | Max width/height ratio for person detection |

## Tips for Best Results

1. **Camera Position**:
   - Static camera works best
   - Position camera so people walk across the line clearly

2. **Line Placement**:
   - Draw the line perpendicular to the direction of movement
   - Place it where people are clearly visible
   - Avoid areas with shadows or reflections

3. **Video Quality**:
   - Good lighting conditions improve detection
   - Higher resolution videos work better
   - Avoid shaky or moving camera footage

4. **Calibration**:
   - Let the video play for a few seconds before drawing the line
   - This allows the background subtractor to learn the background
   - Use the pause feature (SPACE) to carefully position the line

## Display Information

### Top-Left Panel:
- **IN**: Count in green (people crossing one direction)
- **OUT**: Count in red (people crossing opposite direction)
- **Inside**: Net count (IN - OUT)

### On-Screen Elements:
- **Yellow line**: Your custom counting boundary
- **Green boxes**: Detected persons
- **Blue dots**: Tracked centroid with ID number
- **IN/OUT labels**: Direction indicators near the line

## Troubleshooting

### No detections appearing
- Adjust `self.min_area` (smaller value = more sensitive)
- Check video has actual movement
- Ensure good lighting in video

### False detections
- Increase `self.min_area` (larger value = less sensitive)
- Check `aspect_ratio` threshold for filtering

### Counts seem inaccurate
- Redraw the line perpendicular to movement
- Adjust `self.line_offset` for larger detection zone
- Increase `self.max_disappeared` for slower-moving people

### Video won't open
- Check file path is correct
- Ensure OpenCV supports the video format
- Try converting to MP4 format

## Video Replay

When the video ends, you'll be prompted to replay:
- Type 'y' to restart from the beginning (keeps counters)
- Type 'n' to exit the application

## Output

The application prints to console:
- Video information (dimensions, FPS)
- Real-time crossing events ("Person IN" / "Person OUT")
- Final count when you quit

Example output:
```
============================================================
PERSON COUNTER - Video Mode
============================================================
Video: people.mp4
Dimensions: 1920x1080
FPS: 30

Point 1 set at: (450, 320)
Point 2 set at: (1200, 320)
Counting line defined! Processing will start.

Person IN - Total IN: 1
Person OUT - Total OUT: 1
Person IN - Total IN: 2
...

Session ended - Final Count: IN=15, OUT=8
```

## Requirements

- Python 3.7+
- OpenCV (opencv-python) 4.8.0+
- NumPy 1.24.0+

## License

Free to use for personal and commercial projects.
