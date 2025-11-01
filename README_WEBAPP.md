# Person Counter Web Application

A simple web application for counting people using YOLO with zone detection and line counting features.

## Features

- **Video Upload**: Upload any video file (MP4, AVI, MOV, MKV)
- **Zone Detection**: Draw polygons to define zones and track which person IDs are inside
- **Line Counting**: Draw lines to count people crossing IN and OUT
- **Real-time Processing**: YOLO-based person detection and tracking
- **Download Results**: Get the processed video with all annotations

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and go to:
```
http://localhost:5000
```

## How to Use

### 1. Upload Video
- Click "Choose Video File" and select your video
- Wait for the upload to complete
- You'll see a preview of the first frame

### 2. Draw Zones (Polygons)
- Enter a name for the zone (e.g., "Entry Area", "Exit Zone")
- Click "Add Zone (Polygon)"
- Click on the video preview to add points for your polygon
- Right-click when done to complete the polygon
- The zone will detect which person IDs are inside it

### 3. Draw Counting Lines
- Enter a name for the line (e.g., "Main Entrance", "Gate 1")
- Click "Add Counting Line"
- Click to set the start point
- Click to set the end point
- The line will count people crossing IN and OUT

### 4. Process Video
- Click "🚀 Process Video" when ready
- Wait for processing (may take several minutes depending on video length)
- The processed video will automatically download when complete

## Features Explained

### Zone Detection
- **Purpose**: Track which person IDs are currently inside a defined area
- **How it works**: Uses point-in-polygon algorithm to check if person centroids are inside zones
- **Output**: Shows count of IDs in each zone on the video

### Line Counting
- **Purpose**: Count people crossing a line in both directions
- **How it works**: Tracks when person centroids cross from one side to another
- **Output**: Shows IN and OUT counts for each line

## Tips

- Draw multiple zones to monitor different areas
- Draw multiple lines to count at different entry/exit points
- Use descriptive names for zones and lines
- The first frame preview helps you position zones and lines accurately
- Processing time depends on video length and resolution

## File Structure

```
OpenCV/
├── app.py                  # Flask application
├── templates/
│   └── index.html         # Web interface
├── uploads/               # Uploaded videos (created automatically)
├── outputs/               # Processed videos (created automatically)
├── static/                # Preview images (created automatically)
└── requirements.txt       # Python dependencies
```

## Troubleshooting

**Video won't upload:**
- Check file size (max 500MB)
- Ensure file format is supported (MP4, AVI, MOV, MKV)

**Processing takes too long:**
- Video processing is CPU/GPU intensive
- Larger videos take longer
- Consider using shorter clips for testing

**Can't draw zones/lines:**
- Make sure you've uploaded a video first
- Check that you entered a name before clicking Add Zone/Line

## Technology Stack

- **Backend**: Flask (Python)
- **Computer Vision**: OpenCV + Ultralytics YOLO
- **Frontend**: Vanilla JavaScript + HTML5 Canvas
- **Tracking**: ByteTrack algorithm
