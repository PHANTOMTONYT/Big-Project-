import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import supervision as sv
import json
from pathlib import Path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Create necessary folders
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(exist_ok=True)
Path('static').mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using ray casting algorithm"""
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def which_side_of_line(point, line_start, line_end):
    """Determine which side of the line a point is on"""
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    return ((x2 - x1) * (y0 - y1) - (y2 - y1) * (x0 - x1))

def process_video(video_path, zones, lines, output_path):
    """Process video with zone detection and line counting using Supervision"""

    # Initialize YOLO model
    model = YOLO('yolo11n.pt')

    # Open video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize Supervision annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.6)

    # Create Supervision zones
    sv_zones = {}
    zone_annotators = {}
    for zone in zones:
        polygon_points = np.array([[int(p['x'] * width), int(p['y'] * height)] for p in zone['points']], dtype=np.int32)
        sv_zones[zone['name']] = sv.PolygonZone(polygon=polygon_points)
        zone_annotators[zone['name']] = sv.PolygonZoneAnnotator(
            zone=sv_zones[zone['name']],
            color=sv.Color(r=zone['color'][2], g=zone['color'][1], b=zone['color'][0]),  # BGR to RGB
            thickness=3,
            text_thickness=2,
            text_scale=0.8
        )

    # Create Supervision line zones for counting
    sv_line_zones = {}
    line_annotators = {}
    for line in lines:
        start = sv.Point(int(line['start']['x'] * width), int(line['start']['y'] * height))
        end = sv.Point(int(line['end']['x'] * width), int(line['end']['y'] * height))
        sv_line_zones[line['name']] = sv.LineZone(start=start, end=end)
        line_annotators[line['name']] = sv.LineZoneAnnotator(
            thickness=3,
            text_thickness=2,
            text_scale=0.6,
            color=sv.Color(r=line['color'][2], g=line['color'][1], b=line['color'][0])
        )

    # Tracking data
    zone_ids = {zone['name']: set() for zone in zones}
    track_history = {}  # Store centroid history for trails

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {total_frames} frames with Supervision")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}/{total_frames}")

        # Run YOLO tracking
        results = model.track(
            frame,
            persist=True,
            conf=0.4,
            iou=0.5,
            classes=[0],  # Only persons
            verbose=False,
            tracker="bytetrack.yaml"
        )

        # Convert YOLO results to Supervision detections
        if results and len(results) > 0:
            detections = sv.Detections.from_ultralytics(results[0])

            # Annotate frame with boxes and labels
            if len(detections) > 0:
                labels = []
                for i in range(len(detections)):
                    tracker_id = detections.tracker_id[i] if detections.tracker_id is not None else None
                    conf = detections.confidence[i]
                    label = f"ID:{tracker_id} {conf:.2f}" if tracker_id else f"Person {conf:.2f}"
                    labels.append(label)

                # Draw boxes and labels using Supervision
                frame = box_annotator.annotate(scene=frame, detections=detections)
                frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

                # Draw centroids and tracking trails
                for i in range(len(detections)):
                    # Get bounding box
                    x1, y1, x2, y2 = detections.xyxy[i].astype(int)

                    # Calculate centroid
                    centroid_x = (x1 + x2) // 2
                    centroid_y = (y1 + y2) // 2
                    centroid = (centroid_x, centroid_y)

                    # Draw centroid as a circle
                    cv2.circle(frame, centroid, 6, (255, 0, 255), -1)  # Magenta circle
                    cv2.circle(frame, centroid, 8, (255, 255, 255), 2)  # White border

                    # Draw tracking trail if tracker_id exists
                    if detections.tracker_id is not None:
                        tracker_id = int(detections.tracker_id[i])

                        # Update history
                        if tracker_id not in track_history:
                            track_history[tracker_id] = []
                        track_history[tracker_id].append(centroid)

                        # Keep only last 30 points for trail
                        if len(track_history[tracker_id]) > 30:
                            track_history[tracker_id] = track_history[tracker_id][-30:]

                        # Draw trail
                        points = track_history[tracker_id]
                        if len(points) > 1:
                            for j in range(1, len(points)):
                                # Calculate thickness based on age (newer = thicker)
                                thickness = max(1, int(3 * (j / len(points))))
                                cv2.line(frame, points[j-1], points[j], (255, 0, 255), thickness)

                # Process zones with Supervision
                for zone_name, sv_zone in sv_zones.items():
                    # Trigger zone for current detections
                    sv_zone.trigger(detections=detections)

                    # Get tracker IDs in zone
                    mask = sv_zone.trigger(detections=detections)
                    zone_ids[zone_name] = set()
                    if detections.tracker_id is not None:
                        for idx, in_zone in enumerate(mask):
                            if in_zone:
                                zone_ids[zone_name].add(int(detections.tracker_id[idx]))

                # Process line zones with Supervision
                for line_name, sv_line_zone in sv_line_zones.items():
                    # Trigger line crossing and get counts
                    sv_line_zone.trigger(detections=detections)
                    # LineZone automatically tracks in/out counts internally

        # Draw zones with custom ACTIVE/NOT ACTIVE status
        for zone in zones:
            zone_name = zone['name']
            polygon = [(int(p['x'] * width), int(p['y'] * height)) for p in zone['points']]
            pts = np.array(polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))

            # Check if zone is active
            is_active = len(zone_ids[zone_name]) > 0

            # Choose color based on active status
            if is_active:
                border_color = (0, 255, 0)  # Green when active
                fill_color = (0, 255, 0)
            else:
                border_color = (0, 0, 255)  # Red when not active
                fill_color = (0, 0, 255)

            # Draw semi-transparent zone
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], fill_color)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            # Draw zone border
            cv2.polylines(frame, [pts], True, border_color, 3)

            # Zone label with active status
            status = "ACTIVE" if is_active else "NOT ACTIVE"
            label = f"{zone_name}: {status}"

            # Calculate label position at top of polygon
            label_pos = polygon[0]

            # Add background to label for better visibility
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame,
                         (label_pos[0] - 5, label_pos[1] - label_h - 15),
                         (label_pos[0] + label_w + 5, label_pos[1] - 5),
                         border_color, -1)
            cv2.putText(frame, label, (label_pos[0], label_pos[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw line zones using Supervision with enhanced IN/OUT labels
        for idx, (line_name, line_annotator) in enumerate(line_annotators.items()):
            # Draw line with counts using Supervision
            frame = line_annotator.annotate(
                frame=frame,
                line_counter=sv_line_zones[line_name]
            )

            # Get line details for custom IN/OUT labels
            line = lines[idx]
            start_x = int(line['start']['x'] * width)
            start_y = int(line['start']['y'] * height)
            end_x = int(line['end']['x'] * width)
            end_y = int(line['end']['y'] * height)

            # Calculate midpoint and perpendicular direction
            mid_x = (start_x + end_x) // 2
            mid_y = (start_y + end_y) // 2

            # Get current counts from Supervision LineZone
            in_count = sv_line_zones[line_name].in_count
            out_count = sv_line_zones[line_name].out_count

            # Draw IN label above the line
            in_label = f"IN: {in_count}"
            (in_w, in_h), _ = cv2.getTextSize(in_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            # Background for IN label
            cv2.rectangle(frame, (mid_x - 20, mid_y - 60), (mid_x + in_w + 20, mid_y - 25), (0, 0, 0), -1)
            cv2.rectangle(frame, (mid_x - 20, mid_y - 60), (mid_x + in_w + 20, mid_y - 25), (0, 255, 0), 2)
            cv2.putText(frame, in_label, (mid_x, mid_y - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Draw OUT label below the line
            out_label = f"OUT: {out_count}"
            (out_w, out_h), _ = cv2.getTextSize(out_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            # Background for OUT label
            cv2.rectangle(frame, (mid_x - 20, mid_y + 25), (mid_x + out_w + 20, mid_y + 60), (0, 0, 0), -1)
            cv2.rectangle(frame, (mid_x - 20, mid_y + 25), (mid_x + out_w + 20, mid_y + 60), (0, 0, 255), 2)
            cv2.putText(frame, out_label, (mid_x, mid_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Draw arrow indicators for direction
            # Arrow pointing up for IN
            arrow_up = np.array([[mid_x - 60, mid_y - 40], [mid_x - 70, mid_y - 50], [mid_x - 50, mid_y - 50]], np.int32)
            cv2.fillPoly(frame, [arrow_up], (0, 255, 0))

            # Arrow pointing down for OUT
            arrow_down = np.array([[mid_x - 60, mid_y + 40], [mid_x - 70, mid_y + 30], [mid_x - 50, mid_y + 30]], np.int32)
            cv2.fillPoly(frame, [arrow_down], (0, 0, 255))

        # Draw summary panel at top-right corner
        if lines:  # Only if we have counting lines
            panel_height = 100
            panel_width = 250
            panel_x = width - panel_width - 10
            panel_y = 10

            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (width - 10, panel_y + panel_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            # Border
            cv2.rectangle(frame, (panel_x, panel_y), (width - 10, panel_y + panel_height), (255, 255, 255), 2)

            # Title
            cv2.putText(frame, "TOTAL COUNTS", (panel_x + 10, panel_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Calculate total IN and OUT
            total_in = sum(sv_line_zones[line_name].in_count for line_name in sv_line_zones)
            total_out = sum(sv_line_zones[line_name].out_count for line_name in sv_line_zones)

            # Display totals
            cv2.putText(frame, f"IN:  {total_in}", (panel_x + 20, panel_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"OUT: {total_out}", (panel_x + 20, panel_y + 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Write frame
        out.write(frame)

    cap.release()
    out.release()

    # Get line counts from Supervision LineZones
    final_line_counts = {}
    for line_name, sv_line_zone in sv_line_zones.items():
        final_line_counts[line_name] = {
            'in': int(sv_line_zone.in_count),
            'out': int(sv_line_zone.out_count)
        }

    return {
        'zones': {name: list(ids) for name, ids in zone_ids.items()},
        'lines': final_line_counts,
        'total_frames': total_frames
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get video info
        cap = cv2.VideoCapture(filepath)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get first frame for preview
        ret, frame = cap.read()
        if ret:
            preview_path = os.path.join('static', 'preview.jpg')
            cv2.imwrite(preview_path, frame)
        cap.release()

        return jsonify({
            'success': True,
            'filename': filename,
            'width': width,
            'height': height,
            'fps': fps,
            'frames': frame_count,
            'preview': '/static/preview.jpg'
        })

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    filename = data.get('filename')
    zones = data.get('zones', [])
    lines = data.get('lines', [])

    if not filename:
        return jsonify({'error': 'No filename provided'}), 400

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_filename = 'processed_' + filename
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    try:
        # Process video
        stats = process_video(input_path, zones, lines, output_path)

        return jsonify({
            'success': True,
            'output': output_filename,
            'stats': stats
        })
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download(filename):
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
