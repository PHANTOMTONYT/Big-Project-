import cv2
import numpy as np
import os
from ultralytics import YOLO


class PersonCounter:
    def __init__(self, video_path, model_name='yolo11n.pt', output_path=None):
        """
        Initialize the Person Counter with YOLO v11/v12
        video_path: path to video file
        model_name: YOLO model to use (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)
                   n=nano (fastest), s=small, m=medium, l=large, x=extra large (most accurate)
        output_path: path to save processed video (optional)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        self.count_in = 0
        self.count_out = 0

        # Get video dimensions
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps == 0:
            self.fps = 30

        # Video writer for saving output
        self.output_path = output_path
        self.video_writer = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
            if self.video_writer.isOpened():
                print(f"✓ Output video will be saved to: {self.output_path}")
            else:
                print(f"⚠ Warning: Could not initialize video writer for {self.output_path}")
                self.video_writer = None

        # Line drawing variables
        self.line_points = []
        self.line_defined = False
        self.drawing_mode = True
        self.line_offset = 15

        # Track state for each ID
        self.track_history = {}  # Store previous centroids
        self.track_counted = {}  # Track if already counted
        self.track_initial_side = {}  # Which side of line track started on

        # Initialize YOLO model with tracking
        print(f"Loading YOLO model: {model_name}")
        print("This may take a moment on first run (downloading model)...")
        self.model = YOLO(model_name)

        # Set model parameters for better accuracy
        self.conf_threshold = 0.4  # Confidence threshold
        self.iou_threshold = 0.5   # IoU threshold for NMS

        print(f"✓ Model loaded successfully!")

    def which_side_of_line(self, point, line_start, line_end):
        """Determine which side of the line a point is on"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        return ((x2 - x1) * (y0 - y1) - (y2 - y1) * (x0 - x1))

    def point_to_line_distance(self, point, line_start, line_end):
        """Calculate perpendicular distance from point to line"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end

        num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

        if den == 0:
            return 0
        return num / den

    def check_line_crossing(self, track_id, new_centroid):
        """Check if track crossed the counting line"""
        if len(self.line_points) != 2:
            return

        # Check if already counted
        if self.track_counted.get(track_id, False):
            return

        # Get previous centroid
        if track_id not in self.track_history:
            # First time seeing this track, store initial side
            side = self.which_side_of_line(new_centroid, self.line_points[0], self.line_points[1])
            self.track_initial_side[track_id] = side
            self.track_history[track_id] = new_centroid
            return

        old_centroid = self.track_history[track_id]
        line_start = self.line_points[0]
        line_end = self.line_points[1]

        # Calculate which side of the line each centroid is on
        old_side = self.which_side_of_line(old_centroid, line_start, line_end)
        new_side = self.which_side_of_line(new_centroid, line_start, line_end)

        # Update history
        self.track_history[track_id] = new_centroid

        # Check if crossed the line (sign changed)
        if old_side * new_side < 0:
            # Verify the crossing is near the line (not far away)
            dist = self.point_to_line_distance(new_centroid, line_start, line_end)

            if dist < self.line_offset * 3:  # Within reasonable distance
                # Determine direction of crossing
                if old_side > 0 and new_side < 0:
                    self.count_in += 1
                    self.track_counted[track_id] = True
                    print(f"✓ Person IN  [ID: {track_id}] - Total IN: {self.count_in}")
                elif old_side < 0 and new_side > 0:
                    self.count_out += 1
                    self.track_counted[track_id] = True
                    print(f"✓ Person OUT [ID: {track_id}] - Total OUT: {self.count_out}")

    def draw_info(self, frame, results):
        """Draw counting line, bounding boxes, and counter info on frame"""

        # Draw the counting line
        if len(self.line_points) == 2:
            # Draw main line
            cv2.line(frame, self.line_points[0], self.line_points[1], (0, 255, 255), 3)

            # Draw direction indicators
            mid_x = (self.line_points[0][0] + self.line_points[1][0]) // 2
            mid_y = (self.line_points[0][1] + self.line_points[1][1]) // 2

            # Add semi-transparent background for labels
            cv2.rectangle(frame, (mid_x + 10, mid_y - 35), (mid_x + 80, mid_y - 5), (0, 0, 0), -1)
            cv2.rectangle(frame, (mid_x + 10, mid_y + 5), (mid_x + 100, mid_y + 40), (0, 0, 0), -1)

            cv2.putText(frame, "IN", (mid_x + 20, mid_y - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "OUT", (mid_x + 20, mid_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        elif len(self.line_points) == 1 and self.drawing_mode:
            cv2.circle(frame, self.line_points[0], 8, (0, 255, 255), -1)
            cv2.putText(frame, "Click second point", self.line_points[0],
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Process YOLO results
        if results and len(results) > 0:
            result = results[0]

            # Get boxes with tracking IDs
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes

                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    # Get confidence
                    conf = float(box.conf[0])

                    # Get class
                    cls = int(box.cls[0])

                    # Only process persons (class 0 in COCO)
                    if cls != 0:
                        continue

                    # Get tracking ID if available
                    track_id = None
                    if hasattr(box, 'id') and box.id is not None:
                        track_id = int(box.id[0])

                    # Calculate centroid
                    centroid_x = (x1 + x2) // 2
                    centroid_y = (y1 + y2) // 2
                    centroid = (centroid_x, centroid_y)

                    # Check line crossing if we have tracking and line is defined
                    if track_id is not None and self.line_defined:
                        self.check_line_crossing(track_id, centroid)

                    # Choose color based on confidence
                    if conf > 0.7:
                        color = (0, 255, 0)  # Green for high confidence
                    elif conf > 0.5:
                        color = (0, 255, 255)  # Yellow for medium
                    else:
                        color = (0, 165, 255)  # Orange for low

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label with ID and confidence
                    if track_id is not None:
                        label = f"ID:{track_id} {conf:.2f}"
                    else:
                        label = f"Person {conf:.2f}"

                    # Add background to label
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                    # Draw centroid
                    cv2.circle(frame, centroid, 5, (255, 0, 255), -1)

                    # Draw tracking trail if available
                    if track_id is not None and track_id in self.track_history:
                        old_centroid = self.track_history.get(track_id)
                        if old_centroid:
                            cv2.line(frame, old_centroid, centroid, (255, 0, 255), 2)

        # Draw counter panel at top left
        panel_height = 140
        panel_width = 320
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Draw counts with icons
        cv2.putText(frame, f"IN:  {self.count_in}", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        cv2.putText(frame, f"OUT: {self.count_out}", (15, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)

        total_inside = self.count_in - self.count_out
        cv2.putText(frame, f"Inside: {total_inside}", (15, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Draw active tracks count
        active_tracks = len(self.track_history)
        cv2.putText(frame, f"Tracking: {active_tracks}", (self.width - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Draw instructions if line not defined
        if not self.line_defined:
            instruction = "Click TWO points to draw counting line"
            (text_w, text_h), _ = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (5, self.height - text_h - 30),
                         (text_w + 15, self.height - 10), (0, 0, 0), -1)
            cv2.putText(frame, instruction, (10, self.height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        return frame

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for drawing the line"""
        if event == cv2.EVENT_LBUTTONDOWN and self.drawing_mode:
            if len(self.line_points) < 2:
                self.line_points.append((x, y))
                print(f"✓ Point {len(self.line_points)} set at: ({x}, {y})")

                if len(self.line_points) == 2:
                    self.line_defined = True
                    self.drawing_mode = False
                    print("✓ Counting line defined! Tracking started.")
                    print("=" * 70)
                    print("Controls: SPACE=Pause | R=Redraw | S=Reset | Q=Quit")
                    print("=" * 70)

    def run(self):
        """Main loop to run the person counter"""
        print("=" * 70)
        print("PERSON COUNTER - YOLO v11 with Advanced Tracking")
        print("=" * 70)
        print(f"Video: {os.path.basename(self.video_path)}")
        print(f"Resolution: {self.width}x{self.height} @ {self.fps} FPS")
        print(f"Confidence Threshold: {self.conf_threshold}")
        print("\nFEATURES:")
        print("  ✓ YOLO v11 person detection")
        print("  ✓ Advanced object tracking with persistent IDs")
        print("  ✓ Custom line drawing for any angle")
        print("  ✓ Real-time IN/OUT counting")
        print("\nINSTRUCTIONS:")
        print("  1. Click TWO points to draw counting line")
        print("  2. Watch accurate person tracking and counting")
        print("\nCONTROLS:")
        print("  SPACE - Pause/Resume")
        print("  R     - Redraw counting line")
        print("  S     - Reset counters")
        print("  Q     - Quit")
        print("=" * 70)

        cv2.namedWindow('YOLO Person Counter')
        cv2.setMouseCallback('YOLO Person Counter', self.mouse_callback)

        paused = False
        frame = None
        results = None

        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("\n" + "=" * 70)
                    print(f"VIDEO ENDED")
                    print(f"Final Count: IN={self.count_in} | OUT={self.count_out} | Inside={self.count_in - self.count_out}")
                    print("=" * 70)

                    response = input("\nReplay video? (y/n): ")
                    if response.lower() == 'y':
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        # Reset tracking history but keep counts
                        self.track_history = {}
                        self.track_counted = {}
                        self.track_initial_side = {}
                        continue
                    else:
                        break

                # Run YOLO tracking
                # persist=True keeps track IDs consistent across frames
                results = self.model.track(
                    frame,
                    persist=True,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    classes=[0],  # Only detect persons (class 0)
                    verbose=False,
                    tracker="bytetrack.yaml"  # Use ByteTrack tracker
                )

                display_frame = self.draw_info(frame.copy(), results)

                # Write frame to output video if enabled
                if self.video_writer is not None:
                    self.video_writer.write(display_frame)
            else:
                if frame is not None and results is not None:
                    display_frame = self.draw_info(frame.copy(), results)

            cv2.imshow('YOLO Person Counter', display_frame)

            wait_time = max(1, int(1000 / self.fps)) if not paused else 0
            key = cv2.waitKey(wait_time) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print("⏸ PAUSED" if paused else "▶ RESUMED")
            elif key == ord('r'):
                self.line_points = []
                self.line_defined = False
                self.drawing_mode = True
                print("🖊 Redraw the counting line")
            elif key == ord('s'):
                self.count_in = 0
                self.count_out = 0
                self.track_counted = {}
                print("🔄 Counters reset!")

        self.cleanup()

    def cleanup(self):
        """Release resources"""
        self.cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"\n✓ Output video saved to: {self.output_path}")
        cv2.destroyAllWindows()
        total_inside = self.count_in - self.count_out
        print(f"\n✓ Session Complete")
        print(f"  IN: {self.count_in} | OUT: {self.count_out} | Inside: {total_inside}")


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 70)
    print("YOLO Person Counter - Ultralytics Edition")
    print("=" * 70)

    if len(sys.argv) < 2:
        print("\nUsage: python person_counter_yolo.py <path_to_video> [model] [output_path]")
        print("\nExamples:")
        print("  python person_counter_yolo.py video.mp4")
        print("  python person_counter_yolo.py video.mp4 yolo11s.pt output1.mp4")
        print("  python person_counter_yolo.py \"C:/Videos/people.mp4\" yolo11n.pt \"C:/Videos/output.mp4\"")
        print("\nAvailable models (speed vs accuracy):")
        print("  yolo11n.pt - Nano (fastest, good for real-time)")
        print("  yolo11s.pt - Small (balanced)")
        print("  yolo11m.pt - Medium (more accurate)")
        print("  yolo11l.pt - Large (high accuracy)")
        print("  yolo11x.pt - Extra Large (highest accuracy, slowest)")
        print("\nDefault: yolo11n.pt")
        print("=" * 70)
        sys.exit(1)

    video_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else 'yolo11n.pt'
    output_path = sys.argv[3] if len(sys.argv) > 3 else 'output1.mp4'

    try:
        counter = PersonCounter(video_path, model_name=model_name, output_path=output_path)
        counter.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
