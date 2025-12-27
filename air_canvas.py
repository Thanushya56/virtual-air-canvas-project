import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
from color_palette import show_color_picker

# ====== MediaPipe Hand Setup ======
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# ====== Global Drawing Settings ======
CANVAS_W, CANVAS_H = 1080, 720

current_color = (0, 0, 255)  # Red in BGR
current_brush_size = 4

eraser_active = False
last_paint_color = current_color

# Stroke types: 'free', 'rect', 'circle', 'text'
strokes = []  # list of dicts representing strokes
current_stroke_points = []

mode = "free"  # 'free', 'rect', 'circle'
is_drawing = False
shape_start = None
shape_end = None

cooldown_time = 1.0
last_button_time = time.time()

prev_x, prev_y = None, None
smoothing = 0.3

last_cursor_pos = (CANVAS_W // 2, CANVAS_H // 2)  # default for text placement

cv2.namedWindow("Air Canvas", cv2.WINDOW_AUTOSIZE)

def new_blank_canvas():
    return np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8) * 255

def draw_buttons(img):
    # CLEAR
    cv2.rectangle(img, (20, 10), (140, 60), (50, 50, 50), -1)
    cv2.putText(img, "CLEAR", (40, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)

    # COLOR
    cv2.rectangle(img, (160, 10), (260, 60), current_color, -1)
    text_color = (255, 255, 255) if sum(current_color) < 382 else (0, 0, 0)
    cv2.putText(img, "COLOR", (175, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                text_color, 2)

    # ERASER
    eraser_bg = (0, 0, 0) if eraser_active else (150, 150, 150)
    cv2.rectangle(img, (280, 10), (400, 60), eraser_bg, -1)
    cv2.putText(img, "ERASE", (295, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)

    # UNDO
    cv2.rectangle(img, (420, 10), (520, 60), (80, 80, 80), -1)
    cv2.putText(img, "UNDO", (435, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)

    # SAVE
    cv2.rectangle(img, (540, 10), (660, 60), (100, 100, 100), -1)
    cv2.putText(img, "SAVE", (560, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)

    # Brush size + mode text
    cv2.putText(img, f"Size: {current_brush_size}", (690, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

    mode_text = f"Mode: {mode.upper()}"
    if eraser_active:
        mode_text += " (ERASE)"
    cv2.putText(img, mode_text, (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)

    # Help hint
    cv2.putText(img, "F: Free  R: Rect  C: Circle  T: Text  U: Undo",
                (20, CANVAS_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (60, 60, 60), 2)

def save_canvas(canvas):
    filename = f"air_canvas_{int(time.time())}.png"
    cv2.imwrite(filename, canvas)
    print(f"Canvas saved as {filename}")

def draw_stroke_on(canvas, frame, stroke):
    t = stroke["type"]
    color = stroke["color"]
    thickness = stroke["thickness"]

    if t == "free":
        pts = stroke["points"]
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], color, thickness)
            cv2.line(frame, pts[i - 1], pts[i], color, thickness)

    elif t == "rect":
        start = stroke["start"]
        end = stroke["end"]
        cv2.rectangle(canvas, start, end, color, thickness)
        cv2.rectangle(frame, start, end, color, thickness)

    elif t == "circle":
        center = stroke["center"]
        radius = stroke["radius"]
        cv2.circle(canvas, center, radius, color, thickness)
        cv2.circle(frame, center, radius, color, thickness)

    elif t == "text":
        pos = stroke["pos"]
        text = stroke["text"]
        cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, color, 2)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, color, 2)

def main():
    global current_color, current_brush_size, eraser_active, last_paint_color
    global prev_x, prev_y, last_button_time
    global mode, is_drawing, shape_start, shape_end
    global current_stroke_points, last_cursor_pos

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CANVAS_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CANVAS_H)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        canvas = new_blank_canvas()
        draw_buttons(frame)

        drawing_enabled = False

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks,
                                       mp_hands.HAND_CONNECTIONS)

                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                x, y = int(index_tip.x * w), int(index_tip.y * h)
                last_cursor_pos = (x, y)

                # Smoothing
                if prev_x is None or prev_y is None:
                    smoothed_x, smoothed_y = x, y
                else:
                    smoothed_x = int(prev_x + smoothing * (x - prev_x))
                    smoothed_y = int(prev_y + smoothing * (y - prev_y))
                prev_x, prev_y = smoothed_x, smoothed_y
                smoothed = (smoothed_x, smoothed_y)

                cv2.circle(frame, smoothed, 8, (0, 0, 0), -1)

                # Pinch distance (thumb - index) decides drawing on/off
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                pinch_distance = np.hypot(thumb_x - x, thumb_y - y)
                drawing_enabled = pinch_distance >= 40

                # Brush size = distance between index & middle fingers (when not erasing)
                if not eraser_active:
                    mid_x, mid_y = int(middle_tip.x * w), int(middle_tip.y * h)
                    finger_gap = np.hypot(mid_x - x, mid_y - y)
                    min_d, max_d = 10, 150
                    ratio = np.clip((finger_gap - min_d) / (max_d - min_d), 0, 1)
                    current_brush_size = int(2 + ratio * (40 - 2))

                # ----- Handle top buttons -----
                if smoothed_y < 65:
                    now = time.time()
                    if now - last_button_time > cooldown_time:
                        # CLEAR
                        if 20 <= smoothed_x <= 140:
                            strokes.clear()

                        # COLOR
                        elif 160 <= smoothed_x <= 260:
                            new_color = show_color_picker(current_color)
                            if new_color:
                                eraser_active = False
                                current_color = new_color
                                last_paint_color = current_color

                        # ERASE
                        elif 280 <= smoothed_x <= 400:
                            eraser_active = not eraser_active
                            if eraser_active:
                                last_paint_color = current_color
                                current_color = (255, 255, 255)
                                current_brush_size = max(current_brush_size, 25)
                            else:
                                current_color = last_paint_color

                        # UNDO
                        elif 420 <= smoothed_x <= 520:
                            if strokes:
                                strokes.pop()

                        # SAVE
                        elif 540 <= smoothed_x <= 660:
                            # redraw all strokes onto a fresh canvas and save
                            save_canvas(render_full_canvas())

                        last_button_time = now

                # ----- Drawing / Shape Logic -----
                # We DON'T modify strokes directly here; only when drawing ends
                if drawing_enabled and smoothed_y >= 65:
                    if not is_drawing:
                        # Drawing started
                        is_drawing = True
                        if mode == "free":
                            current_stroke_points = [smoothed]
                        elif mode == "rect":
                            shape_start = smoothed
                            shape_end = smoothed
                        elif mode == "circle":
                            shape_start = smoothed
                            shape_end = smoothed
                    else:
                        # Continue drawing
                        if mode == "free":
                            current_stroke_points.append(smoothed)
                        elif mode in ("rect", "circle"):
                            shape_end = smoothed
                else:
                    if is_drawing:
                        # Drawing ended -> finalize stroke
                        if mode == "free" and len(current_stroke_points) > 1:
                            strokes.append({
                                "type": "free",
                                "color": current_color,
                                "thickness": current_brush_size,
                                "points": current_stroke_points.copy()
                            })
                        elif mode == "rect" and shape_start and shape_end:
                            strokes.append({
                                "type": "rect",
                                "color": current_color,
                                "thickness": current_brush_size,
                                "start": shape_start,
                                "end": shape_end
                            })
                        elif mode == "circle" and shape_start and shape_end:
                            center = shape_start
                            radius = int(np.hypot(shape_end[0] - center[0],
                                                  shape_end[1] - center[1]))
                            strokes.append({
                                "type": "circle",
                                "color": current_color,
                                "thickness": current_brush_size,
                                "center": center,
                                "radius": radius
                            })

                        current_stroke_points = []
                        shape_start, shape_end = None, None
                        is_drawing = False

        else:
            prev_x, prev_y = None, None
            is_drawing = False
            current_stroke_points = []

        # ----- Draw all existing strokes -----
        for stroke in strokes:
            draw_stroke_on(canvas, frame, stroke)

        # ----- Draw current stroke preview -----
        if mode == "free" and len(current_stroke_points) > 1:
            temp_stroke = {
                "type": "free",
                "color": current_color,
                "thickness": current_brush_size,
                "points": current_stroke_points
            }
            draw_stroke_on(canvas, frame, temp_stroke)
        elif mode == "rect" and shape_start and shape_end:
            temp_stroke = {
                "type": "rect",
                "color": current_color,
                "thickness": current_brush_size,
                "start": shape_start,
                "end": shape_end
            }
            draw_stroke_on(canvas, frame, temp_stroke)
        elif mode == "circle" and shape_start and shape_end:
            center = shape_start
            radius = int(np.hypot(shape_end[0] - center[0],
                                  shape_end[1] - center[1]))
            temp_stroke = {
                "type": "circle",
                "color": current_color,
                "thickness": current_brush_size,
                "center": center,
                "radius": radius
            }
            draw_stroke_on(canvas, frame, temp_stroke)

        # Show windows
        cv2.imshow("Air Canvas", canvas)
        cv2.imshow("Tracking", frame)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('u'):
            if strokes:
                strokes.pop()
        elif key == ord('f'):
            mode = "free"
        elif key == ord('r'):
            mode = "rect"
        elif key == ord('c'):
            mode = "circle"
        elif key == ord('t'):
            # Add text at last cursor position
            print("Enter text to place on canvas: ")
            user_text = input().strip()
            if user_text:
                strokes.append({
                    "type": "text",
                    "color": current_color,
                    "thickness": 2,
                    "pos": last_cursor_pos,
                    "text": user_text
                })

    cap.release()
    cv2.destroyAllWindows()

def render_full_canvas():
    canvas = new_blank_canvas()
    dummy_frame = canvas.copy()
    for stroke in strokes:
        draw_stroke_on(canvas, dummy_frame, stroke)
    return canvas

if __name__ == "__main__":
    main()
