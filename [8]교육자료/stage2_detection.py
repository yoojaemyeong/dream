"""
=============================================================
  Tello EDU 드론 - 2단계: 실시간 객체 탐지 (Object Detection)
=============================================================
목적: 드론 영상에서 YOLOv8로 사물을 실시간 탐지하고 객체화합니다.

사전 준비:
  pip install djitellopy opencv-python ultralytics

YOLOv8 모델 크기 선택 (속도 vs 정확도):
  yolov8n.pt → 가장 빠름 (드론 실시간 처리에 권장)
  yolov8s.pt → 빠름 + 준수한 정확도
  yolov8m.pt → 중간
  yolov8l.pt → 느리지만 높은 정확도

사용법:
  python stage2_detection.py
  종료: 'q' 키 / 탐지 대상 변경: 'd' 키 (사람 only 토글)
=============================================================
"""

import cv2
from djitellopy import Tello
from ultralytics import YOLO
import time
import numpy as np


# ── 설정값 ────────────────────────────────────────────────────
MODEL_PATH    = "yolov8n.pt"     # 모델 파일 (없으면 자동 다운로드)
CONF_THRESHOLD = 0.5             # 신뢰도 임계값 (0.0 ~ 1.0)
TARGET_CLASS   = None            # None = 전체 클래스, 0 = 사람(person)만


class ObjectDetector:
    """YOLOv8 기반 실시간 객체 탐지 클래스"""

    def __init__(self, model_path=MODEL_PATH, conf=CONF_THRESHOLD):
        print(f"[INFO] YOLO 모델 로딩: {model_path}")
        self.model = YOLO(model_path)
        self.conf  = conf
        self.filter_person_only = False   # True이면 사람만 탐지, 기본 False

        # COCO 클래스 이름 (80개 클래스)
        self.class_names = self.model.names
        print(f"[INFO] 모델 로딩 완료 | 탐지 가능 클래스 수: {len(self.class_names)}")

    def detect(self, frame):
        """
        프레임에서 객체를 탐지합니다.

        Returns:
            annotated_frame : 바운딩박스가 그려진 프레임
            detections      : 탐지된 객체 목록
                              [{"label": str, "conf": float,
                                "bbox": (x1,y1,x2,y2), "center": (cx,cy)}]
        """
        classes_filter = [0] if self.filter_person_only else None

        results = self.model(
            frame,
            conf=self.conf,
            classes=classes_filter,
            verbose=False          # 콘솔 출력 억제
        )

        detections = []
        annotated_frame = frame.copy()

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label  = self.class_names[cls_id]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                area   = (x2 - x1) * (y2 - y1)

                detections.append({
                    "label":  label,
                    "conf":   conf,
                    "bbox":   (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "area":   area
                })

                # ── 바운딩박스 그리기 ──────────────────────────
                color = (0, 255, 0) if label == "person" else (255, 165, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # 레이블 배경 (가독성)
                text  = f"{label} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame,
                              (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                cv2.putText(annotated_frame, text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # 중심점 표시
                cv2.circle(annotated_frame, (cx, cy), 4, (0, 0, 255), -1)

        return annotated_frame, detections


def draw_hud(frame, detections, battery, filter_person):
    """화면 상단 HUD(정보 패널, Head Up Display) 그리기"""
    h, w = frame.shape[:2]

    # 배터리
    cv2.putText(frame, f"Battery: {battery}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 탐지 객체 수
    cv2.putText(frame, f"Objects: {len(detections)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 필터 모드
    mode_text = "Person Only" if filter_person else "All Objects"
    cv2.putText(frame, f"Mode: {mode_text}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

    # 탐지된 객체 목록 (우측)
    for i, det in enumerate(detections[:6]):   # 최대 6개 표시
        txt = f"{det['label']}: {det['conf']:.2f}"
        cv2.putText(frame, txt, (w - 250, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

    # 화면 중앙 십자선
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (200, 200, 200), 1)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (200, 200, 200), 1)

    # 단축키 안내 (하단)
    cv2.putText(frame, "Q: Quit  |  D: Toggle Person-Only",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)


def main():
    # ── 1. 모델 초기화 ─────────────────────────────────────────
    detector = ObjectDetector()

    # ── 2. 드론 연결 ───────────────────────────────────────────
    tello = Tello()
    tello.connect()
    battery = tello.get_battery()
    print(f"[INFO] 배터리: {battery}%")

    tello.streamon()
    frame_read = tello.get_frame_read()
    time.sleep(1)

    print("[INFO] 객체 탐지 시작 | 'q': 종료 | 'd': Person-Only 토글")

    # FPS 측정용
    prev_time = time.time()
    fps = 0

    # ── 3. 메인 루프 ───────────────────────────────────────────
    while True:
        frame = frame_read.frame
        if frame is None:
            continue

        # FPS 계산
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time

        # 객체 탐지
        annotated, detections = detector.detect(frame)

        # HUD 오버레이
        battery = tello.get_battery()
        draw_hud(annotated, detections, battery, detector.filter_person_only)

        # FPS 표시
        cv2.putText(annotated, f"FPS: {fps:.1f}",
                    (annotated.shape[1] - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Tello EDU - Stage 2: Object Detection", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            # 사람 전용 탐지 토글
            detector.filter_person_only = not detector.filter_person_only
            mode = "Person Only" if detector.filter_person_only else "All Objects"
            print(f"[INFO] 탐지 모드 변경: {mode}")

    # ── 4. 정리 ───────────────────────────────────────────────
    tello.streamoff()
    cv2.destroyAllWindows()
    print("[INFO] 객체 탐지 종료")


if __name__ == "__main__":
    main()
