"""
=============================================================
  Tello EDU 드론 - 3단계: 객체 추적 및 1m 거리 유지 비행
=============================================================
목적: 탐지된 사람(또는 지정 객체)을 1미터 거리를 유지하며 따라갑니다.

원리:
  1. YOLO로 대상 객체 탐지 → 화면상 바운딩박스 추출
  2. PID 제어기를 이용해 3가지 축 오차를 계산:
     - 좌우(Yaw)  : 객체 중심 X ↔ 화면 중심 X의 차이
     - 상하(UD)   : 객체 중심 Y ↔ 화면 중심 Y의 차이
     - 전후(FB)   : 바운딩박스 면적 ↔ 목표 면적(1m 거리)의 차이
  3. 계산된 속도 명령을 드론에 전송 (send_rc_control)

거리 캘리브레이션:
  TARGET_AREA 값은 "사람이 드론과 1m 거리일 때의 바운딩박스 픽셀 면적"
  기본값은 960×720 해상도, 성인 기준으로 설정되어 있습니다.
  환경에 따라 TARGET_AREA를 조정하세요.

사전 준비:
  pip install djitellopy opencv-python ultralytics

단축키:
  SPACE : 이륙/착륙 토글
  q     : 추적 종료 + 착륙
  t     : 수동 이륙
  l     : 수동 착륙
  r     : 추적 ON/OFF 토글 (비행 중 일시 정지)
=============================================================
"""

import cv2
import time
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════
#  설정 상수
# ══════════════════════════════════════════════════════════════

MODEL_PATH   = "yolov8n.pt"
TARGET_CLASS = 0          # 0 = person (COCO 클래스 번호)
CONF_THRESH  = 0.55       # YOLO 신뢰도 임계값

# 1미터 거리에서의 목표 바운딩박스 면적 (픽셀²)
# → 실제 환경에서 드론을 1m 앞에 두고 면적 값을 측정하여 업데이트하세요
TARGET_AREA  = 80_000

# PID 게인 값 (P: 비례, I: 적분, D: 미분)
# 값이 너무 크면 진동, 너무 작으면 반응이 느립니다
PID_YAW  = {"Kp": 0.25,  "Ki": 0.0,  "Kd": 0.1 }   # 좌우 회전
PID_UD   = {"Kp": 0.25,  "Ki": 0.0,  "Kd": 0.1 }   # 상하 이동
PID_FB   = {"Kp": 0.0003,"Ki": 0.0,  "Kd": 0.0001}  # 전후 이동

# 드론 속도 제한 (cm/s, Tello 최대 100)
MAX_SPEED   = 30
DEAD_ZONE   = 15   # 이 픽셀 이내의 오차는 무시 (진동 방지)


# ══════════════════════════════════════════════════════════════
#  PID 제어기
# ══════════════════════════════════════════════════════════════

class PIDController:
    """단순 PID 제어기"""

    def __init__(self, Kp, Ki, Kd, setpoint=0.0, output_limits=(-100, 100)):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.output_limits = output_limits

        self._prev_error = 0.0
        self._integral   = 0.0
        self._prev_time  = time.time()

    def compute(self, measured_value):
        now = time.time()
        dt  = now - self._prev_time
        if dt <= 0:
            dt = 1e-6

        error          = self.setpoint - measured_value
        self._integral += error * dt
        derivative     = (error - self._prev_error) / dt

        output = (self.Kp * error +
                  self.Ki * self._integral +
                  self.Kd * derivative)

        # 출력 범위 제한
        lo, hi = self.output_limits
        output = max(lo, min(hi, output))

        self._prev_error = error
        self._prev_time  = now
        return output, error

    def reset(self):
        self._prev_error = 0.0
        self._integral   = 0.0
        self._prev_time  = time.time()


# ══════════════════════════════════════════════════════════════
#  객체 추적 드론 클래스
# ══════════════════════════════════════════════════════════════

class TelloTracker:
    def __init__(self):
        # ── 모델 ──────────────────────────────────────────────
        print("[INFO] YOLO 모델 로딩 중...")
        self.model = YOLO(MODEL_PATH)
        self.class_names = self.model.names

        # ── 드론 ──────────────────────────────────────────────
        self.tello = Tello()
        self.tello.connect()
        self.battery = self.tello.get_battery()
        print(f"[INFO] 배터리: {self.battery}%")

        self.tello.streamon()
        self.frame_read = self.tello.get_frame_read()
        time.sleep(1)

        # ── PID 제어기 ─────────────────────────────────────────
        # 화면 해상도를 모르므로 일단 0으로 초기화, 첫 프레임에서 설정
        self.pid_yaw = PIDController(**PID_YAW, output_limits=(-MAX_SPEED, MAX_SPEED))
        self.pid_ud  = PIDController(**PID_UD,  output_limits=(-MAX_SPEED, MAX_SPEED))
        self.pid_fb  = PIDController(**PID_FB,  output_limits=(-MAX_SPEED, MAX_SPEED))

        # 화면 중심 (첫 프레임에서 설정)
        self.frame_cx = None
        self.frame_cy = None

        # ── 상태 플래그 ────────────────────────────────────────
        self.is_flying   = False
        self.is_tracking = False   # True: 드론이 실제로 이동 명령 전송

        # 무대상 타임아웃 (N초 이상 대상을 잃으면 호버링)
        self.LOST_TIMEOUT = 2.0
        self._last_seen   = None

    # ── 이륙 / 착륙 ────────────────────────────────────────────
    def takeoff(self):
        if not self.is_flying:
            print("[ACTION] 이륙!")
            self.tello.takeoff()
            self.tallo.move_up(50)
            self.is_flying   = True
            self.is_tracking = True
            self.pid_yaw.reset()
            self.pid_ud.reset()
            self.pid_fb.reset()
            self._last_seen = None

    def land(self):
        if self.is_flying:
            print("[ACTION] 착륙!")
            self.tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.2)
            self.tello.land()
            self.is_flying   = False
            self.is_tracking = False

    # ── 대상 객체 선택 (가장 큰 바운딩박스) ─────────────────────
    def _pick_target(self, results):
        best = None
        best_area = 0
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id != TARGET_CLASS:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best = {
                        "bbox":   (x1, y1, x2, y2),
                        "center": ((x1+x2)//2, (y1+y2)//2),
                        "area":   area,
                        "conf":   float(box.conf[0])
                    }
        return best

    # ── 드론 속도 명령 계산 (PID) ──────────────────────────────
    def _compute_control(self, target):
        cx, cy = target["center"]
        area   = target["area"]

        # 화면 중심과의 오차 (픽셀)
        err_x = cx - self.frame_cx    # + = 오른쪽
        err_y = cy - self.frame_cy    # + = 아래쪽
        err_a = area - TARGET_AREA    # + = 너무 가까움

        # Dead zone 적용 (미세 진동 방지)
        if abs(err_x) < DEAD_ZONE: err_x = 0
        if abs(err_y) < DEAD_ZONE: err_y = 0

        # PID 계산
        yaw_speed, _ = self.pid_yaw.compute(-err_x)    # 좌우 회전 (부호 반전)
        ud_speed,  _ = self.pid_ud.compute(-err_y)     # 상하 이동
        fb_speed,  _ = self.pid_fb.compute(-err_a)     # 전후 이동

        return int(yaw_speed), int(ud_speed), int(fb_speed)

    # ── 메인 루프 ──────────────────────────────────────────────
    def run(self):
        print("[INFO] 추적 루프 시작")
        print("  SPACE : 이륙/착륙   |  r : 추적 ON/OFF   |  q : 종료")

        prev_time = time.time()

        while True:
            frame = self.frame_read.frame
            if frame is None:
                continue

            h, w = frame.shape[:2]

            # 화면 중심 초기화
            if self.frame_cx is None:
                self.frame_cx, self.frame_cy = w // 2, h // 2

            # ── YOLO 탐지 ──────────────────────────────────────
            results = self.model(frame, conf=CONF_THRESH,
                                 classes=[TARGET_CLASS], verbose=False)
            target  = self._pick_target(results)

            # FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-9)
            prev_time = curr_time

            # ── 드론 제어 ──────────────────────────────────────
            if self.is_flying and self.is_tracking:
                if target:
                    self._last_seen = time.time()
                    yaw, ud, fb = self._compute_control(target)
                    # lr=0 (좌우 이동 X, 회전으로만 방향 조정)
                    self.tello.send_rc_control(0, fb, ud, yaw)
                else:
                    # 대상 소실 처리
                    elapsed = (time.time() - self._last_seen
                               if self._last_seen else self.LOST_TIMEOUT)
                    if elapsed >= self.LOST_TIMEOUT:
                        # 호버링
                        self.tello.send_rc_control(0, 0, 0, 0)
            elif self.is_flying:
                # 추적 일시정지 → 호버링
                self.tello.send_rc_control(0, 0, 0, 0)

            # ── 화면 렌더링 ────────────────────────────────────
            display = self._render(frame, target, fps)
            cv2.imshow("Tello EDU - Stage 3: Tracking", display)

            # ── 키 입력 처리 ───────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.land()
                break
            elif key == ord(' '):           # SPACE: 이륙/착륙 토글
                if self.is_flying:
                    self.land()
                else:
                    self.takeoff()
            elif key == ord('t'):           # t: 이륙
                self.takeoff()
            elif key == ord('l'):           # l: 착륙
                self.land()
            elif key == ord('r'):           # r: 추적 토글
                if self.is_flying:
                    self.is_tracking = not self.is_tracking
                    state = "ON" if self.is_tracking else "OFF (호버링)"
                    print(f"[INFO] 추적 상태: {state}")

        # ── 정리 ───────────────────────────────────────────────
        if self.is_flying:
            self.land()
        self.tello.streamoff()
        cv2.destroyAllWindows()
        print("[INFO] 프로그램 종료")

    # ── 화면 렌더링 함수 ───────────────────────────────────────
    def _render(self, frame, target, fps):
        h, w = frame.shape[:2]
        display = frame.copy()

        # 화면 중앙 십자선
        cv2.line(display, (self.frame_cx - 30, self.frame_cy),
                 (self.frame_cx + 30, self.frame_cy), (200, 200, 200), 1)
        cv2.line(display, (self.frame_cx, self.frame_cy - 30),
                 (self.frame_cx, self.frame_cy + 30), (200, 200, 200), 1)

        if target:
            x1, y1, x2, y2 = target["bbox"]
            cx, cy          = target["center"]
            area            = target["area"]

            # 추적 대상 바운딩박스 (녹색)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 중심 → 화면 중앙 오차 선 (빨간 선)
            cv2.line(display, (self.frame_cx, self.frame_cy),
                     (cx, cy), (0, 0, 255), 2)
            cv2.circle(display, (cx, cy), 5, (0, 255, 255), -1)

            # 현재 면적 vs 목표 면적 비율
            ratio = area / TARGET_AREA
            dist_txt = (f"거리: {'가까움' if ratio > 1.1 else '멀다' if ratio < 0.9 else '적정'}"
                        f"  ({ratio:.2f}x)")
            cv2.putText(display, dist_txt, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ── HUD ────────────────────────────────────────────────
        bat_color = (0, 255, 0) if self.battery > 30 else (0, 100, 255)
        cv2.putText(display, f"Battery: {self.battery}%",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bat_color, 2)
        cv2.putText(display, f"FPS: {fps:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 비행 상태
        fly_state = "FLYING" if self.is_flying else "LANDED"
        fly_color = (0, 255, 0) if self.is_flying else (0, 100, 255)
        cv2.putText(display, fly_state,
                    (w - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fly_color, 2)

        # 추적 상태
        trk_state = "TRACKING" if self.is_tracking else "HOVERING"
        trk_color = (0, 255, 255) if self.is_tracking else (0, 165, 255)
        cv2.putText(display, trk_state,
                    (w - 160, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, trk_color, 2)

        # 대상 없음 경고
        if not target:
            cv2.putText(display, "TARGET LOST",
                        (w // 2 - 90, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # 안내 텍스트
        cv2.putText(display, "SPACE:Takeoff/Land  R:Track  Q:Quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        return display


# ══════════════════════════════════════════════════════════════
#  진입점
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    tracker = TelloTracker()
    tracker.run()
