"""
=============================================================
  Tello EDU 드론 - 1단계: 실시간 영상 스트리밍
=============================================================
목적: Tello EDU 드론에서 실시간 영상을 받아 화면에 표시합니다.

사전 준비:
  pip install djitellopy opencv-python

사용법:
  1. Tello EDU 드론을 켠다
  2. PC의 Wi-Fi를 Tello 네트워크(TELLO-XXXXXX)에 연결
  3. 이 스크립트 실행: python stage1_video.py
  4. 종료: 영상 창에서 'q' 키 누르기
=============================================================
"""

import cv2
from djitellopy import Tello
import time


def main():
    # ── 1. 드론 연결 ───────────────────────────────────────────
    tello = Tello()
    tello.connect()

    # 배터리 잔량 확인 (20% 이하 경고)
    battery = tello.get_battery()
    print(f"[INFO] 배터리 잔량: {battery}%")
    if battery < 20:
        print("[경고] 배터리가 부족합니다. 충전 후 사용하세요.")

    # ── 2. 영상 스트리밍 시작 ──────────────────────────────────
    tello.streamon()
    frame_read = tello.get_frame_read()

    # 스트림 초기화 대기 (첫 프레임이 안정화될 때까지)
    time.sleep(1)

    print("[INFO] 실시간 영상 스트리밍 시작 | 종료: 'q' 키")

    # ── 3. 영상 출력 루프 ──────────────────────────────────────
    while True:
        # 현재 프레임 가져오기
        frame = frame_read.frame

        if frame is None:
            print("[경고] 프레임을 받지 못했습니다. 재시도 중...")
            time.sleep(0.05)
            continue

        # 프레임 크기 (기본: 960×720)
        h, w = frame.shape[:2]

        # ── 화면에 정보 오버레이 ──────────────────────────────
        # 배터리 정보 표시
        battery = tello.get_battery()
        cv2.putText(frame, f"Battery: {battery}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 해상도 정보 표시
        cv2.putText(frame, f"Resolution: {w}x{h}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 화면 중앙 십자선 (중심점 참조용)
        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 1)

        # ── 영상 출력 ─────────────────────────────────────────
        cv2.imshow("Tello EDU - Stage 1: Video Stream", frame)

        # 'q' 키를 누르면 종료
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] 사용자 종료 요청")
            break

    # ── 4. 정리 ───────────────────────────────────────────────
    tello.streamoff()
    cv2.destroyAllWindows()
    print("[INFO] 스트리밍 종료")


if __name__ == "__main__":
    main()
