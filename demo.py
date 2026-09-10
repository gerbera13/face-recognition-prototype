"""
================================================================================
  ДЕМОНСТРАЦИЯ — БАНК. РАСПОЗНАВАНИЕ ЛИЦ (ЖИВОЕ ДЕМО)
================================================================================

  Запуск:
    python demo.py

  Управление в окне камеры:
    E — зарегистрировать лицо как клиента
    L — показать список клиентов
    S — сохранить базу
    Q — выход

  Сценарий:
    1. Запустите
    2. Нажмите E, введите ID и ФИО → лицо сохранено
    3. Отвернитесь / покажите другое лицо → «НЕ ОПОЗНАН»
    4. Вернитесь → система вас узнает
"""

import os
import sys
import time
from datetime import datetime

import cv2

from bank_face_system import FaceBank, FaceDetector, LivenessDetector, MatchResult, draw_result

DB_PATH = "facebank_db.pkl"


def main():
    print("=" * 60)
    print("  БАНК — система распознавания лиц (демо)")
    print("=" * 60)

    fb = FaceBank()
    detector = FaceDetector()

    if os.path.exists(DB_PATH):
        fb.load(DB_PATH)
    else:
        print("  [!] База пуста. Нажмите E чтобы зарегистрировать лицо.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[!] Камера не найдена. Проверьте подключение веб-камеры.")
        sys.exit(1)

    liveness = LivenessDetector()
    last_check = 0
    current_match = MatchResult(found=False)

    print("\n  Управление: [E]nroll  [L]ist  [S]ave  [Q]uit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        if now - last_check > 0.5:
            current_match = fb.identify(frame)
            last_check = now

        frame = draw_result(frame, current_match, detector)

        # HUD
        cv2.putText(frame, f"DB: {fb.customer_count} clients", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {datetime.now():%H:%M:%S}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        is_live, motion = liveness.check(frame)
        lc = (0, 255, 0) if is_live else (0, 0, 255)
        cv2.putText(frame, f"Liveness: {motion:.1f}%", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, lc, 1)

        cv2.imshow("FaceBank Demo", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break

        elif key == ord('e'):
            cid = input("  ID клиента (напр. 001): ").strip()
            fio = input("  ФИО клиента: ").strip()
            if cid and fio:
                photo = f"enroll_{cid}_{int(time.time())}.jpg"
                cv2.imwrite(photo, frame)
                fb.enroll(photo, cid, fio)
                print()

        elif key == ord('l'):
            print(f"\n  Клиентов в базе: {fb.customer_count}")
            for line in fb.list_customers():
                print(line)
            print()

        elif key == ord('s'):
            fb.save(DB_PATH)

    cap.release()
    cv2.destroyAllWindows()
    fb.save(DB_PATH)
    print("\nЗавершено. База сохранена.")


if __name__ == "__main__":
    main()
