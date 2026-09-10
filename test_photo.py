"""Тестовый запуск без камеры — на фото."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import cv2
from bank_face_system import FaceBank, FeatureExtractor, FaceDetector

detector = FaceDetector()

# Загружаем тестовое фото (или создаём заглушку)
test_img = "test_face.jpg"
if not os.path.exists(test_img):
    print(f"[!] Положите любое фото лица в {test_img} и перезапустите.")
    print("    Например: скопируйте своё селфи и назовите test_face.jpg")
    sys.exit(1)

img = cv2.imread(test_img)
rect = detector.detect_largest(img)

if rect is None:
    print("[!] Лицо на фото не найдено. Попробуйте другое фото.")
    sys.exit(1)

print(f"[+] Лицо найдено! Координаты: {rect}")
x, y, w, h = rect
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imshow("Test - face detected", img)
print("Окно закроется через 5 секунд...")
cv2.waitKey(5000)
cv2.destroyAllWindows()
print("Готово! Система работает.")
