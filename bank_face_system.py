"""
================================================================================
  ПРОТОТИП СИСТЕМЫ РАСПОЗНАВАНИЯ ЛИЦ ДЛЯ БАНКА
================================================================================

  Pipeline:
    камера/фото → детекция лица (Haar cascade) → извлечение признаков →
    → сравнение с базой → вердикт СВОЙ / ЧУЖОЙ

  Используется только OpenCV — без dlib, без face_recognition.

  Как это устроено:
  1. Детекция лица: Haar cascade (встроен в OpenCV, не требует скачивания)
  2. Выравнивание: аффинное по положению глаз
  3. Признаки: комбинация LBP-гистограмм + HOG лица
  4. Сравнение: cosine distance между векторами признаков

  Для продакшена заменяется на:
    - InsightFace / ArcFace — точные нейросетевые эмбеддинги
    - YuNet (OpenCV DNN) + SFace — более точная детекция и распознавание
    - Микросервис на FastAPI/gRPC + векторная БД (Milvus)
"""

from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# =============================================================================
#  МОДЕЛИ ДАННЫХ
# =============================================================================

@dataclass
class Customer:
    """Клиент банка с зарегистрированным биометрическим профилем."""
    customer_id: str
    full_name: str
    features: np.ndarray              # вектор признаков лица (полное лицо)
    features_upper: np.ndarray | None = None  # вектор верхней части лица (для маски)
    photo_path: str | None = None
    registered_at: float = field(default_factory=time.time)

    def features_as_list(self) -> list[float]:
        return self.features.tolist()

    def features_upper_as_list(self) -> list[float] | None:
        return self.features_upper.tolist() if self.features_upper is not None else None

    def __repr__(self) -> str:
        return f"Customer({self.customer_id}, {self.full_name})"


@dataclass
class MatchResult:
    """Результат поиска лица в базе клиентов."""
    found: bool
    customer: Customer | None = None
    confidence: float = 0.0
    threshold: float = 0.5

    @property
    def verdict(self) -> str:
        if not self.found:
            return "НЕ ОПОЗНАН"
        return f"КЛИЕНТ: {self.customer.full_name} ({self.confidence:.0%})"

    def __repr__(self) -> str:
        return f"MatchResult(verdict='{self.verdict}')"


# =============================================================================
#  ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ (Feature Extractor)
# =============================================================================

class FeatureExtractor:
    """
    Извлекает вектор признаков из области лица.
    Комбинирует несколько методов для надёжности:
    - LBP (Local Binary Patterns) — текстура лица
    - HOG (Histogram of Oriented Gradients) — форма лица
    - Цветовая гистограмма — общий тон кожи
    """

    FACE_SIZE = (128, 128)
    LBP_GRID = (8, 8)

    @classmethod
    def extract(cls, face_img: np.ndarray) -> np.ndarray:
        """
        Извлекает признаковый вектор из изображения лица.
        Вход: изображение лица (должно быть уже выровнено и обрезано).
        Выход: numpy-вектор признаков.
        """
        resized = cv2.resize(face_img, cls.FACE_SIZE)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
        if resized.ndim == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        features = []

        # 1) LBP-текстура
        lbp = cls._lbp_histogram(gray)
        features.append(lbp)

        # 2) HOG-форма
        hog = cls._hog(gray)
        features.append(hog)

        # 3) Цветовая гистограмма (HSV)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        for ch in [0, 1]:
            hist = cv2.calcHist([hsv], [ch], None, [32], [0, 256])
            cv2.normalize(hist, hist)
            features.append(hist.flatten())

        combined = np.concatenate(features).astype(np.float32)
        return combined / (np.linalg.norm(combined) + 1e-8)

    @classmethod
    def extract_upper(cls, face_img: np.ndarray) -> np.ndarray:
        """
        Извлекает признаки только из верхней 60% лица (глаза, лоб, брови).
        Используется когда нижняя часть лица скрыта маской.
        """
        h = face_img.shape[0]
        upper = face_img[0:int(h * 0.6), :]
        return cls.extract(upper)

    @staticmethod
    def _lbp_histogram(gray: np.ndarray, grid: tuple = (8, 8)) -> np.ndarray:
        """LBP-гистограмма по сетке ячеек."""
        h, w = gray.shape
        cell_h, cell_w = h // grid[0], w // grid[1]
        histograms = []
        for i in range(grid[0]):
            for j in range(grid[1]):
                cell = gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                lbp = FeatureExtractor._lbp_cell(cell)
                hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
                histograms.append(hist.astype(np.float32))
        result = np.concatenate(histograms)
        result /= (np.sum(result) + 1e-8)
        return result

    @staticmethod
    def _lbp_cell(cell: np.ndarray) -> np.ndarray:
        """Вычисляет LBP для одной ячейки."""
        h, w = cell.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = cell[i, j]
                code = 0
                code |= (cell[i-1, j-1] >= center) << 7
                code |= (cell[i-1, j]   >= center) << 6
                code |= (cell[i-1, j+1] >= center) << 5
                code |= (cell[i,   j+1] >= center) << 4
                code |= (cell[i+1, j+1] >= center) << 3
                code |= (cell[i+1, j]   >= center) << 2
                code |= (cell[i+1, j-1] >= center) << 1
                code |= (cell[i,   j-1] >= center) << 0
                lbp[i-1, j-1] = code
        return lbp

    @staticmethod
    def _hog(gray: np.ndarray, cells: tuple = (8, 8), bins: int = 9) -> np.ndarray:
        """Упрощённый HOG-дескриптор."""
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        h, w = gray.shape
        cell_h, cell_w = h // cells[0], w // cells[1]
        hog_feat = []
        for i in range(cells[0]):
            for j in range(cells[1]):
                cell_mag = mag[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                cell_ang = ang[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                hist, _ = np.histogram(cell_ang, bins=bins, range=(0, 180), weights=cell_mag)
                hist = hist.astype(np.float32)
                hist /= (np.sum(hist) + 1e-8)
                hog_feat.append(hist)
        return np.concatenate(hog_feat)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Косинусное сходство двух векторов (0..1)."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# =============================================================================
#  ДЕТЕКТОР ЛИЦ
# =============================================================================

class FaceDetector:
    """
    Мульти-каскадный детектор лиц (Haar cascades, OpenCV).
    Использует каскады: фронтальный, alt2, профиль, глаза — с fallback-стратегией
    для частично скрытых лиц (маска, шапка, очки).

    Для продакшена: заменить на YuNet (OpenCV DNN) или RetinaFace/MTCNN.
    """

    def __init__(self):
        base = os.path.dirname(__file__)

        def _load(name):
            path = os.path.join(base, name)
            if os.path.exists(path):
                return cv2.CascadeClassifier(path)
            return None

        self._frontal = _load("haarcascade_frontalface_default.xml")
        self._alt2 = _load("haarcascade_frontalface_alt2.xml")
        self._profile = _load("haarcascade_profileface.xml")
        self._eye = _load("haarcascade_eye.xml")

        if self._frontal is None and self._alt2 is None:
            self._frontal = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def _preprocess(self, gray: np.ndarray) -> np.ndarray:
        """CLAHE-эквализация для улучшения контраста при плохом освещении."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Находит лица на изображении. Многоэтапная стратегия:
        1. Фронтальный каскад (строгий)
        2. Alt2 каскад (средний)
        3. Фронтальный каскад с ослабленными параметрами (для маски/шапки)
        4. Глаза → оценка области лица (для маски)
        5. Профильный каскад
        Возвращает список (x, y, w, h) для каждого лица.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        eq = self._preprocess(gray)
        h, w = gray.shape
        results = []

        def _detect(cascade, img, sf, mn, ms):
            if cascade is None:
                return []
            faces = cascade.detectMultiScale(img, scaleFactor=sf, minNeighbors=mn, minSize=ms)
            return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

        # Этап 1: строгий фронтальный (полное лицо)
        faces = _detect(self._frontal, eq, 1.1, 5, (60, 60))
        if faces:
            return faces

        # Этап 2: alt2 каскад (более устойчив к вариациям)
        faces = _detect(self._alt2, eq, 1.1, 4, (50, 50))
        if faces:
            return faces

        # Этап 3: ослабленный фронтальный (частично скрытое лицо — маска, шапка)
        for sf in (1.05, 1.03):
            for mn in (4, 3):
                faces = _detect(self._frontal, eq, sf, mn, (50, 50))
                if faces:
                    return faces
                faces = _detect(self._alt2, eq, sf, mn, (40, 40))
                if faces:
                    return faces

        # Этап 4: детекция глаз → оценка области лица (для маски)
        eyes = _detect(self._eye, eq, 1.05, 3, (25, 15))
        if len(eyes) >= 1:
            eyes_sorted = sorted(eyes, key=lambda e: e[0])
            if len(eyes_sorted) >= 2:
                # Два глаза: строим лицо вокруг них
                ex1, ey1, ew1, eh1 = eyes_sorted[0]
                ex2, ey2, ew2, eh2 = eyes_sorted[-1]
                eye_center_x = (ex1 + ew1 // 2 + ex2 + ew2 // 2) // 2
                eye_center_y = (ey1 + eh1 // 2 + ey2 + eh2 // 2) // 2
                eye_dist = abs(ex2 - ex1)
                face_w = int(eye_dist * 2.5)
                face_h = int(face_w * 1.3)
                face_x = max(0, eye_center_x - face_w // 2)
                face_y = max(0, eye_center_y - int(face_w * 0.6))
                face_w = min(face_w, w - face_x)
                face_h = min(face_h, h - face_y)
                if face_w > 50 and face_h > 50:
                    return [(face_x, face_y, face_w, face_h)]
            else:
                # Один глаз: грубая оценка
                ex, ey, ew, eh = eyes_sorted[0]
                face_w = int(ew * 6)
                face_h = int(face_w * 1.3)
                face_x = max(0, ex + ew // 2 - face_w // 2)
                face_y = max(0, ey + eh // 2 - int(face_w * 0.5))
                face_w = min(face_w, w - face_x)
                face_h = min(face_h, h - face_y)
                if face_w > 50 and face_h > 50:
                    return [(face_x, face_y, face_w, face_h)]

        # Этап 5: профильный каскад
        faces = _detect(self._profile, eq, 1.1, 4, (50, 50))
        if faces:
            return faces

        # Этап 6: последний шанс — очень мягкие параметры
        faces = _detect(self._frontal, eq, 1.01, 2, (40, 40))
        if faces:
            return faces

        return []

    def detect_largest(self, image: np.ndarray) -> tuple | None:
        """Находит самое крупное лицо на фото."""
        faces = self.detect(image)
        if not faces:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    @staticmethod
    def crop_face(image: np.ndarray, rect: tuple, padding: float = 0.2) -> np.ndarray:
        """Вырезает область лица с отступом."""
        x, y, w, h = rect
        h_img, w_img = image.shape[:2]
        px = int(w * padding)
        py = int(h * padding)
        x1 = max(0, x - px)
        y1 = max(0, y - py)
        x2 = min(w_img, x + w + px)
        y2 = min(h_img, y + h + py)
        return image[y1:y2, x1:x2]


# =============================================================================
#  ЯДРО СИСТЕМЫ — FaceBank
# =============================================================================

class FaceBank:
    """
    Центральный класс системы распознавания лиц для банка.
    Регистрирует клиентов, ищет по лицу (1:N matching).
    """

    DEFAULT_THRESHOLD = 0.5
    DB_PATH = "facebank_db.pkl"

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self._customers: dict[str, Customer] = {}
        self.threshold = threshold
        self._detector = FaceDetector()

    # ----- Регистрация -----

    def enroll(self, photo_path: str, customer_id: str, full_name: str) -> Customer | None:
        """Регистрирует клиента в базе по фото."""
        image = cv2.imread(str(photo_path))
        if image is None:
            print(f"  [!] Не удалось загрузить фото: {photo_path}")
            return None

        face_rect = self._detector.detect_largest(image)
        if face_rect is None:
            print(f"  [!] Лицо не найдено: {photo_path}")
            return None

        face_img = FaceDetector.crop_face(image, face_rect)
        features = FeatureExtractor.extract(face_img)
        features_upper = FeatureExtractor.extract_upper(face_img)

        customer = Customer(
            customer_id=customer_id,
            full_name=full_name,
            features=features,
            features_upper=features_upper,
            photo_path=photo_path,
        )
        self._customers[customer_id] = customer
        print(f"  [+] Клиент зарегистрирован: {full_name} ({customer_id})")
        return customer

    # ----- Поиск (1:N) -----

    def identify(self, frame: np.ndarray) -> MatchResult:
        """Поиск лица по всей базе клиентов. Сравнивает и полное лицо, и верхнюю часть (для маски)."""
        face_rect = self._detector.detect_largest(frame)
        if face_rect is None:
            return MatchResult(found=False)

        face_img = FaceDetector.crop_face(frame, face_rect)
        features = FeatureExtractor.extract(face_img)
        features_upper = FeatureExtractor.extract_upper(face_img)

        best_sim = -1.0
        best_customer = None

        for customer in self._customers.values():
            sim_full = FeatureExtractor.cosine_similarity(customer.features, features)

            sim_up = 0.0
            if customer.features_upper is not None:
                sim_up = FeatureExtractor.cosine_similarity(customer.features_upper, features_upper)

            sim = max(sim_full, sim_up)
            if sim > best_sim:
                best_sim = sim
                best_customer = customer

        found = best_sim >= self.threshold and best_customer is not None
        return MatchResult(
            found=found,
            customer=best_customer if found else None,
            confidence=float(best_sim),
            threshold=self.threshold,
        )

    # ----- Верификация (1:1) -----

    def verify(self, frame: np.ndarray, customer_id: str) -> MatchResult:
        """Проверка: принадлежит ли лицо указанному клиенту?"""
        customer = self._customers.get(customer_id)
        if customer is None:
            return MatchResult(found=False)

        face_rect = self._detector.detect_largest(frame)
        if face_rect is None:
            return MatchResult(found=False)

        face_img = FaceDetector.crop_face(frame, face_rect)
        features = FeatureExtractor.extract(face_img)
        sim = FeatureExtractor.cosine_similarity(customer.features, features)
        found = sim >= self.threshold

        return MatchResult(
            found=found,
            customer=customer if found else None,
            confidence=float(sim),
            threshold=self.threshold,
        )

    # ----- Персистентность -----

    def save(self, path: str = DB_PATH):
        data = {
            "customers": {
                cid: {
                    "customer_id": c.customer_id,
                    "full_name": c.full_name,
                    "features": c.features_as_list(),
                    "features_upper": c.features_upper_as_list(),
                    "photo_path": c.photo_path,
                    "registered_at": c.registered_at,
                }
                for cid, c in self._customers.items()
            },
            "threshold": self.threshold,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"  [+] База сохранена: {len(self._customers)} клиентов → {path}")

    def load(self, path: str = DB_PATH):
        if not os.path.exists(path):
            print(f"  [!] Файл не найден: {path}")
            return

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.threshold = data.get("threshold", self.DEFAULT_THRESHOLD)
        self._customers = {}
        for cid, cdata in data["customers"].items():
            features_upper = None
            if cdata.get("features_upper"):
                features_upper = np.array(cdata["features_upper"])
            self._customers[cid] = Customer(
                customer_id=cdata["customer_id"],
                full_name=cdata["full_name"],
                features=np.array(cdata["features"]),
                features_upper=features_upper,
                photo_path=cdata.get("photo_path"),
                registered_at=cdata.get("registered_at", 0),
            )
        print(f"  [+] База загружена: {len(self._customers)} клиентов ← {path}")

    # ----- Утилиты -----

    @property
    def customer_count(self) -> int:
        return len(self._customers)

    def list_customers(self) -> list[str]:
        return [f"  {c.customer_id}: {c.full_name}" for c in self._customers.values()]


# =============================================================================
#  LIVENESS DETECTION
# =============================================================================

class LivenessDetector:
    """Детектор живого лица: анализ движения между кадрами."""

    def __init__(self, motion_threshold: float = 1.5):
        self.motion_threshold = motion_threshold
        self._prev_gray = None
        self._pixel_count = 0

    def check(self, frame: np.ndarray) -> tuple[bool, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            self._pixel_count = gray.shape[0] * gray.shape[1]
            return False, 0.0

        delta = cv2.absdiff(self._prev_gray, gray)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        motion_pct = np.sum(thresh > 0) / self._pixel_count * 100
        self._prev_gray = gray
        return motion_pct > self.motion_threshold, motion_pct


# =============================================================================
#  ВИЗУАЛИЗАЦИЯ
# =============================================================================

def draw_result(frame: np.ndarray, match: MatchResult, detector: FaceDetector) -> np.ndarray:
    """Рисует рамку и статус распознавания."""
    face_rect = detector.detect_largest(frame)
    if face_rect is None:
        return frame

    x, y, w, h = face_rect

    if match.found:
        color = (0, 255, 0)
        label = f"КЛИЕНТ: {match.customer.full_name}"
    else:
        color = (0, 0, 255)
        label = "НЕ ОПОЗНАН"

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.rectangle(frame, (x, y - 35), (x + w, y), color, cv2.FILLED)
    cv2.putText(frame, label, (x + 6, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"match: {match.confidence:.0%}", (x + 6, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame
