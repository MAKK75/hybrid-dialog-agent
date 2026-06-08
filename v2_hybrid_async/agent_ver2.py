import os
import time
import threading
import queue
import json
import psutil
import cv2
import numpy as np
import speech_recognition as sr
import pygame
import requests
import chromadb
import subprocess
import sys
from dotenv import load_dotenv
load_dotenv()

if not os.path.exists('face_detection_yunet_2023mar.onnx') or not os.path.exists('face_recognition_sface_2021dec.onnx'):
    print("ОШИБКА: Файлы моделей YuNet и SFace (.onnx) не найдены в папке!")
    sys.exit(1)

class Config:
    YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID") #ключики
    YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")   
    
    MODEL_ID = "yandexgpt-lite" 
    API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    
    DB_PATH = "./agent_memory_db"
    
    DETECTOR_MODEL = 'face_detection_yunet_2023mar.onnx'
    RECOGNIZER_MODEL = 'face_recognition_sface_2021dec.onnx'
    
    FACE_MATCH_THRESHOLD = 0.65 
    
    REGISTRATION_TIME = 5.0  #Время на сбор в базу  
    DISK_CRITICAL_PERCENT = 10  #Это для теста дискового пространства 
    OWNER_RETURN_TIME = 10 #Для сценария возвращения хозяина  
    SYSTEM_CHECK_INTERVAL = 300 #Для проверки диска, можно уменьшить, чтобы посмотреть (повысил, чтобы агент лишний раз не перебивал) 

def translit(text):
    mapping = {'а':'a', 'б':'b', 'в':'v', 'г':'g', 'д':'d', 'е':'e', 'ё':'e', 'ж':'zh', 
               'з':'z', 'и':'i', 'й':'y', 'к':'k', 'л':'l', 'м':'m', 'н':'n', 'о':'o', 
               'п':'p', 'р':'r', 'с':'s', 'т':'t', 'у':'u', 'ф':'f', 'х':'kh', 'ц':'ts', 
               'ч':'ch', 'ш':'sh', 'щ':'shch', 'ъ':'', 'ы':'y', 'ь':'', 'э':'e', 'ю':'yu', 'я':'ya'}
    res = "".join(mapping.get(char, char) for char in text.lower())
    return res.capitalize()

def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)

class FaceTracker:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.x, self.y, self.w, self.h = bbox
        self.first_seen = time.time()
        self.last_seen = time.time()
        
        self.name = "Незнакомец"
        self.relation = "Unknown"
        self.is_verified = False
        self.prompted = False  
        self.feature = None    

class MemorySystem:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=Config.DB_PATH)
        self.people = self.client.get_or_create_collection(
            name="people", metadata={"hnsw:space": "cosine"}
        )

    def add_person(self, name, face_embedding, relation="Гость"):
        doc_id = f"p_{int(time.time())}"
        flat_embedding = face_embedding.flatten().tolist()
        self.people.add(
            embeddings=[flat_embedding],
            documents=[json.dumps({"name": name, "relation": relation})],
            ids=[doc_id]
        )

    def find_person(self, face_embedding):
        if self.people.count() == 0: return None
        flat_embedding = face_embedding.flatten().tolist()
        results = self.people.query(query_embeddings=[flat_embedding], n_results=1)
        
        if results['distances'] and len(results['distances'][0]) > 0:
            if results['distances'][0][0] <= Config.FACE_MATCH_THRESHOLD:
                return json.loads(results['documents'][0][0])
        return None

class Orchestrator:
    def __init__(self):
        pygame.mixer.init()
        self.memory = MemorySystem()
        self.system_mode = "registering" if self.memory.people.count() == 0 else "working"
        
        self.latest_frame = None
        self.active_tracks = {}
        self.next_track_id = 1
        
        self.last_seen_owner_time = time.time()
        self.owner_present = True 
        
        self.pending_stranger_emb = None
        self.stranger_prompt_time = 0 
        
        self.is_speaking = False
        self.running = True
        self.task_queue = queue.Queue()

    def speak(self, text):
        self.is_speaking = True
        print(f"\n[Агент]: {text}\n")
        try:
            subprocess.run(["edge-tts", "--voice", "ru-RU-SvetlanaNeural", "--rate=+15%", "--text", text, "--write-media", "voice.mp3"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            pygame.mixer.music.load("voice.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() and self.running: 
                time.sleep(0.05)
            pygame.mixer.music.unload()
            if os.path.exists("voice.mp3"): os.remove("voice.mp3")
        except Exception as e: 
            print(f"Ошибка аудио: {e}")
        finally:
            self.is_speaking = False

    def handle_local_intent(self, text):
        t = text.lower()
        try:
            active_names =[track.name for track in list(self.active_tracks.values()) if track.is_verified]
        except RuntimeError:
            active_names =[]
            
        if "кого" in t and "видишь" in t:
            if not active_names: return "Я сейчас никого не вижу."
            return f"Я вижу: {', '.join(active_names)}."
        
        if "как тебя зовут" in t:
            return "Я ваша локальная охранная система."
            
        if "кто перед тобой" in t or "кто я" in t:
            if not active_names: return "Передо мной никого нет."
            return f"Передо мной находится: {', '.join(active_names)}."
        return None

    def ask_llm(self, text, context_data=""):
        headers = {
            "Authorization": f"Api-Key {Config.YANDEX_API_KEY}",
            "x-folder-id": Config.YANDEX_FOLDER_ID,
            "Content-Type": "application/json"
        }
        disk_free = 100 - psutil.disk_usage('/').percent
        system_prompt = (
            "Ты ИИ-ассистент, система безопасности. Отвечай кратко, "
            "живым языком, максимум 1-2 предложения. Избегай списков.\n"
            f"КОНТЕКСТ: В камере сейчас: {context_data}. Свободно на диске: {disk_free:.1f}%."
        )


        data = {
            "modelUri": f"gpt://{Config.YANDEX_FOLDER_ID}/{Config.MODEL_ID}/latest",
            "completionOptions": {"stream": False, "temperature": 0.5, "maxTokens": "150"},
            "messages":[
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": text}
            ]
        }
        
        try:
            response = requests.post(Config.API_URL, headers=headers, json=data, timeout=5)
            response.raise_for_status()
            return response.json()['result']['alternatives'][0]['message']['text'].strip()
        except Exception:
            return "Связь с сервером прервана."

    def camera_thread(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                if w > 640:
                    scale = 640 / w
                    frame = cv2.resize(frame, (640, int(h * scale)))
                self.latest_frame = frame
            time.sleep(0.01) 
        cap.release()

    def vision_thread(self):
        detector = cv2.FaceDetectorYN.create(Config.DETECTOR_MODEL, "", (640, 480), 0.8, 0.3, 5000)
        recognizer = cv2.FaceRecognizerSF.create(Config.RECOGNIZER_MODEL, "")

        while self.running:
            # Здесь регулировал, чтобы хватало ресурсов на одновременную работу камеры и мозга
            if self.latest_frame is None:
                time.sleep(0.05)
                continue
            
            frame = self.latest_frame.copy()
            
            # Подгонял размер динамически
            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            

            if self.pending_stranger_emb is not None:
                if time.time() - self.stranger_prompt_time > 25: 
                    print("\n[DEBUG] Таймаут ожидания имени. Сброс состояния.")
                    self.pending_stranger_emb = None
                    for t in self.active_tracks.values(): t.prompted = False

            current_time = time.time()
            _, faces = detector.detect(frame)
            owner_in_frame = False

            if faces is not None:
                for face in faces:
                    coords = face[:-1].astype(np.int32)
                    bbox = [coords[0], coords[1], coords[2], coords[3]]
                    
                    matched_tid = None
                    try:
                        tracks_items = list(self.active_tracks.items())
                    except RuntimeError:
                        tracks_items =[]

                    for tid, track in tracks_items:
                        if get_iou(bbox,[track.x, track.y, track.w, track.h]) > 0.4:
                            matched_tid = tid
                            break
                    
                    if matched_tid is None:
                        matched_tid = self.next_track_id
                        self.next_track_id += 1
                        self.active_tracks[matched_tid] = FaceTracker(matched_tid, bbox)
                    
                    track = self.active_tracks[matched_tid]
                    track.x, track.y, track.w, track.h = bbox
                    track.last_seen = current_time

                    try:
                        aligned_face = recognizer.alignCrop(frame, face)
                        feature = recognizer.feature(aligned_face)
                        track.feature = feature
                    except Exception:
                        continue

                    if track.name == "Незнакомец":
                        person = self.memory.find_person(feature)
                        if person:
                            track.name = person['name']
                            track.relation = person['relation']
                            track.is_verified = True
                    
                    if track.name != "Незнакомец":
                        if track.relation == "Создатель":
                            owner_in_frame = True
                    else:
                        time_tracked = current_time - track.first_seen
                        
                        if self.system_mode == "registering":
                            if time_tracked >= Config.REGISTRATION_TIME:
                                self.memory.add_person("Создатель", feature, "Создатель")
                                self.system_mode = "working"
                                self.last_seen_owner_time = time.time()
                                self.task_queue.put({"type": "registration_done"})
                                
                                track.name = "Создатель"
                                track.relation = "Создатель"
                                track.is_verified = True
                        
                        elif self.system_mode == "working":
                            if time_tracked > 2.5 and not track.prompted and self.pending_stranger_emb is None:
                                self.pending_stranger_emb = feature
                                self.stranger_prompt_time = time.time()
                                self.task_queue.put({"type": "stranger_detected"})
                                track.prompted = True

            # Чистим треки (на всякий)
            self.active_tracks = {tid: t for tid, t in self.active_tracks.items() if current_time - t.last_seen < 1.0}
            
            away_time = current_time - self.last_seen_owner_time
            if self.owner_present and away_time > Config.OWNER_RETURN_TIME:
                print(f"\n[DEBUG] Хозяин покинул кадр.")
                self.owner_present = False

            if owner_in_frame:
                self.last_seen_owner_time = current_time
                if not self.owner_present:
                    print("\n[DEBUG] Хозяин вернулся!")
                    self.owner_present = True
                    self.task_queue.put({"type": "owner_returned"})

            time.sleep(0.02) # Добавили плавности

    def audio_thread(self):
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True 
        
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print("\n[Аудио] Микрофон откалиброван и готов.")
            was_speaking = True
            
            while self.running:
                if self.is_speaking or self.system_mode == "registering":
                    time.sleep(0.3)
                    was_speaking = True
                    continue
                
                if was_speaking:
                    time.sleep(0.3) 
                    print("\n[Аудио] Микрофон активен (вы можете говорить)...")
                    was_speaking = False

                try:
                    audio = recognizer.listen(source, timeout=4, phrase_time_limit=7)
                    query = recognizer.recognize_google(audio, language="ru-RU").lower()
                    
                    print(f"[Вы]: {query}")
                    
                    if any(w in query for w in["выход", "стоп", "отключись"]):
                        with self.task_queue.mutex:
                            self.task_queue.queue.clear()
                        self.task_queue.put({"type": "exit"})
                        break
                        
                    self.task_queue.put({"type": "voice_query", "text": query})
                except (sr.WaitTimeoutError, sr.UnknownValueError):
                    pass 

    def monitor_thread(self):
        last_warning_time = time.time() 
        while self.running:
            time.sleep(10) 
            usage = psutil.disk_usage('/').percent
            if usage >= Config.DISK_CRITICAL_PERCENT:
                if time.time() - last_warning_time > Config.SYSTEM_CHECK_INTERVAL:
                    self.task_queue.put({"type": "disk_warning", "usage": usage})
                    last_warning_time = time.time()

    def brain_thread(self):
        while self.running:
            try:
                task = self.task_queue.get(timeout=1)
                
                if task["type"] == "exit":
                    self.speak("Отключаю системы. До свидания!")
                    self.running = False
                    break
                    
                elif task["type"] == "registration_done":
                    self.speak("Отлично! Лицо сохранено. Перехожу в рабочий режим.")
                    
                elif task["type"] == "owner_returned":
                    ans = self.ask_llm("Я не видел создателя больше минуты, и вот он вернулся в кадр. Поздоровайся с ним кратко.")
                    self.speak(ans)
                    
                elif task["type"] == "stranger_detected":
                    self.speak("Я вижу новое лицо. Хозяин, подскажите, как зовут этого человека?")
                    
                elif task["type"] == "disk_warning":
                    ans = self.ask_llm(f"Место на диске заканчивается (занято {task['usage']}%). Коротко скажи об этом.")
                    self.speak(ans)
                    
                elif task["type"] == "voice_query":
                    query = task["text"]
                    
                    if self.pending_stranger_emb is not None:
                        name_clean = query.replace("это", "").replace("его зовут", "").replace("ее зовут", "").replace("меня зовут", "").strip()
                        name_clean = name_clean.title()
                        words_in_name = name_clean.split()
                        
                        if 0 < len(words_in_name) <= 3:
                            self.memory.add_person(name_clean, self.pending_stranger_emb, "Гость")
                            self.speak(f"Рад знакомству, {name_clean}. Сохранил вас в базу.")
                            self.pending_stranger_emb = None
                            continue
                        else:
                            self.speak("Я не совсем понял. Назовите, пожалуйста, просто имя.")
                            continue

                    local_answer = self.handle_local_intent(query)
                    if local_answer:
                        self.speak(local_answer)
                    else:
                        try:
                            active_names =[track.name for track in list(self.active_tracks.values()) if track.is_verified]
                        except RuntimeError:
                            active_names =[]
                        who = ", ".join(active_names) if active_names else "никого"
                        ans = self.ask_llm(query, context_data=who)
                        self.speak(ans)
                        
            except queue.Empty:
                continue

    def run(self):
        threading.Thread(target=self.camera_thread, daemon=True).start()
        
        while self.latest_frame is None:
            time.sleep(0.1)

        print("\n=== Агент запущен. Скажите 'Выход' для завершения. ===")
        if self.system_mode == "registering":
            self.speak("Система активирована. Пожалуйста, посмотрите в камеру.")

        threading.Thread(target=self.vision_thread, daemon=True).start()
        threading.Thread(target=self.audio_thread, daemon=True).start()
        threading.Thread(target=self.monitor_thread, daemon=True).start()
        threading.Thread(target=self.brain_thread, daemon=True).start()

        while self.running:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()
                
                mode_text = "Registration Mode..." if self.system_mode == "registering" else "Working Mode"
                cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                #Получаем рамки для отрисовки
                try:
                    tracks_to_draw = list(self.active_tracks.values())
                except RuntimeError:
                    tracks_to_draw =[]

                for track in tracks_to_draw:
                    x, y, w, h = track.x, track.y, track.w, track.h
                    color = (0, 255, 0) if track.is_verified else (0, 165, 255)
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    display_name = translit(track.name)
                    
                    if not track.is_verified:
                        time_tracked = time.time() - track.first_seen
                        if self.system_mode == "registering":
                            time_left = max(0, Config.REGISTRATION_TIME - time_tracked)
                            display_name = f"Scanning ({time_left:.1f}s)"
                        else:
                            time_left = max(0, 2.5 - time_tracked)
                            if time_left > 0:
                                display_name += f" ({time_left:.1f}s)"
                            
                    cv2.putText(frame, display_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if self.pending_stranger_emb is not None:
                    cv2.putText(frame, "LISTENING TO MICROPHONE...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.rectangle(frame, (0, frame.shape[0]-40), (220, frame.shape[0]), (0, 0, 0), -1)
                cv2.putText(frame, f"DB Size: {self.memory.people.count()}", (10, frame.shape[0]-15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                status = "Speaking" if self.is_speaking else "Listening..."
                color = (0, 0, 255) if self.is_speaking else (0, 255, 0)
                cv2.putText(frame, status, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.imshow("Agent Vision", frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.task_queue.put({"type": "exit"})
                break

        cv2.destroyAllWindows()
        os._exit(0)

if __name__ == "__main__":
    try:
        Orchestrator().run()
    except KeyboardInterrupt:
        os._exit(0)

