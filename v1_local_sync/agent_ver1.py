import os
import time
import threading
import queue
import gc
import logging

import cv2
import torch
import whisper
import speech_recognition as sr
import pygame
from PIL import Image
from gtts import gTTS
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info


logging.getLogger("transformers").setLevel(logging.ERROR)

class Config:
    MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
    VLM_MAX_PIXELS = 768 * 28 * 28
    ASR_MODEL_TYPE = "base"
    HISTORY_LIMIT = 5
    TEMP_AUDIO_PATH = "input.wav"
    RESPONSE_AUDIO_PATH = "output.mp3"

class VisionSystem: # Изображения захватываю фоном в реальном времени
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame


    def get_current_frame(self):
        for _ in range(10): 
            self.cap.grab()
        
        ret, frame = self.cap.read() 
        if ret:
            filename = f"view_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Снимок сделан")
            return filename
        return None

    def release(self):
        self.running = False
        self.cap.release()

class MultimodalCore: # В этом классе происходит управление моделями (VLM, ASR, TTS)
    def __init__(self):
        print(f"Работает ли CUDA? (CUDA: {torch.cuda.is_available()})...")
        
        self.asr = whisper.load_model(Config.ASR_MODEL_TYPE, device="cpu")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            Config.MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.processor = AutoProcessor.from_pretrained(
            Config.MODEL_ID, 
            min_pixels=256*28*28, 
            max_pixels=Config.VLM_MAX_PIXELS
        )
        
        pygame.mixer.init()
        self.history = []

    def process_voice(self, audio_data):
        with open(Config.TEMP_AUDIO_PATH, "wb") as f:
            f.write(audio_data.get_wav_data())
        result = self.asr.transcribe(Config.TEMP_AUDIO_PATH, language="ru", fp16=False)
        return result["text"].strip()

    def generate_response(self, text, image_path):
        gc.collect()
        torch.cuda.empty_cache()

        vision_messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(image_path).convert("RGB")},
                {"type": "text", "text": "Что сейчас перед камерой? Назови только объект одним словом."}
            ]
        }]
        v_prompt = self.processor.apply_chat_template(vision_messages, tokenize=False, add_generation_prompt=True)
        v_i, v_v = process_vision_info(vision_messages)
        v_inputs = self.processor(text=[v_prompt], images=v_i, videos=v_v, padding=True, return_tensors="pt").to("cuda")
        
        v_out = self.model.generate(**v_inputs, max_new_tokens=10, do_sample=False)
        current_vision = self.processor.batch_decode([v_out[0][len(v_inputs.input_ids[0]):]], skip_special_tokens=True)[0]
        current_vision = current_vision.strip().strip('.').strip(':')
        
        print(f"Робот видит следующее: {current_vision}")

        ## Моделька маленькая, поэтому зрение или память выбирается по ключевым словам
        memory_keywords = ["раньше", "первым", "предыдущим", "сначала", "до этого", "помнишь"]
        is_memory_query = any(word in text.lower() for word in memory_keywords)

        if is_memory_query:
            # В режиме памяти модели даём историю
            history_str = ""
            for i, m in enumerate(self.history):
                role = "Я" if m['role'] == 'assistant' else "Ты"
                history_str += f"{role} сказал: {m['content']}\n"
            
            final_prompt = (
                f"Это твоя память о прошлых объектах:\n{history_str}\n"
                f"Вопрос пользователя: {text}\n"
                "Ответь кратко на основе памяти."
            )
        else:
            # Если нужен режим зрения, то память модели не показываем
            final_prompt = (
                f"Инструкция: Перед тобой находится {current_vision}. "
                f"Ответь пользователю на вопрос: '{text}'. "
                "Твой ответ должен содержать название объекта, который ты видишь сейчас."
            )

        # Выход агента
        final_messages = [{"role": "user", "content": final_prompt}]
        f_prompt = self.processor.apply_chat_template(final_messages, tokenize=False, add_generation_prompt=True)
        f_inputs = self.processor(text=[f_prompt], padding=True, return_tensors="pt").to("cuda")
        
        f_out = self.model.generate(**f_inputs, max_new_tokens=50, do_sample=True, temperature=0.2)
        response_text = self.processor.batch_decode([f_out[0][len(f_inputs.input_ids[0]):]], skip_special_tokens=True)[0]

        response_text = response_text.replace("ТВОЙ КРАТКИЙ ОТВЕТ:", "").strip()

        # Здесь кладём в историю
        self.history.append({"role": "user", "content": text})
        self.history.append({"role": "assistant", "content": response_text})
        if len(self.history) > 8: self.history = self.history[-8:]
            
        if image_path and os.path.exists(image_path):
            try: os.remove(image_path)
            except: pass

        return response_text

    def speak(self, text):
        tts = gTTS(text=text, lang='ru')
        tts.save(Config.RESPONSE_AUDIO_PATH)
        pygame.mixer.music.load(Config.RESPONSE_AUDIO_PATH)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.unload()

class Orchestrator: # Само приложение
    def __init__(self):
        self.vision = VisionSystem()
        self.brain = MultimodalCore()
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

    def run(self):
        print("\nВсё включено, активно\n")
        
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1.5)

        while True:
            try:
                with self.mic as source:
                    print("Говорите: ")
                    audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=5)

                query = self.brain.process_voice(audio)
                if len(query) < 3: continue
                
                print(f"[*] Вы: {query}")
                
                if "выход" in query.lower() or "стоп" in query.lower():
                    break

                img = self.vision.get_current_frame()
                print(f"Сейчас обрабатывается картинка: {img}") # Для отладки
                response = self.brain.generate_response(query, img)
                
                print(f"Ответ: {response}")
                self.brain.speak(response)

            except Exception as e:
                print(f"Видимо что-то не так {e}")
                continue

if __name__ == "__main__":
    app = Orchestrator()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nЗавершение...")
    finally:
        app.vision.release()