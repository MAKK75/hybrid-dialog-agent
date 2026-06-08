import os
import sys
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'       
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'      
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' 
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
import base64
import tempfile 
from openai import OpenAI
from mem0 import Memory
import warnings
import logging
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("mem0").setLevel(logging.ERROR)

if not os.path.exists('face_detection_yunet_2023mar.onnx') or not os.path.exists('face_recognition_sface_2021dec.onnx'):
    print("ОШИБКА: Файлы моделей YuNet и SFace (.onnx) не найдены в папке!")
    sys.exit(1)

ENG_TO_RU_GLOSSARY = {
    "apple": "эппл",
    "google": "гугл",
    "openai": "оупен эй ай",
    "deepseek": "дипсик",
    "chromadb": "крома дб",
    "chroma": "крома",
    "python": "пайтон",
    "whisper": "виспир",
    "vlm": "ви эл эм",
    "llm": "эл эл эм",
    "api": "эй пи ай",
    "gpt": "джи пи ти",
    "yandexgpt": "яндекс джи пи ти",
    "yandex": "яндекс",
    "silero": "силеро",
    "tts": "ти ти эс",
    "stt": "эс ти ти",
    "yunet": "юнет",
    "sface": "сфейс",
    "onnx": "онникс",
    "cuda": "куда",
    "spacy": "спейси",
    "nlp": "эн эл пи",
    "iphone": "айфон",
    "ipad": "айпад",
    "mac": "мак",
    "macbook": "макбук",
    "gazprom": "газпром",
    "telegram": "телеграм",
    "youtube": "ютуб",
    "microsoft": "майкрософт",
    "windows": "виндовс",
    "android": "андройд",
    "github": "гитхаб",
    "linux": "линукс",
    "intel": "интел",
    "nvidia": "нвидиа",
    "amd": "а эм де",
    "ai": "эй ай",
    "chatgpt": "чат джи пи ти",
    "facebook": "фейсбук",
    "instagram": "инстаграм",
    "twitter": "твиттер",
    "tesla": "тесла",
    "spacex": "спейс икс",
    "amazon": "амазон",
    "netflix": "нетфликс",
    "sony": "сони",
    "samsung": "самсунг",
    "playstation": "плейстейшн",
    "xbox": "иксбокс",
    "uber": "убер",
    "zoom": "зум",
}

def translit_eng_word_to_ru(word):
    w = word.lower()
    if w in ENG_TO_RU_GLOSSARY:
        return ENG_TO_RU_GLOSSARY[w]
    
    if w.startswith("yu"):
        w = "ю" + w[2:]
    elif w.startswith("yo"):
        w = "йо" + w[2:]
    elif w.startswith("ya"):
        w = "я" + w[2:]
    elif w.startswith("ye"):
        w = "е" + w[2:]
    elif w.startswith("y") and len(w) > 1 and w[1] in "aeiou":
        w = "й" + w[1:]

    replacements = [
        ("wr", "r"),
        ("kn", "n"),
        ("ck", "k"),
        ("ph", "f"),
        ("sh", "ш"),
        ("ch", "ч"),
        ("tch", "ч"),
        ("th", "т"),  
        ("qu", "кв"),
        ("wh", "в"),
        ("gh", "г"),
    ]
    for eng, ru in replacements:
        w = w.replace(eng, ru)
        
    def magic_e_repl(match):
        vowel = match.group(1)
        consonant = match.group(2)
        if vowel == 'i':
            return "ай" + consonant
        elif vowel == 'a':
            return "ей" + consonant
        elif vowel == 'o':
            return "о" + consonant
        elif vowel == 'u':
            return "ью" + consonant
        return vowel + consonant
    
    w = re.sub(r'([aeiou])([b-df-hj-np-tv-z])e$', magic_e_repl, w)

    w = w.replace("oo", "у")
    w = w.replace("ee", "и")
    w = w.replace("ea", "и")
    w = w.replace("ay", "ей")
    w = w.replace("ai", "ей")
    w = w.replace("oy", "ой")
    w = w.replace("ey", "и")

    w = re.sub(r'c([eiy])', r'с\1', w)
    w = re.sub(r'g([eiy])', r'дж\1', w)
    w = w.replace("c", "к")
    w = w.replace("g", "г")
    w = w.replace("x", "кс")

    if w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou":
        w = w[:-1] + "и"
    elif w == "y":
        w = "уай"

    char_map = {
        "a": "а",
        "b": "б",
        "d": "д",
        "e": "е",
        "f": "ф",
        "h": "х",
        "i": "и",
        "j": "дж",
        "k": "к",
        "l": "л",
        "m": "м",
        "n": "н",
        "o": "о",
        "p": "п",
        "q": "к",
        "r": "р",
        "s": "с",
        "t": "т",
        "u": "у",
        "v": "в",
        "w": "в",
        "y": "ай",
        "z": "з"
    }
    
    res = ""
    for char in w:
        res += char_map.get(char, char)
    return res

def int_to_words_ru(n, gender="m"):
    if n == 0:
        return "ноль"
        
    units_m = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
    units_f = ["", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
    teens = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
    tens = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
    hundreds = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"]

    def _convert_chunk(num, is_feminine=False):
        w = []
        h = num // 100
        t = (num % 100) // 10
        u = num % 10
        if h > 0:
            w.append(hundreds[h])
        if t == 1:
            w.append(teens[u])
        else:
            if t > 1:
                w.append(tens[t])
            if u > 0:
                w.append(units_f[u] if is_feminine else units_m[u])
        return " ".join(w)

    if n < 0:
        return "минус " + int_to_words_ru(abs(n), gender)

    parts = []
    
    millions = n // 1000000
    if millions > 0:
        m_words = _convert_chunk(millions, is_feminine=False)
        last_digit = millions % 10
        last_two = millions % 100
        suffix = "миллионов"
        if last_two not in [11, 12, 13, 14]:
            if last_digit == 1:
                suffix = "миллион"
            elif last_digit in [2, 3, 4]:
                suffix = "миллиона"
        parts.append(f"{m_words} {suffix}")
        n %= 1000000

    thousands = n // 1000
    if thousands > 0:
        t_words = _convert_chunk(thousands, is_feminine=True)
        last_digit = thousands % 10
        last_two = thousands % 100
        suffix = "тысяч"
        if last_two not in [11, 12, 13, 14]:
            if last_digit == 1:
                suffix = "тысяча"
            elif last_digit in [2, 3, 4]:
                suffix = "тысячи"
        parts.append(f"{t_words} {suffix}")
        n %= 1000

    if n > 0 or not parts:
        parts.append(_convert_chunk(n, is_feminine=(gender == "f")))

    return " ".join(filter(None, parts)).strip()

def get_percent_suffix_int(n):
    last_digit = n % 10
    last_two = n % 100
    if last_two in [11, 12, 13, 14]:
        return "процентов"
    if last_digit == 1:
        return "процент"
    if last_digit in [2, 3, 4]:
        return "процента"
    return "процентов"

ORD_NOM_M = {
    1: "первый", 2: "второй", 3: "третий", 4: "четвертый", 5: "пятый",
    6: "шестой", 7: "седьмой", 8: "восьмой", 9: "девятый", 10: "десятый",
    11: "одиннадцатый", 12: "двенадцатый", 13: "тринадцатый", 14: "четырнадцатый",
    15: "пятнадцатый", 16: "шестнадцатый", 17: "семнадцатый", 18: "восемнадцатый",
    19: "девятнадцатый", 20: "двадцатый", 30: "тридцатый", 40: "сороковой",
    50: "пятидесятый", 60: "шестидесятый", 70: "семидесятый", 80: "восьмидесятый",
    90: "девяностый", 100: "сотый", 200: "двухсотый", 300: "трехсотый",
    400: "четырехсотый", 500: "пятисотый", 600: "шестисотый", 700: "семисотый",
    800: "восьмисотый", 900: "девятисотый"
}

def inflect_ordinal(nom_m, case_gender):
    if case_gender == "nom_m":
        return nom_m
    if nom_m == "третий":
        return {
            "nom_n": "третье",
            "nom_f": "третья",
            "prep": "третьем",
            "gen": "третьего",
            "dat": "третьему"
        }.get(case_gender, nom_m)
        
    if nom_m.endswith("ый") or nom_m.endswith("ой") or nom_m.endswith("ий"):
        stem = nom_m[:-2]
    else:
        stem = nom_m
        
    endings = {
        "nom_n": "ое",   
        "nom_f": "ая",   
        "prep": "ом",    
        "gen": "ого",    
        "dat": "ому"     
    }
    return stem + endings.get(case_gender, "")

def get_ordinal_word(val, case_gender):
    nom_m = ""
    if val in ORD_NOM_M:
        nom_m = ORD_NOM_M[val]
    elif val % 1000 == 0 and val < 1000000:
        k = val // 1000
        prefixes = {
            1: "", 2: "двух", 3: "трех", 4: "четырех", 5: "пяти",
            6: "шести", 7: "семи", 8: "восьми", 9: "девяти", 10: "десяти"
        }
        pref = prefixes.get(k, int_to_words_ru(k, gender="m").replace(" ", ""))
        nom_m = "тысячный" if k == 1 else pref + "тысячный"
    elif val % 1000000 == 0:
        k = val // 1000000
        prefixes = {
            1: "", 2: "двух", 3: "трех", 4: "четырех", 5: "пяти",
            6: "шести", 7: "семи", 8: "восьми", 9: "девяти", 10: "десяти"
        }
        pref = prefixes.get(k, int_to_words_ru(k, gender="m").replace(" ", ""))
        nom_m = "миллионный" if k == 1 else pref + "миллионный"
        
    if nom_m:
        return inflect_ordinal(nom_m, case_gender)
    return ""

def int_to_ordinal_ru(n, case_gender="nom_m"):
    if n == 0:
        return inflect_ordinal("нулевой", case_gender)
        
    if n % 100 in range(10, 20):
        last_val = n % 100
        preceding = (n // 100) * 100
    elif n % 10 != 0:
        last_val = n % 10
        preceding = (n // 10) * 10
    elif n % 100 != 0:
        last_val = n % 100
        preceding = (n // 100) * 100
    elif n % 1000 != 0:
        last_val = n % 1000
        preceding = (n // 1000) * 1000
    elif n % 1000000 != 0:
        last_val = n % 1000000
        preceding = (n // 1000000) * 1000000
    else:
        last_val = n
        preceding = 0
        
    ord_word = get_ordinal_word(last_val, case_gender)
    if preceding > 0:
        card_words = int_to_words_ru(preceding, gender="m")
        return f"{card_words} {ord_word}"
    return ord_word

def get_hour_suffix(n):
    last_digit = n % 10
    last_two = n % 100
    if last_two in [11, 12, 13, 14]:
        return "часов"
    if last_digit == 1:
        return "час"
    if last_digit in [2, 3, 4]:
        return "часа"
    return "часов"

def get_minute_suffix(n):
    last_digit = n % 10
    last_two = n % 100
    if last_two in [11, 12, 13, 14]:
        return "минут"
    if last_digit == 1:
        return "минута"
    if last_digit in [2, 3, 4]:
        return "минуты"
    return "минут"

def get_unit_suffix(n, unit, is_decimal=False):
    if is_decimal:
        return {"тыс": "тысячи", "млн": "миллиона", "млрд": "миллиарда", "трлн": "триллиона"}.get(unit, unit)
        
    last_digit = n % 10
    last_two = n % 100
    
    if unit == "тыс":
        if last_two in [11, 12, 13, 14]: return "тысяч"
        if last_digit == 1: return "тысяча"
        if last_digit in [2, 3, 4]: return "тысячи"
        return "тысяч"
        
    forms = {
        "млн": ["миллион", "миллиона", "миллионов"],
        "млрд": ["миллиард", "миллиарда", "миллиардов"],
        "трлн": ["триллион", "триллиона", "триллионов"]
    }
    if unit not in forms:
        return unit
        
    nom, gen_sg, gen_pl = forms[unit]
    if last_two in [11, 12, 13, 14]:
        return gen_pl
    if last_digit == 1:
        return nom
    if last_digit in [2, 3, 4]:
        return gen_sg
    return gen_pl


def convert_numbers_to_words(text):
    
    def repl_time(match):
        hr = int(match.group(1))
        mn = int(match.group(2))
        if mn == 0:
            return f"{int_to_words_ru(hr, gender='m')} {get_hour_suffix(hr)}"
        return f"{int_to_words_ru(hr, gender='m')} {get_hour_suffix(hr)} {int_to_words_ru(mn, gender='f')} {get_minute_suffix(mn)}"

    def repl_decimal_unit(match):
        int_part = int(match.group(1))
        frac_part = int(match.group(2))
        unit = match.group(3).strip('.')
        int_words = int_to_words_ru(int_part, gender="f")
        int_suffix = "целая" if (int_part % 10 == 1 and int_part % 100 != 11) else "целых"
        frac_words = int_to_words_ru(frac_part, gender="f")
        digits = len(match.group(2))
        frac_suffix = "десятая" if digits == 1 and frac_part % 10 == 1 else "десятых" if digits == 1 else "сотая" if digits == 2 and frac_part % 10 == 1 else "сотых" if digits == 2 else "тысячная" if frac_part % 10 == 1 else "тысячных"
        return f"{int_words} {int_suffix} {frac_words} {frac_suffix} {get_unit_suffix(0, unit, is_decimal=True)}"

    def repl_decimal_percent(match):
        int_part = int(match.group(1))
        frac_part = int(match.group(2))
        int_words = int_to_words_ru(int_part, gender="f")
        int_suffix = "целая" if (int_part % 10 == 1 and int_part % 100 != 11) else "целых"
        frac_words = int_to_words_ru(frac_part, gender="f")
        digits = len(match.group(2))
        frac_suffix = "десятая" if digits == 1 and frac_part % 10 == 1 else "десятых" if digits == 1 else "сотая" if digits == 2 and frac_part % 10 == 1 else "сотых" if digits == 2 else "тысячная" if frac_part % 10 == 1 else "тысячных"
        return f"{int_words} {int_suffix} {frac_words} {frac_suffix} процента"

    def repl_decimal(match):
        int_part = int(match.group(1))
        frac_part = int(match.group(2))
        int_words = int_to_words_ru(int_part, gender="f")
        int_suffix = "целая" if (int_part % 10 == 1 and int_part % 100 != 11) else "целых"
        frac_words = int_to_words_ru(frac_part, gender="f")
        digits = len(match.group(2))
        frac_suffix = "десятая" if digits == 1 and frac_part % 10 == 1 else "десятых" if digits == 1 else "сотая" if digits == 2 and frac_part % 10 == 1 else "сотых" if digits == 2 else "тысячная" if frac_part % 10 == 1 else "тысячных"
        return f"{int_words} {int_suffix} {frac_words} {frac_suffix}"

    def repl_integer_unit(match):
        val = int(match.group(1))
        unit = match.group(2).strip('.')
        words = int_to_words_ru(val, gender="f" if unit == "тыс" else "m")
        return f"{words} {get_unit_suffix(val, unit, is_decimal=False)}"

    def repl_integer_percent(match):
        val = int(match.group(1))
        return f"{int_to_words_ru(val, gender='m')} {get_percent_suffix_int(val)}"

    def repl_integer(match):
        val = int(match.group(1))
        return int_to_words_ru(val, gender="m")

    def repl_day_gen(match):
        return f"{match.group(1)} {int_to_ordinal_ru(int(match.group(2)), 'gen')} {match.group(3)}"

    def repl_day_dat(match):
        return f"{match.group(1)} {int_to_ordinal_ru(int(match.group(2)), 'dat')} {match.group(3)}"

    def repl_day_nom(match):
        return f"{int_to_ordinal_ru(int(match.group(1)), 'nom_n')} {match.group(2)}"

    def repl_year_prep(match):
        return f"{match.group(1)} {int_to_ordinal_ru(int(match.group(2)), 'prep')} {match.group(3)}"

    def repl_year_dat(match):
        return f"к {int_to_ordinal_ru(int(match.group(1)), 'dat')} году"

    def repl_year_gen_prep(match):
        return f"{match.group(1)} {int_to_ordinal_ru(int(match.group(2)), 'gen')} года"

    def repl_year_gen_plain(match):
        return f"{int_to_ordinal_ru(int(match.group(1)), 'gen')} года"

    def repl_year_prep_plain(match):
        return f"{int_to_ordinal_ru(int(match.group(1)), 'prep')} году"

    def repl_year_nom_plain(match):
        return f"{int_to_ordinal_ru(int(match.group(1)), 'nom_m')} год"

    
    text = re.sub(r'\b(\d{1,2}):(\d{2})\b', repl_time, text)
    
    text = re.sub(r'(\d+)[,.](\d+)\s*(тыс\.?|млн\.?|млрд\.?|трлн\.?)', repl_decimal_unit, text)
    
    text = re.sub(r'(\d+)[,.](\d+)\s*%', repl_decimal_percent, text)
    
    text = re.sub(r'(\d+)[,.](\d+)', repl_decimal, text)
    
    text = re.sub(r'(\d+)\s*(тыс\.?|млн\.?|млрд\.?|трлн\.?)', repl_integer_unit, text)
    
    text = re.sub(r'(\d+)\s*%', repl_integer_percent, text)
    
    text = re.sub(r'\b(с|до|после|около|от|из|без|для)\s+(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)', repl_day_gen, text)
    text = re.sub(r'\b(к)\s+(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)', repl_day_dat, text)
    text = re.sub(r'\b(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)', repl_day_nom, text)
    
    text = re.sub(r'\b(в|во|обо?|при)\s+(\d+)\s+(году|веке)', repl_year_prep, text)
    text = re.sub(r'\bк\s+(\d+)\s+году', repl_year_dat, text)
    text = re.sub(r'\b(до|с|после|из|около|от)\s+(\d+)\s+года', repl_year_gen_prep, text)
    text = re.sub(r'\b(\d+)\s+года', repl_year_gen_plain, text)
    text = re.sub(r'\b(\d+)\s+году', repl_year_prep_plain, text)
    text = re.sub(r'\b(\d+)\s+год', repl_year_nom_plain, text)
    
    text = re.sub(r'(\d+)', repl_integer, text)
    
    return text

def should_extract_facts(text):
    t_lower = text.lower().strip()
    
    if t_lower.endswith("?"):
        return False
        
    question_and_cmd_words = [
        "кто", "что", "где", "когда", "почему", "зачем", "как", "какой", 
        "сколько", "чей", "какие", "какая", "какое", "расскажи", "покажи", 
        "найди", "поищи", "загугли", "видишь", "опиши", "объясни", "назови",
        "выход", "стоп", "назад", "привет", "здравствуй", "здравствуйте", "пока"
    ]
    words = t_lower.split()
    if not words:
        return False
        
    if words[0] in question_and_cmd_words:
        return False
        
    stop_words = ["да", "нет", "ок", "ладно", "понял", "хорошо", "отлично", "спасибо"]
    if t_lower in stop_words or len(words) < 3:
        if "зовут" in t_lower or t_lower.startswith("я "):
            return True
        return False
        
    return True

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
    SYSTEM_CHECK_INTERVAL = 3000 #Для проверки диска, можно уменьшить, чтобы посмотреть (повысил, чтобы агент лишний раз не перебивал) 

    # --- РУБИЛЬНИК СБОРА ФАКТОВ ---
    ENABLE_FACT_EXTRACTION = True  # True — собирать факты о пользователе, False — отключить сбор

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
        self.last_db_check = 0 

class FakeEmbedding:
    """Заглушка для ChromaDB, предотвращающая скачивание текстовых ONNX моделей из интернета."""
    def __call__(self, input):
        return [[0.0] * 128 for _ in input]

    def name(self) -> str:
        return "fake_embedding" 


class MemorySystem:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=Config.DB_PATH)
        self.people = self.client.get_or_create_collection(
            name="people", 
            metadata={"hnsw:space": "cosine"},
            embedding_function=FakeEmbedding() 
        )
        
        mem0_config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": f"gpt://{Config.YANDEX_FOLDER_ID}/yandexgpt-5-lite/latest",
                    "api_key": Config.YANDEX_API_KEY,
                    "openai_base_url": "https://ai.api.cloud.yandex.net/v1",
                    "max_tokens": 1000,
                    "temperature": 0.1
                }
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                }
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "mem0_agent_facts",
                    "path": os.path.join(Config.DB_PATH, "mem0_chroma")
                }
            }
        }

        try:
            self.mem0 = Memory.from_config(mem0_config)
            print("[Система] Mem0 успешно инициализирован (YandexGPT 5 + Локальный Embedder).")
        except Exception as e:
            print(f"[Система] Критическая ошибка при инициализации Mem0: {e}")
            raise e

    def extract_facts_via_yandex_gpt(self, person_name, text):
        headers = {
            "Authorization": f"Api-Key {Config.YANDEX_API_KEY}",
            "x-folder-id": Config.YANDEX_FOLDER_ID,
            "Content-Type": "application/json"
        }
        system_prompt = (
            f"Ты — модуль извлечения долгосрочных фактов. Твоя задача — проанализировать реплику пользователя "
            f"и извлечь из неё краткие, чёткие факты о человеке по имени {person_name} (увлечения, предпочтения, "
            f"любимые напитки/вещи, биография).\n"
            f"Правила:\n"
            f"1. Формулируй каждый факт кратко в третьем лице. Например: '{person_name} любит пить черный чай по утрам'.\n"
            f"2. Пиши каждый факт СТРОГО с новой строки. Не используй дефисы, цифры, маркеры списка (*, -), кавычки или вводные фразы.\n"
            f"3. Если в реплике нет фактической информации о пользователе (вопросы, команды, общие фразы), ничего не пиши (верни пустой ответ)."
        )
        data = {
            "modelUri": f"gpt://{Config.YANDEX_FOLDER_ID}/{Config.MODEL_ID}/latest",
            "completionOptions": {"stream": False, "temperature": 0.1, "maxTokens": 300},
            "messages":[
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": text}
            ]
        }
        try:
            response = requests.post(Config.API_URL, headers=headers, json=data, timeout=8)
            if response.status_code == 200:
                res_json = response.json()
                content = res_json['result']['alternatives'][0]['message']['text'].strip()
                lines = [line.strip().strip("*-• ").strip() for line in content.split("\n")]
                return [line for line in lines if len(line) > 5 and not line.lower().startswith("пользователь")]
        except Exception as e:
            print(f"[Ошибка извлечения фактов через YandexGPT]: {e}")
        return []

    def add_fact(self, person_name, text, is_global=False):
        """Интеллектуальный сбор фактов с использованием собственного парсера и прямого импорта в Mem0"""
        user_id = "global_shared" if is_global else translit(person_name).lower().replace(" ", "_")
        
        extracted_facts = self.extract_facts_via_yandex_gpt(person_name, text)
        if not extracted_facts:
            print(f"\n[Mem0 - Пропуск] Из фразы '{text}' не выделено новых фактов.")
            return

        for fact in extracted_facts:
            try:
                self.mem0.add(fact, user_id=user_id, infer=False)
                print(f"[Mem0 Direct] Успешно сохранен факт для {user_id}: {fact}")
            except Exception as e:
                print(f"[Mem0 Direct] Ошибка при сохранении факта: {e}")

    def get_facts_for(self, person_name, query=None):
        """Возвращает список фактов. Если передан query, выполняется семантический поиск."""
        user_id = translit(person_name).lower().replace(" ", "_")
        all_facts = []
        
        try:
            if query:
                personal_res = self.mem0.search(query, filters={"user_id": user_id})
                global_res = self.mem0.search(query, filters={"user_id": "global_shared"})
            else:
                personal_res = self.mem0.get_all(filters={"user_id": user_id})
                global_res = self.mem0.get_all(filters={"user_id": "global_shared"})
            
            if personal_res and "results" in personal_res:
                for item in personal_res["results"]:
                    all_facts.append(item["memory"])
            if global_res and "results" in global_res:
                for item in global_res["results"]:
                    all_facts.append(item["memory"])
                    
        except Exception as e:
            print(f"[Mem0] Ошибка при извлечении фактов: {e}")
            
        return list(set(all_facts))

    def update_person_name(self, face_embedding, new_name):
        if self.people.count() == 0: return None
        flat_embedding = face_embedding.flatten().tolist()
        results = self.people.query(query_embeddings=[flat_embedding], n_results=1)
        
        if results['ids'] and len(results['ids'][0]) > 0:
            if results['distances'][0][0] <= Config.FACE_MATCH_THRESHOLD:
                doc_id = results['ids'][0][0]
                doc_data = json.loads(results['documents'][0][0])
                
                doc_data['name'] = new_name
                self.people.update(
                    ids=[doc_id],
                    documents=[json.dumps(doc_data)]
                )
                return doc_data
        return None

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
    
    def find_person_by_name(self, name):
        if self.people.count() == 0: return None
        db_data = self.people.get()
        if db_data and 'documents' in db_data:
            for i, doc_str in enumerate(db_data['documents']):
                try:
                    doc = json.loads(doc_str)
                    if doc.get("name", "").lower() == name.lower():
                        return {
                            "id": db_data['ids'][i],
                            "name": doc.get("name"),
                            "relation": doc.get("relation")
                        }
                except Exception:
                    pass
        return None

class Orchestrator:
    def __init__(self):
        pygame.mixer.init()
       
        print("\n[Система] Загрузка локальной нейросети голоса (Silero TTS)...")
        import torch
        self.tts_device = torch.device('cpu')
        torch.set_num_threads(4)
        local_file = 'v4_ru.pt'
        if not os.path.exists(local_file):
            torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v4_ru.pt', local_file)
        self.tts_model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
        self.tts_model.to(self.tts_device)
        self.tts_sample_rate = 48000
        self.tts_speaker = 'xenia' 
        
        print("[Система] Инициализация аудиодвижка (первичный прогрев)... Пожалуйста, подождите.")
        t_start = time.time()
        try:
            self.tts_model.apply_tts(text="Инициализация аудиосистемы завершена.", speaker=self.tts_speaker, sample_rate=self.tts_sample_rate)
            self.tts_model.apply_tts(text="Привет", speaker=self.tts_speaker, sample_rate=self.tts_sample_rate)
        except Exception as e:
            print(f"[Система] Ошибка прогрева: {e}")
        print(f"[Система] Голос готов к работе! Прогрев занял {time.time()-t_start:.2f} сек.")
        
        print("\n[Система] Загрузка локального распознавателя речи (Whisper Small)...")
        try:
            from faster_whisper import WhisperModel
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            self.whisper_model = WhisperModel("small", device=device, compute_type=compute_type)
            print(f"[Система] Локальный STT Whisper ('small') успешно загружен на {device.upper()}.")
        except Exception as e:
            print(f"[Система] Локальный Whisper недоступен ({e}). Будет использован Google STT.")
            print("[Система] Для активации Whisper выполните: pip install faster-whisper")
            self.whisper_model = None

        self.memory = MemorySystem()
        self.system_mode = "registering" if self.memory.people.count() == 0 else "working"
        
        self.current_extraction_thread = None  
        self.manual_active_user = None         
        
        self.latest_frame = None
        self.active_tracks = {}
        self.next_track_id = 1
        
        self.last_seen_owner_time = time.time()
        self.owner_present = True 
        
        self.pending_stranger_emb = None
        self.stranger_prompt_time = 0 
        
        self.last_active_user = "Создатель"
        self.last_speak_finished_time = 0
        
        self.is_speaking = False
        self.is_processing = False
        self.running = True
        self.task_queue = queue.Queue()

    def speak(self, input_data):
        self.is_speaking = True
        print("\n[Агент]: ", end="", flush=True)

        if isinstance(input_data, str):
            input_data = [input_data]

        import re
        import threading
        import queue
        import time

        sentence_queue = queue.Queue()
        audio_queue = queue.Queue()
        
        def tts_worker():
            try:
                import soundfile as sf
                idx = 0
                while True:
                    sentence = sentence_queue.get()
                    if sentence is None: 
                        audio_queue.put(None)
                        break
                    
                    filename = f"voice_{idx}_{int(time.time()*1000)}.wav"
                    idx += 1
                    try:
                        t1 = time.time()
                        
                        clean_sentence = sentence
                        
                        def replace_eng(match):
                            word = match.group(0)
                            return translit_eng_word_to_ru(word)
                        clean_sentence = re.sub(r'[a-zA-Z]+', replace_eng, clean_sentence)
                        
                        clean_sentence = convert_numbers_to_words(clean_sentence)
                        
                        audio_tensor = self.tts_model.apply_tts(
                            text=clean_sentence, 
                            speaker=self.tts_speaker, 
                            sample_rate=self.tts_sample_rate
                        )
                        sf.write(filename, audio_tensor.numpy(), self.tts_sample_rate)
                        print(f"\n[Debug] Часть '{sentence}' -> '{clean_sentence}' сгенерирована за {time.time()-t1:.2f} сек.")
                        audio_queue.put(filename)
                    except Exception as e:
                        print(f"\n[Ошибка Local TTS при синтезе '{sentence}']: {e}")
                    finally:
                        sentence_queue.task_done()
            except Exception as tts_err:
                print(f"\n[Критический сбой tts_worker]: {tts_err}")
                audio_queue.put(None)  

        def player_worker():
            try:
                while True:
                    filename = audio_queue.get()
                    if filename is None: 
                        break
                    try:
                        pygame.mixer.music.load(filename)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy() and self.running: 
                            time.sleep(0.01) 
                        pygame.mixer.music.unload()
                        if os.path.exists(filename): 
                            os.remove(filename)
                    except Exception:
                        pass
                    finally:
                        audio_queue.task_done()
            except Exception as play_err:
                print(f"\n[Критический сбой player_worker]: {play_err}")

        tts_thread = threading.Thread(target=tts_worker, daemon=True)
        player_thread = threading.Thread(target=player_worker, daemon=True)
        tts_thread.start()
        player_thread.start()

        buffer = ""
        try:
            for chunk in input_data:
                print(chunk, end="", flush=True)
                buffer += chunk
                
                if any(p in buffer for p in ['.', '!', '?']):
                    parts = re.split(r'(?<=[.!?])', buffer)
                    for part in parts[:-1]:
                        clean_part = part.strip()
                        if clean_part and any(c.isalnum() for c in clean_part):
                            sentence_queue.put(clean_part)
                    buffer = parts[-1]
            
            if buffer.strip():
                clean_part = buffer.strip()
                if clean_part and any(c.isalnum() for c in clean_part):
                    sentence_queue.put(clean_part)
                    
        except Exception:
            pass
        finally:
            print("\n")
            sentence_queue.put(None)
            player_thread.join(timeout=30)  
            self.last_speak_finished_time = time.time()
            self.is_speaking = False

    def handle_local_intent(self, text):
        t = text.lower()
        active_descriptions = []
        try:
            for track in list(self.active_tracks.values()):
                if track.is_verified:
                    if track.relation == "Создатель":
                        active_descriptions.append(f"{track.name} (Создатель)")
                    else:
                        active_descriptions.append(f"{track.name}")
        except RuntimeError:
            pass
            
        if "кого" in t and "видишь" in t:
            if not active_descriptions: return "Я сейчас никого не вижу."
            return f"Я вижу: {', '.join(active_descriptions)}."
        
        if "как тебя зовут" in t:
            return "Я ваша локальная охранная система."
            
        if "кто перед тобой" in t or "кто я" in t:
            if not active_descriptions: return "Передо мной никого нет."
            return f"Передо мной находится {', '.join(active_descriptions)}."
        return None

    def ask_llm(self, text, context_data="", person_name="Создатель"):
        headers = {
            "Authorization": f"Api-Key {Config.YANDEX_API_KEY}",
            "x-folder-id": Config.YANDEX_FOLDER_ID,
            "Content-Type": "application/json"
        }
        disk_free = 100 - psutil.disk_usage('/').percent
        
        knowledge_context = ""
        person_facts = self.memory.get_facts_for(person_name, query=text)
        if person_facts:
            facts_str = "\n- ".join(person_facts)
            knowledge_context = (
                f"Тебе известны следующие факты о собеседнике по имени {person_name}:\n"
                f"{facts_str}\n"
                f"Используй эти факты естественно во время диалога, если это уместно. "
                f"Если пользователь спрашивает 'что ты знаешь обо мне', обязательно кратко перечисли эти факты."
            )

        system_prompt = (
            "Ты умный, эрудированный и харизматичный ИИ-напарник с широким кругозором. Твоя задача — вести интересный диалог. "
            "Ты обязан отвечать на ЛЮБЫЕ общеобразовательные и энциклопедические вопросы пользователя (о науке, космосе, истории, ракетах, технологиях и известных людях) свободно и подробно, "
            "используя весь свой заложенный запас знаний. Не ограничивай свои знания только локальным контекстом компьютера.\n"
            "При этом, если вопрос касается лично собеседника, опирайся на известные тебе факты.\n"
            "Отвечай кратко, живо, максимум 2-3 предложения. Избегай скучных списков и не повторяйся.\n"
            "ВАЖНО: Пиши все числа и числительные только словами на русском языке (например, 'четыре' вместо '4', 'второй' вместо '2-й'), "
            "так как твой ответ будет озвучен синтезатором речи. Не используй латиницу без крайней необходимости.\n"
            f"{knowledge_context}\n"
            f"КОНТЕКСТ: Человек перед камерой: {context_data}. Свободно на диске: {disk_free:.1f}%."
        )

        data = {
            "modelUri": f"gpt://{Config.YANDEX_FOLDER_ID}/{Config.MODEL_ID}/latest",
            "completionOptions": {"stream": True, "temperature": 0.5, "maxTokens": "150"},
            "messages":[
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": text}
            ]
        }
        
        try:
            response = requests.post(Config.API_URL, headers=headers, json=data, timeout=5, stream=True)
            response.raise_for_status()
            
            full_text = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        msg = chunk['result']['alternatives'][0]['message']['text']
                        new_text = msg[len(full_text):]
                        full_text = msg
                        yield new_text
                    except Exception:
                        continue
        except Exception:
            yield "Связь с сервером прервана."

    def ask_vlm(self, base64_image, recognized_people=""):
        headers = {
            "Authorization": f"Api-Key {Config.YANDEX_API_KEY}",
            "Content-Type": "application/json"
        }
        
        prompt_text = "Ты глаза ИИ. Внимательно изучи изображение с веб-камеры и опиши обстановку живо, интересно и кратко (максимум 2 предложения)."
        if recognized_people:
            prompt_text += f" К твоему сведению, человек в кадре — это {recognized_people}. Обязательно назови его по имени или упомяни его статус Создателя в описании обстановки."

        payload = {
            "model": f"gpt://{Config.YANDEX_FOLDER_ID}/qwen3.6-35b-a3b/latest",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 2000,  
            "temperature": 0.5
        }
        
        try:
            response = requests.post("https://ai.api.cloud.yandex.net/v1/chat/completions", headers=headers, json=payload, timeout=15)
            
            if response.status_code != 200:
                print(f"\n[Debug VLM Ошибка Сети]: {response.text}")
                return "Извините, сервер визуального анализа отклонил запрос."
                
            resp_json = response.json()
            message_data = resp_json.get("choices", [{}])[0].get("message", {})
            ans = message_data.get("content", "")
            
            if not ans or ans.strip() == "":
                reasoning = message_data.get("reasoning_content", "")
                if reasoning:
                    paragraphs = [p.strip() for p in reasoning.split("\n") if p.strip()]
                    if paragraphs:
                        ans = paragraphs[-1].strip('"')  
            
            if not ans or ans.strip() == "":
                print(f"\n[Debug VLM Пустой ответ]: {resp_json}")
                return "Модель обработала кадр, но вернула пустое описание."
                
            return ans.strip()
            
        except Exception as e:
            print(f"\n[Debug VLM Исключение]: {e}")
            return "К сожалению, произошла техническая ошибка при анализе картинки."

    def search_yandex_gen(self, search_query):
        headers = {
            "Authorization": f"Api-Key {Config.YANDEX_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "messages": [
                {
                    "content": search_query,
                    "role": "ROLE_USER"
                }
            ],
            "folderId": Config.YANDEX_FOLDER_ID,
            "searchType": "SEARCH_TYPE_RU"
        }
        try:
            url = "https://searchapi.api.cloud.yandex.net/v2/gen/search"
            response = requests.post(url, headers=headers, json=data, timeout=12)
            if response.status_code == 200:
                res_json = response.json()
                answer = ""
                
                if isinstance(res_json, list):
                    for item in res_json:
                        if isinstance(item, dict):
                            answer += item.get("message", {}).get("content", "")
                
                elif isinstance(res_json, dict):
                    answer = res_json.get("message", {}).get("content", "")
                
                if answer:
                    answer = re.sub(r'\[\d+(?:\s*,\s*\d+)*\]', '', answer)
                    return answer.strip()
                    
            print(f"\n[Search API] Ошибка {response.status_code}: {response.text}")
        except Exception as e:
            print(f"\n[Search API] Ошибка соединения: {e}")
        return None
        

    def extract_fact_background(self, user_text, person_name="Создатель"):
        self.memory.add_fact(person_name, user_text)

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
            if self.is_speaking:
                self.last_seen_owner_time = time.time()
                for track in self.active_tracks.values():
                    track.last_seen = time.time()
                time.sleep(0.2)
                continue

            if self.latest_frame is None:
                time.sleep(0.05)
                continue
            
            frame = self.latest_frame.copy()
            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            
            if self.pending_stranger_emb is not None:
                if time.time() - self.stranger_prompt_time > 40: 
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
                        tracks_items = []

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

                    if track.name == "Незнакомец" and (current_time - track.last_db_check > 0.3):
                        track.last_db_check = current_time
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
                            if time_tracked >= Config.REGISTRATION_TIME and not track.prompted and self.pending_stranger_emb is None:
                                if not self.is_speaking and not self.is_processing:
                                    self.pending_stranger_emb = feature
                                    self.stranger_prompt_time = time.time()
                                    self.task_queue.put({"type": "registration_prompt"})
                                    track.prompted = True
                        
                        elif self.system_mode == "working":
                            if time_tracked > Config.REGISTRATION_TIME and not track.prompted and self.pending_stranger_emb is None:
                                if not self.is_speaking and not self.is_processing:
                                    self.pending_stranger_emb = feature
                                    self.stranger_prompt_time = time.time()
                                    self.task_queue.put({"type": "stranger_detected"})
                                    track.prompted = True

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
                    if not self.is_speaking and not self.is_processing:
                        self.task_queue.put({"type": "owner_returned"})

            time.sleep(0.02)

    def audio_thread(self):
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = False  
        recognizer.pause_threshold = 1.2
        recognizer.phrase_threshold = 0.2
        recognizer.non_speaking_duration = 0.6
        
        try:
            with sr.Microphone() as source:
                print("[Аудио] Настройка уровня шума...")
                recognizer.adjust_for_ambient_noise(source, duration=1.5)
                if recognizer.energy_threshold < 150:
                    recognizer.energy_threshold = 150
                print(f"[Аудио] Микрофон откалиброван (порог: {int(recognizer.energy_threshold)}). Готов к работе.")
        except Exception as e:
            print(f"[Аудио] Не удалось откалибровать микрофон ({e}). Установлен дефолтный порог 150.")
            recognizer.energy_threshold = 150

        was_speaking = True
        
        while self.running:
            try:
                if self.is_speaking or self.is_processing:
                    time.sleep(0.2)
                    was_speaking = True
                    continue
                
                if was_speaking:
                    time.sleep(0.5)  
                    if not (self.is_speaking or self.is_processing):
                        print("\n[Аудио] Микрофон активен (вы можете говорить)...")
                    was_speaking = False

                with sr.Microphone() as source:
                    try:
                        audio = recognizer.listen(source, timeout=3, phrase_time_limit=10)
                    except sr.WaitTimeoutError:
                        continue  
                
                if self.is_speaking or self.is_processing or (time.time() - self.last_speak_finished_time < 1.2):
                    continue
                    
                query = None
                
                if self.whisper_model:
                    try:
                        wav_data = audio.get_wav_data(convert_rate=16000, convert_width=2)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                            temp_wav.write(wav_data)
                            temp_wav_path = temp_wav.name
                            
                        try:
                            known_names = []
                            try:
                                if self.memory.people.count() > 0:
                                    db_data = self.memory.people.get()
                                    if db_data and 'documents' in db_data:
                                        for doc_str in db_data['documents']:
                                            try:
                                                doc = json.loads(doc_str)
                                                if "name" in doc:
                                                    known_names.append(doc["name"])
                                            except Exception:
                                                pass
                            except Exception:
                                pass

                            glossary_words = list(ENG_TO_RU_GLOSSARY.keys()) + list(ENG_TO_RU_GLOSSARY.values())
                            prompt_words = ["Привет", "Валси", "меня зовут", "мое имя", "моё имя"] + known_names + glossary_words
                            dynamic_prompt = ", ".join(sorted(list(set(prompt_words)))[:100])

                            segments, info = self.whisper_model.transcribe(
                                temp_wav_path, 
                                beam_size=3, 
                                language="ru",
                                initial_prompt=dynamic_prompt,
                                vad_filter=True  
                            )
                            query = "".join(segment.text for segment in segments).strip().lower()
                        finally:
                            if os.path.exists(temp_wav_path):
                                os.unlink(temp_wav_path)
                    except Exception as whisper_err:
                        print(f"\n[Аудио] Сбой локального Whisper ({whisper_err}), пробуем Google STT...")
                
                if not query:
                    try:
                        query = recognizer.recognize_google(audio, language="ru-RU").lower()
                    except sr.UnknownValueError:
                        continue
                    except sr.RequestError as e:
                        print(f"\n[Аудио] Google STT недоступен ({e}).")
                        continue
                
                if self.is_speaking or self.is_processing:
                    continue
                    
                if not query or query.strip() == "":
                    continue
                    
                print(f"[Вы]: {query}")
                
                if any(w in query for w in ["выход", "стоп", "отключись"]):
                    with self.task_queue.mutex:
                        self.task_queue.queue.clear()
                    self.task_queue.put({"type": "exit"})
                    break
                    
                self.task_queue.put({"type": "voice_query", "text": query})
            except Exception as e:
                time.sleep(1)

    def monitor_thread(self):
        last_warning_time = time.time() 
        
        while self.running:
            usage = psutil.disk_usage('/').percent
            free_percent = 100 - usage
            if free_percent <= Config.DISK_CRITICAL_PERCENT:
                current_time = time.time()
                if current_time - last_warning_time > Config.SYSTEM_CHECK_INTERVAL:
                    self.task_queue.put({"type": "disk_warning", "usage": usage})
                    last_warning_time = current_time
            time.sleep(5)

    def brain_thread(self):
        while self.running:
            try:
                task = self.task_queue.get(timeout=1)
                self.is_processing = True 
                
                if task["type"] == "exit":
                    self.speak("Отключаю системы. До свидания!")
                    self.running = False
                    self.is_processing = False
                    break
                    
                elif task["type"] == "registration_prompt":
                    self.speak("Здравствуйте! Я вижу вас впервые. Как мне к вам обращаться?")
                    self.stranger_prompt_time = time.time()
                    
                elif task["type"] == "owner_returned":
                    ans = self.ask_llm("Я не видел создателя больше минуты, и вот он вернулся в кадр. Поздоровайся с ним кратко.")
                    self.speak(ans)
                    
                elif task["type"] == "stranger_detected":
                    self.speak("Я вижу новое лицо. Подскажите, как зовут этого человека?")
                    self.stranger_prompt_time = time.time()
                    
                elif task["type"] == "disk_warning":
                    ans = self.ask_llm(f"Место на диске заканчивается (занято {task['usage']}%). Коротко скажи об этом.")
                    self.speak(ans)
                    
                elif task["type"] == "voice_query":
                    query = task["text"]
                    
                    if self.current_extraction_thread and self.current_extraction_thread.is_alive():
                        print("\n[Синхронизация] Ожидание завершения записи предыдущего факта...")
                        self.current_extraction_thread.join()
                    
                    active_name = self.last_active_user
                    active_relation = "Создатель"
                    
                    try:
                        active_tracks_list = list(self.active_tracks.values())
                        verified_tracks = [t for t in active_tracks_list if t.is_verified]
                        verified_names = [t.name for t in verified_tracks]
                    except Exception:
                        verified_tracks = []
                        verified_names = []
                        
                    if self.manual_active_user and self.manual_active_user in verified_names:
                        active_name = self.manual_active_user
                        for track in verified_tracks:
                            if track.name == self.manual_active_user:
                                active_relation = track.relation
                                break
                    elif verified_tracks:
                        active_name = verified_tracks[0].name
                        active_relation = verified_tracks[0].relation
                        self.manual_active_user = None  
                        
                    self.last_active_user = active_name
                    
                    target_name = None
                    matched_trigger = None
                    query_lower = query.lower()
                    
                    long_triggers = [
                        "с тобой разговаривает",
                        "с тобой говорит",
                        "сейчас говорит",
                        "передаю слово",
                        "это говорит",
                        "говорит"
                    ]
                    
                    for trigger in long_triggers:
                        if trigger in query_lower:
                            matched_trigger = trigger
                            idx = query_lower.find(trigger) + len(trigger)
                            raw_name_part = query[idx:].strip(" ,:-.!?")
                            words = raw_name_part.split()
                            if words:
                                target_name = words[0].title()
                            break
                    
                    if not target_name:
                        short_triggers = ["это", "я"]
                        for trigger in short_triggers:
                            if query_lower.startswith(trigger + " "):
                                raw_name_part = query[len(trigger):].strip(" ,:-.!?")
                                words = raw_name_part.split()
                                if words:
                                    potential_name = words[0].title()
                                    try:
                                        known_names = [t.name.lower() for t in list(self.active_tracks.values()) if t.is_verified]
                                    except Exception:
                                        known_names = []
                                        
                                    if potential_name.lower() in known_names:
                                        target_name = potential_name
                                        matched_trigger = trigger
                                        break
                    
                    if target_name:
                        active_name = target_name
                        active_relation = "Гость"
                        
                        try:
                            for track in list(self.active_tracks.values()):
                                if track.name.lower() == target_name.lower():
                                    active_name = track.name
                                    active_relation = track.relation
                                    break
                        except Exception:
                            pass
                            
                        self.last_active_user = active_name
                        self.manual_active_user = active_name  
                        
                        trigger_idx = query_lower.find(matched_trigger)
                        name_pos = query_lower.find(target_name.lower(), trigger_idx)
                        if name_pos != -1:
                            cleaned_query = query[name_pos + len(target_name):].strip(" ,:-.!?")
                        else:
                            cleaned_query = ""
                            
                        if cleaned_query:
                            query = cleaned_query
                            print(f"\n[Контекст] Переключение на {active_name}. Продолжение реплики: '{query}'")
                        else:
                            self.speak(f"Привет, {active_name}! Я переключился на твой профиль. Слушаю тебя.")
                            self.is_processing = False
                            continue

                    if self.pending_stranger_emb is not None:
                        name_clean = query.lower()
                        for phrase in ["меня зовут", "мое имя", "моё имя", "его зовут", "ее зовут", "это ", "я "]:
                            name_clean = name_clean.replace(phrase, "")
                            
                        name_clean = re.sub(r'[^a-zA-Zа-яА-ЯёЁ\s-]', '', name_clean)
                        name_clean = name_clean.strip().title()
                        words_in_name = name_clean.split()
                        
                        if 0 < len(words_in_name) <= 3:
                            existing_person = self.memory.find_person_by_name(name_clean)
                            
                            if existing_person:
                                relation = existing_person["relation"]
                                flat_embedding = self.pending_stranger_emb.flatten().tolist()
                                self.memory.people.update(
                                    ids=[existing_person["id"]],
                                    embeddings=[flat_embedding]
                                )
                                if relation == "Создатель":
                                    self.speak(f"О, {name_clean}! Прошу прощения, не узнал вас сразу из-за освещения. Рад вашему возвращению!")
                                    self.system_mode = "working"
                                    self.last_active_user = name_clean
                                else:
                                    self.speak(f"А, это вы, {name_clean}! Рад видеть вас снова.")
                                    self.last_active_user = name_clean
                            else:
                                relation = "Создатель" if self.system_mode == "registering" else "Гость"
                                self.memory.add_person(name_clean, self.pending_stranger_emb, relation)
                                
                                if relation == "Создатель":
                                    self.speak(f"Приятно познакомиться, {name_clean}! Я зарегистрировал вас как Создателя системы.")
                                    self.system_mode = "working"
                                    self.last_active_user = name_clean 
                                else:
                                    self.speak(f"Рад знакомству, {name_clean}. Сохранил вас в базу как гостя.")
                                    self.last_active_user = name_clean
                                
                            self.pending_stranger_emb = None
                            
                            try:
                                for track in self.active_tracks.values():
                                    if track.name == "Незнакомец" or track.name == "Создатель":
                                        track.name = name_clean
                                        track.relation = relation
                                        track.is_verified = True
                            except Exception:
                                pass
                                
                            self.is_processing = False
                            continue
                        else:
                            self.speak("Я не совсем понял. Назовите, пожалуйста, просто имя.")
                            self.is_processing = False
                            continue

                    explicit_rename_triggers = ["меня зовут", "мое имя", "моё имя", "зови меня", "называй меня"]
                    if any(tr in query.lower() for tr in explicit_rename_triggers) and self.pending_stranger_emb is None:
                        matched_trigger = next(tr for tr in explicit_rename_triggers if tr in query.lower())
                        new_name = query[query.lower().find(matched_trigger) + len(matched_trigger):].strip()
                        
                        new_name = new_name.replace("это", "").strip()
                        new_name = re.sub(r'[^a-zA-Zа-яА-ЯёЁ\s-]', '', new_name)
                        new_name = new_name.strip().title()
                        
                        if 0 < len(new_name.split()) <= 2:
                            active_track = None
                            try:
                                for t in self.active_tracks.values():
                                    if t.is_verified and t.feature is not None:
                                        active_track = t
                                        break
                            except Exception:
                                pass
                            
                            if active_track is not None:
                                updated_data = self.memory.update_person_name(active_track.feature, new_name)
                                if updated_data:
                                    old_name = active_track.name
                                    active_track.name = new_name
                                    self.last_active_user = new_name 
                                    self.speak(f"Хорошо, я запомнил, что вас зовут {new_name}.")
                                    self.is_processing = False
                                    continue

                    memory_keywords = ["знаешь", "обо мне", "факты", "помнишь", "кто я", "как зовут"]
                    is_memory_query = any(w in query.lower() for w in memory_keywords)
                    
                    if Config.ENABLE_FACT_EXTRACTION and is_memory_query and self.current_extraction_thread and self.current_extraction_thread.is_alive():
                        print("\n[Синхронизация] Ожидание записи предыдущего факта перед ответом...")
                        self.current_extraction_thread.join() 

                    if Config.ENABLE_FACT_EXTRACTION:
                        if should_extract_facts(query):
                            self.current_extraction_thread = threading.Thread(
                                target=self.extract_fact_background, 
                                args=(query, active_name), 
                                daemon=True
                            )
                            self.current_extraction_thread.start()
                        else:
                            print(f"\n[Mem0 - Пропуск] Фраза '{query}' отфильтрована локально как не содержащая новых фактов.")
                    else:
                        print(f"\n[Mem0 - Пропуск] Сбор фактов принудительно отключен в Config.ENABLE_FACT_EXTRACTION.")

                    internet_triggers = [
                        "найди в интернете", "поищи в интернете", "поиск в интернете", 
                        "найди информацию", "загугли", "поищи про", "найди про"
                    ]
                    if any(t in query.lower() for t in internet_triggers):
                        search_phrase = query
                        for trigger in internet_triggers:
                            if trigger in query.lower():
                                idx = query.lower().find(trigger)
                                search_phrase = query[idx + len(trigger):].strip(" ,:-.!?")
                                if not search_phrase:
                                    search_phrase = query[:idx].strip(" ,:-.!?")
                                break
                        
                        if search_phrase:
                            phrase_lower = search_phrase.lower()
                            prefixes_to_strip = [
                                "информацию про", "информацию о", "информация про", "информация о",
                                "данные про", "данные о", "про", "о"
                            ]
                            for prefix in prefixes_to_strip:
                                if phrase_lower.startswith(prefix + " "):
                                    search_phrase = search_phrase[len(prefix):].strip()
                                    phrase_lower = search_phrase.lower()
                            
                            self.speak(f"Секунду, ищу в интернете информацию про {search_phrase}...")
                            search_result = self.search_yandex_gen(search_phrase)
                            if search_result:
                                clean_result = search_result
                                clean_result = re.sub(r'\*+', '', clean_result)  
                                clean_result = re.sub(r'#+\s+', '', clean_result)  
                                clean_result = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean_result)  
                                clean_result = re.sub(r'^\s*[-*+]\s+', '', clean_result, flags=re.MULTILINE) 
                                
                                def replace_eng(match):
                                    word = match.group(0)
                                    return translit_eng_word_to_ru(word)
                                clean_result = re.sub(r'[a-zA-Z]+', replace_eng, clean_result)
                                
                                clean_result = convert_numbers_to_words(clean_result)
                                
                                self.speak(clean_result)
                            else:
                                self.speak("Извините, не удалось выполнить поиск. Попробуйте повторить запрос позже.")
                        else:
                            self.speak("Что именно мне найти в интернете?")
                        
                        self.is_processing = False
                        continue

                    vision_triggers = ["камер", "обстановк", "вокруг", "посмотри", "опиши", "видишь", "зрение"]
                    if any(w in query for w in vision_triggers):
                        self.speak("Секунду, анализирую изображение...")
                        try:
                            if self.latest_frame is not None:
                                frame = self.latest_frame.copy()
                                frame_resized = cv2.resize(frame, (512, 384))
                                _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
                                img_base64 = base64.b64encode(buffer).decode('utf-8')
                                
                                active_names_with_relations = []
                                try:
                                    for track in list(self.active_tracks.values()):
                                        if track.is_verified:
                                            if track.relation == "Создатель":
                                                active_names_with_relations.append(f"{track.name} (это Создатель)")
                                            else:
                                                active_names_with_relations.append(f"{track.name} (это {track.relation})")
                                except Exception:
                                    pass
                                recognized_people = ", ".join(active_names_with_relations) if active_names_with_relations else ""
                                
                                ans = self.ask_vlm(img_base64, recognized_people=recognized_people)
                                self.speak(ans)
                            else:
                                self.speak("Я пока не получаю картинку с камеры.")
                        except Exception as e:
                            print(f"[Debug] Ошибка VLM: {e}")
                            self.speak("Произошла ошибка при анализе изображения.")
                            
                        self.is_processing = False
                        continue

                    local_answer = self.handle_local_intent(query)
                    if local_answer:
                        self.speak(local_answer)
                    else:
                        try:
                            active_names = [track.name for track in list(self.active_tracks.values()) if track.is_verified]
                        except RuntimeError:
                            active_names = []
                        who = ", ".join(active_names) if active_names else "никого"
                        
                        ans = self.ask_llm(query, context_data=who, person_name=active_name)
                        self.speak(ans)
                        
                self.is_processing = False 
            except queue.Empty:
                continue
            except Exception as e:
                self.is_processing = False 
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

