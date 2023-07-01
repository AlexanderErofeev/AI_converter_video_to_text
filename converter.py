import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

import os
import whisper
import re
import datetime
import sys
from settings import *


def print_log(log_string, is_error=False):
    text = f"[{datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}] {log_string}"
    print(text, file=sys.stderr if is_error else sys.stdout)


def convert(video_name, model):
    print_log(f"Обработка: {video_name}")

    result = model.transcribe(f"{VIDEO_DIR}\\{video_name}", language='ru', fp16=False)
    processed_text = re.sub(r"([.!?]) ", r"\1\n", result['text'][1:])
    new_name = video_name.replace(VIDEO_FORMAT, TEXT_FORMAT)

    original_file = open(f"{ORIGINAL_TEXT_DIR}\\{new_name}", "w+", encoding='utf8')
    original_file.write(result['text'])
    original_file.close()

    result_file = open(f"{RESULT_TEXT_DIR}\\{new_name}", "w+", encoding='utf8')
    result_file.write(processed_text)
    result_file.close()


def main():
    video_list = [f for f in os.listdir(VIDEO_DIR) if f.endswith(VIDEO_FORMAT)]

    print_log(f"Загрузка модели: {MODEL_NAME}")
    model = whisper.load_model(MODEL_NAME, download_root=MODELS_DIR, device=DEVICE)

    [convert(video_name, model) for video_name in video_list]


if __name__ == '__main__':
    main()
