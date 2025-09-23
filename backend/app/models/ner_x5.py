import os
import io
from pathlib import Path
from typing import BinaryIO, Literal

import numpy as np
import soundfile as sf
from pydub import AudioSegment
from faster_whisper import WhisperModel

from app.settings import settings
from app.utils.logging import get_logger
from app.models.base_model import AudioTranscribeModel


class WhisperService(AudioTranscribeModel):

    def __init__(
        self,
        model_name: str = "large-v3",
        cache_dir: str | None = None,
        log_file: str | None = None,
        device: str = "cpu",
        compute_type: Literal[
            "default",
            "int8",
            "int8_float16",
            "int8_float32",
            "float16",
            "float32"
        ] = "int8",
        chunk_size: int = 30,
        **kwargs
    ):
        super().__init__(model_name=model_name, log_file=log_file)

        self.cache_dir = cache_dir or os.path.join(settings.CACHE_DIR, ".cache", "whisper")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.device = device
        self.compute_type = compute_type
        self.chunk_size = chunk_size
        self.model: WhisperModel = None

        self.logger.info(f"Инициализация FasterWhisperService с моделью: {model_name}")
        self.load_model()

    def load_model(self):
        """
        Загружает модель Whisper в cache_dir.
        """
        try:
            self.logger.info(f"Загрузка модели {self.model_name} в {self.cache_dir}")
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.cache_dir
            )
            self.logger.info("Модель успешно загружена")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели: {e}")
            raise

    def preprocess_audio(self, file: str | Path | BinaryIO, audio_format: str = None) -> list[io.BytesIO]:

        try:
            if isinstance(file, (str, Path)):
                path = Path(file)
                fmt = path.suffix.lstrip(".").lower() or "wav"
                audio = AudioSegment.from_file(path, format=fmt)
            elif isinstance(file, (io.BytesIO, BinaryIO)):
                file.seek(0)
                audio = AudioSegment.from_file(file, format=audio_format)
            else:
                raise ValueError("Unsupported file type")

            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

            chunk_ms = self.chunk_size * 1000
            chunks = [audio[i:i + chunk_ms] for i in range(0, len(audio), chunk_ms)]

            buffers: list[io.BytesIO] = []
            for chunk in chunks:
                buf = io.BytesIO()
                chunk.export(buf, format="wav")
                buf.seek(0)
                buffers.append(buf)

            return buffers
        except Exception as e:
            self.logger.error(f"Ошибка при обработке аудио {file}: {e}")
            raise


    def generate(
        self, 
        file: str | BinaryIO,
        audio_format: str = None,
        language: str = None,
        task: str = "transcribe", 
        **kwargs
    ) -> str:

        try:
            buffers = self.preprocess_audio(file, audio_format)
            all_text: list[str] = []

            for i, buf in enumerate(buffers, start=1):
                self.logger.info(f"Обработка чанка {i}/{len(buffers)}")
                segments, info = self.model.transcribe(
                    buf,
                    language=language,
                    task=task,
                    **kwargs
                )
                text = "".join(seg.text for seg in segments)
                all_text.append(text.strip())

            result_text = " ".join(all_text)

            self.logger.info(
                f"Транскрибация завершена. Язык: {info.language} "
                f"(prob: {info.language_probability:.2f})"
            )

            return result_text
        except Exception as e:
            self.logger.error(f"Ошибка транскрибации {file}: {e}")
            raise