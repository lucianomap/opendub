"""Translate transcripts using LLMs."""

import json
import os
import time
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


class LLMTranslator:
    """Translate text using various LLM providers."""

    def __init__(self, provider: str | None = None):
        """Initialize translator with API provider.

        Args:
            provider: LLM provider (openai, anthropic, google, or auto-detect)
        """
        self.provider = provider or self._detect_provider()
        self.model_name = None  # Will be set in _init_client
        self.client = self._init_client()
        logger.info(f"Using {self.provider} with model {self.model_name}")

    def _detect_provider(self) -> str:
        """Auto-detect available LLM provider based on API keys."""
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        elif os.getenv("GOOGLE_API_KEY"):
            return "google"
        else:
            raise RuntimeError(
                "No LLM API keys found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY"
            )

    def _init_client(self):
        """Initialize LLM client based on provider."""
        if self.provider == "openai":
            import openai

            self.model_name = os.getenv("LLM_MODEL", "gpt-4")
            return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "anthropic":
            import anthropic

            self.model_name = os.getenv("LLM_MODEL", "claude-3-sonnet-20240229")
            return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif self.provider == "google":
            from google import genai

            # Get model from environment or use default
            self.model_name = os.getenv("LLM_MODEL", "gemini-2.0-flash")
            return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def translate(self, transcription: dict, target_language: str) -> dict:
        """Translate transcription to target language.

        Args:
            transcription: Transcription data with segments
            target_language: Target language code (e.g., 'pt', 'es')

        Returns:
            Translated transcription
        """
        source_lang = transcription.get("language", "en")

        # Map language codes to full names
        lang_names = {
            "en": "English",
            "pt": "Portuguese",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "ru": "Russian",
            "ar": "Arabic",
            "hi": "Hindi",
        }

        target_lang_name = lang_names.get(target_language, target_language)
        source_lang_name = lang_names.get(source_lang, source_lang)

        logger.info(f"Translating from {source_lang_name} to {target_lang_name}")
        logger.info(f"Processing {len(transcription['segments'])} segments in one shot")

        # Prepare all segments for translation
        segments_to_translate = []
        segment_indices = []

        for i, seg in enumerate(transcription["segments"]):
            if seg["text"].strip():
                segments_to_translate.append(f"[{i}] {seg['text']}")
                segment_indices.append(i)

        if not segments_to_translate:
            # No segments to translate
            return {
                "text": "",
                "segments": transcription["segments"],
                "language": target_language,
                "source_language": source_lang,
                "duration": transcription["duration"],
            }

        # Combine all texts for single translation
        combined_text = "\n".join(segments_to_translate)

        # Translate everything in one API call
        prompt = f"""Translate the following segments from {source_lang_name} to {target_lang_name}.
Maintain the tone, style, and meaning. Keep translations concise for video dubbing timing.
IMPORTANT: Preserve the segment markers [0], [1], etc. in your translation.
Only return the translated segments, nothing else.

Segments to translate:
{combined_text}"""

        translated_text = self._translate_text_with_retry(prompt)

        # Parse translated results
        translations = {}
        translated_lines = translated_text.strip().split("\n")

        for line in translated_lines:
            if line.strip() and "[" in line and "]" in line:
                try:
                    idx_end = line.index("]")
                    idx_str = line[1:idx_end]
                    idx = int(idx_str)
                    translation = line[idx_end + 1 :].strip()
                    translations[idx] = translation
                except (ValueError, IndexError):
                    logger.warning(f"Failed to parse line: {line}")
                    continue

        # Apply translations to segments
        translated_segments = []
        for i, seg in enumerate(transcription["segments"]):
            translated_seg = seg.copy()
            if i in translations:
                translated_seg["text"] = translations[i]
            elif not seg["text"].strip():
                # Keep empty segments as is
                pass
            else:
                # Log missing translation but keep original
                logger.warning(f"No translation found for segment {i}, keeping original")
            translated_segments.append(translated_seg)

        # Create translated transcription
        translated = {
            "text": " ".join(seg["text"] for seg in translated_segments),
            "segments": translated_segments,
            "language": target_language,
            "source_language": source_lang,
            "duration": transcription["duration"],
        }

        logger.info(f"Translated {len(segment_indices)} segments in a single API call")

        return translated

    def _translate_text_with_retry(self, prompt: str, retry_count: int = 3) -> str:
        """Execute translation with retry logic for rate limits."""

        for attempt in range(retry_count):
            try:
                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=4000,
                    )
                    return response.choices[0].message.content.strip()

                elif self.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=4000,
                        temperature=0.3,
                    )
                    return response.content[0].text.strip()

                elif self.provider == "google":
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config={
                            "temperature": 0.3,
                            "max_output_tokens": 4000,
                        },
                    )
                    if response and response.text:
                        return response.text.strip()
                    else:
                        logger.error("Google Gemini returned empty response")
                        return prompt  # Return original prompt if translation fails

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    # Rate limit or quota error - wait and retry
                    wait_time = (attempt + 1) * 10  # 10, 20, 30 seconds
                    logger.warning(
                        f"Rate limit hit, waiting {wait_time} seconds before retry {attempt + 1}/{retry_count}"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Translation failed: {e}")
                    # Return original prompt as fallback
                    return prompt

        logger.error(f"Translation failed after {retry_count} attempts")
        return prompt  # Return original prompt if all attempts fail

    def save_translation(self, translation: dict, output_path: Path, language: str) -> Path:
        """Save translation to JSON file."""
        json_path = output_path / f"translation_{language}.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(translation, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved translation to {json_path.name}")
        return json_path
