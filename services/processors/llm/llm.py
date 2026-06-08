"""
Text Generation Processor

Uses Ollama for LLM-based text generation.
"""

import os
import time
import logging
import math
import random
import json
import threading
import requests
import re

from services.processors.base import BaseJobProcessor

MAX_LLM_OUTPUT_TOKENS = int(os.getenv("OPENFORK_LLM_MAX_TOKENS", "24576"))


class LLMJobProcessor(BaseJobProcessor):
    """Processor for LLM-based text generation using Ollama."""

    @staticmethod
    def _base_json_schema(schema_type: str) -> dict:
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": schema_type,
        }

    @classmethod
    def _normalize_allowed_values(cls, value) -> list[str]:
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, list):
            values = value
        else:
            return []

        normalized = []
        seen = set()
        for item in values:
            text = str(item or "").strip()
            key = " ".join(text.casefold().split())
            if text and key not in seen:
                normalized.append(text)
                seen.add(key)
        return normalized

    @staticmethod
    def _text_key(value) -> str:
        return " ".join(str(value or "").strip().casefold().split())

    @classmethod
    def _allowed_speakers_for_inputs(cls, inputs: dict, allowed_characters: list[str]) -> list[str]:
        allowed_speakers = cls._normalize_allowed_values(
            inputs.get("allowed_speakers") or inputs.get("allowed_dialogue_speakers")
        )
        if allowed_speakers:
            return allowed_speakers

        if not allowed_characters:
            return []

        speakers = list(allowed_characters)
        if inputs.get("allow_narrator", True) is not False:
            speakers.append("Narrator")
        return speakers

    @classmethod
    def _forbidden_visual_rule_fragment(cls, text: str):
        lowered = text.casefold()
        forbidden_fragments = (
            "no visible writing",
            "do not render",
            "do not include",
            "do not put",
            "never print",
            "no text",
            "without text",
            "no labels",
            "no signage",
            "no logos",
            "no watermark",
            "visible text",
            "visible signage",
            "captions",
            "subtitles",
            "title cards",
            "name tags",
            "speech bubbles",
            "floating words",
            "diagram text",
            "annotation lines",
            "ui/hud",
            "watermarks",
            "printed words",
            "readable symbols",
            "metadata only",
        )
        for fragment in forbidden_fragments:
            if fragment in lowered:
                return fragment
        return None

    @classmethod
    def _sanitize_visual_story_text(cls, text: str) -> str:
        """Remove copied prompt-rule clauses while keeping positive story detail."""
        value = cls._sanitize_rendered_text_bait(str(text or "").strip())
        value = cls._strip_prompt_instruction_artifacts(value)
        if not value or not cls._forbidden_visual_rule_fragment(value):
            return value

        negative_rule_terms = (
            r"(?:visible )?(?:text|writing|signage|labels|logos|watermarks?|captions|subtitles)"
            r"|people|persons|faces|bodies|humanoids?|silhouettes?|characters?|performers?"
            r"|actors?|crowds?|portraits?|human elements?"
        )
        negative_rule_list = (
            rf"(?:{negative_rule_terms})"
            rf"(?:\s*,\s*(?:and\s+|or\s+)?(?:{negative_rule_terms})"
            rf"|\s+(?:or|and)\s+(?:{negative_rule_terms}))*"
        )
        replacements = (
            rf"\bwithout\s+{negative_rule_list}\b",
            rf"\bno\s+{negative_rule_list}\b",
            r"\bnever print[^.;,]*",
            r"\bdo not (?:render|include|put)[^.;,]*",
            r"\bmetadata only\b",
            r"\bno readable symbols\b",
            r"\breadable symbols\b",
            r"\bprinted words\b",
            r"\b(?:captions|subtitles|title cards|name tags|speech bubbles|floating words|diagram text|annotation lines|watermarks?|visible text|visible signage)\b",
            r"\bui/hud\b",
        )
        cleaned = value
        for pattern in replacements:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*:\s*", ": ", cleaned)
        cleaned = re.sub(r"\s+([,.;])", r"\1", cleaned)
        cleaned = re.sub(r"\b(?:or|and)\s*([,.;])", r"\1", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"([,.;]){2,}", r"\1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ;,.")
        if cleaned and not cls._forbidden_visual_rule_fragment(cleaned):
            return cleaned

        split_value = re.sub(r"\bDr\.\s+", "Dr ", value)
        parts = re.split(r"([.;]\s+|\n+)", split_value)
        cleaned_parts = []
        for index in range(0, len(parts), 2):
            clause = parts[index].strip()
            separator = parts[index + 1] if index + 1 < len(parts) else ""
            if not clause:
                continue
            if cls._forbidden_visual_rule_fragment(clause):
                continue
            cleaned_parts.append(clause + separator)

        cleaned = "".join(cleaned_parts).strip()
        if cleaned:
            return re.sub(r"\s+", " ", cleaned).strip(" ;,.")

        return cleaned or value

    @classmethod
    def _strip_prompt_instruction_artifacts(cls, text: str) -> str:
        """Remove trailing instruction fragments that local models sometimes copy into fields."""
        value = str(text or "").strip()
        if not value:
            return value

        patterns = (
            r"\s*(?:[,.;:-]\s*)?\b(?:under|around|about|approximately|approx\.?|max(?:imum)?|no more than)?\s*\d+\s+words?\b\.?\s*$",
            r"\s*(?:[,.;:-]\s*)?\b\d+\s+word\s+(?:prompt|description|limit)\b\.?\s*$",
            r"\s*(?:[,.;:-]\s*)?\b(?:under|around|about|approximately|approx\.?|max(?:imum)?|no more than)?\s*\d+\s+(?:tokens?|characters?)\b\.?\s*$",
        )
        previous = None
        while previous != value:
            previous = value
            for pattern in patterns:
                value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip()

        value = re.sub(r"\s+([,.;])", r"\1", value)
        value = re.sub(r"([,.;]){2,}", r"\1", value)
        return value.strip(" ;,.")

    @classmethod
    def _repair_style_abbreviations(cls, text: str) -> str:
        """Expand clipped style words that weaken downstream image prompts."""
        value = str(text or "").strip()
        if not value:
            return value

        replacements = (
            (r"\bvolum\.\s*", "volumetric "),
            (r"\bp\.\s+(?=soap bubbles\b)", "pearly "),
        )
        for pattern, replacement in replacements:
            value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", value).strip()

    @classmethod
    def _sanitize_rendered_text_bait(cls, text: str) -> str:
        """Replace common visual phrases that make image models render text."""
        value = str(text or "").strip()
        if not value:
            return value

        negative_rule_terms = (
            r"(?:(?:visible|readable)\s+)?(?:text|writing|signage|labels|logos|watermarks?|captions|subtitles)"
            r"|people|persons|faces|bodies|humanoids?|silhouettes?|characters?|performers?"
            r"|actors?|crowds?|portraits?|human elements?"
        )
        negative_rule_list = (
            rf"(?:{negative_rule_terms})"
            rf"(?:\s*,\s*(?:and\s+|or\s+)?(?:{negative_rule_terms})"
            rf"|\s+(?:or|and)\s+(?:{negative_rule_terms}))*"
        )
        replacements = (
            (
                rf"\b(?:no|without)\s+{negative_rule_list}\b",
                "",
            ),
            (
                r"\b(?:expired\s+)?coupons?\s+(?:with|showing|covered\s+in|bearing)?\s*(?:readable\s*)?(?:markings?|words?|letters?|numbers?|text|labels?|stamps?|signatures?)\b",
                "blank voucher-shaped color slips",
            ),
            (r"\bfloating\s+coupons?\b", "floating blank voucher-shaped slips"),
            (r"\bexpired\s+coupons?\b", "flickering blank voucher-shaped slips"),
            (
                r"\b(?:loyalty|access|identity|membership)\s+cards?\b",
                "plain glowing token card",
            ),
            (
                r"\b(?:writing|writes|write|scribbling|scribbles)\s+(?:a\s+|the\s+)?(?:safety\s+)?(?:report|form|document|note|letter|label|signature)[^.!?;]*",
                "making quick hand gestures over a plain unmarked pad",
            ),
            (
                r"\b(?:report|document|ticket|badge)s?\b|\b(?:application|registration|paperwork)\s+forms?\b",
                "plain color slip",
            ),
            (r"\b(?:signature|signatures)\b", "abstract color seal"),
            (r"\b(?:stamp|stamps|stamped)\b", "round color seal"),
            (r"\b(?:notification|notifications)\b", "glowing reaction cue"),
            (r"\b(?:screen|display|interface|ui|hud)\b", "glowing control surface"),
            (
                r"\b(?:logo|watermark|caption|subtitle|title card|speech bubble|floating words|signage)\b",
                "plain visual accent",
            ),
            (
                r"\b(?:readable|written|printed|lettered|numbered|labeled|captioned|subtitled)\s+",
                "",
            ),
        )
        cleaned = value
        for pattern, replacement in replacements:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(?:with|and|or)\s*$", "", cleaned, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", cleaned).strip()

    @classmethod
    def _forbidden_rendered_text_cue(cls, text: str):
        """Detect positive visual requests likely to create rendered writing."""
        value = str(text or "").casefold()
        patterns = (
            (
                r"\b(?:readable|written|writing|handwriting|printed|lettered|numbered|labeled|captioned|subtitled|typography|signature|watermark|logo)\b",
                "visible writing cue",
            ),
            (
                r"\b(?:screen|display|interface|ui/hud|ui|hud|notification|caption|subtitle|title card|speech bubble|floating words|diagram text|annotation|signage|signs?)\b",
                "visible UI/text surface cue",
            ),
            (
                r"\b(?:coupon|card|ticket|badge|form|report|document|paper|slip|sign|stamp)[^.!?]{0,50}\b(?:words?|letters?|numbers?|text|labels?|signatures?|markings?|glyphs?)\b",
                "marked document or prop cue",
            ),
        )
        for pattern, label in patterns:
            if re.search(pattern, value, flags=re.IGNORECASE):
                return label
        return None

    @classmethod
    def _is_generic_scene_title(cls, name: str) -> bool:
        normalized = " ".join(str(name or "").strip().casefold().split())
        if not normalized:
            return True
        if normalized.startswith(("scene ", "beat ", "hook", "setup", "payoff")):
            return True
        return bool(re.fullmatch(r"(?:scene|shot|beat|moment)\s*[-#:]?\s*\d+", normalized))

    @classmethod
    def _scene_title_from_content(cls, scene: dict, index: int, seen_names: set) -> str:
        source = str(scene.get("description") or scene.get("image_prompt") or "").strip()
        source = cls._sanitize_visual_story_text(source)
        source = re.sub(r"\bDr\.\s+", "Dr ", source)
        words = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", source)
        stop_words = {
            "the",
            "and",
            "with",
            "into",
            "from",
            "that",
            "this",
            "their",
            "while",
            "scene",
            "shot",
            "camera",
            "close",
            "wide",
            "medium",
            "open",
            "opens",
            "cold",
            "tiny",
        }
        title_words = []
        for word in words:
            if len(title_words) >= 5:
                break
            normalized = word.strip("'").casefold()
            if len(normalized) < 3 or normalized in stop_words:
                continue
            title_words.append(word.strip("'"))

        if len(title_words) < 2:
            title_words = ["Story", "Moment", str(index + 1)]

        base_title = " ".join(word[:1].upper() + word[1:] for word in title_words[:5])
        base_title = re.sub(r"\s+", " ", base_title).strip()
        if cls._is_generic_scene_title(base_title):
            base_title = f"Story Moment {index + 1}"

        title = base_title
        suffix = 2
        while " ".join(title.casefold().split()) in seen_names:
            title = f"{base_title} {suffix}"
            suffix += 1
        return title

    @classmethod
    def _script_scene_schema(cls, num_scenes, allowed_characters=None, allowed_speakers=None) -> dict:
        character_item_schema = {"type": "string"}
        if allowed_characters:
            character_item_schema["enum"] = allowed_characters

        speaker_schema = {"type": "string"}
        if allowed_speakers:
            speaker_schema["enum"] = allowed_speakers

        schema = {
            **cls._base_json_schema("array"),
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Literal 2-5 word story moment title, not a generic Scene/Beat label",
                    },
                    "description": {"type": "string", "description": "Visual story action"},
                    "characters": {
                        "type": "array",
                        "items": character_item_schema,
                        "minItems": 1,
                        "description": "Exact established cast or documentary subject names visible in the scene; never locations or labels",
                    },
                    "dialogue": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "speaker": speaker_schema,
                                "line": {
                                    "type": "string",
                                    "description": "Short character line under 18 words",
                                },
                            },
                            "required": ["speaker", "line"],
                            "additionalProperties": False,
                        },
                        "minItems": 0,
                        "maxItems": 2,
                        "description": "Zero, one, or two short dialogue/narration lines. Use [] for silent scenes. Use exact cast names for character dialogue or Narrator for voiceover.",
                    },
                    "image_prompt": {
                        "type": "string",
                        "description": "Production-ready first-frame image prompt with subject, action, setting, lighting, and identity anchors",
                    },
                    "aspect_ratio": {"type": "string", "enum": ["16:9", "9:16"]},
                    "camera_movement": {
                        "type": "string",
                        "enum": ["static", "zoom-in", "zoom-out", "pan-left", "pan-right", "dolly-in", "push-in", "pull-back"],
                    },
                    "duration_seconds": {"type": "integer", "minimum": 4, "maximum": 10},
                },
                "required": [
                    "name",
                    "description",
                    "characters",
                    "dialogue",
                    "image_prompt",
                    "aspect_ratio",
                    "camera_movement",
                    "duration_seconds",
                ],
                "additionalProperties": False,
            },
        }

        if num_scenes is not None:
            schema["minItems"] = num_scenes
            schema["maxItems"] = num_scenes
            logging.info(f"Enforcing exactly {num_scenes} scene(s) via JSON schema constraints")

        return schema

    @classmethod
    def _generation_style_schema(cls) -> dict:
        return {
            **cls._base_json_schema("object"),
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "prompt_prefix": {
                    "type": "string",
                    "description": "A compact reusable visual style prefix for image and video prompts. Use complete words, no abbreviations or shorthand.",
                },
                "category": {
                    "type": "string",
                    "enum": ["animation", "cinematic", "artistic", "photo", "stylized", "custom"],
                },
            },
            "required": ["name", "description", "prompt_prefix", "category"],
            "additionalProperties": False,
        }

    @classmethod
    def _activity_scenarios_schema(cls, expected_items, allowed_characters=None) -> dict:
        character_name_schema = {"type": "string"}
        if allowed_characters:
            character_name_schema["enum"] = allowed_characters

        scenarios = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "character_name": character_name_schema,
                    "activity": {"type": "string"},
                    "scene_prompt": {"type": "string"},
                    "image_edit_prompt": {"type": "string"},
                    "video_prompt": {"type": "string"},
                    "sound_fx_prompt": {
                        "type": "string",
                        "description": "Non-dialogue text-to-audio prompt for ambience and physical sound effects only; no vocals, lyrics, or spoken words.",
                    },
                    "camera_movement": {
                        "type": "string",
                        "enum": ["static", "zoom-in", "zoom-out", "pan-left", "pan-right", "dolly-in", "push-in", "pull-back"],
                    },
                    "duration_seconds": {"type": "integer", "minimum": 4, "maximum": 10},
                },
                "required": [
                    "character_name",
                    "activity",
                    "scene_prompt",
                    "image_edit_prompt",
                    "video_prompt",
                    "sound_fx_prompt",
                    "camera_movement",
                    "duration_seconds",
                ],
                "additionalProperties": False,
            },
        }

        if expected_items is not None:
            scenarios["minItems"] = expected_items
            scenarios["maxItems"] = expected_items

        return {
            **cls._base_json_schema("object"),
            "properties": {
                "scenarios": scenarios,
            },
            "required": ["scenarios"],
            "additionalProperties": False,
        }

    @classmethod
    def _character_profiles_schema(cls, expected_items) -> dict:
        characters = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Proper character/object name, never a story beat label",
                    },
                    "description": {
                        "type": "string",
                        "description": "Reusable physical identity and story role; never workflow rules, reference-image instructions, or rendering constraints",
                    },
                    "visual_prompt": {
                        "type": "string",
                        "description": "Concrete reusable visual identity anchors for image generation. For people, include face, silhouette, wardrobe, colors, accessories, and permanent signature props. For non-human subjects, include shape, scale, color, surface texture, and physically attached identity parts. Never include environment, scene action, workflow rules, transparent-background instructions, no-text rules, forbidden-change lists, or reference-image format instructions.",
                    },
                    "personality": {"type": "string"},
                    "voice_profile": {
                        "type": "string",
                        "description": "Reusable TTS identity: age, pitch, pace, rhythm, energy, and emotional default",
                    },
                    "voice_direction": {
                        "type": "string",
                        "description": "Line delivery guidance that can be reused for TTS jobs",
                    },
                    "qwen3_speaker": {
                        "type": "string",
                        "enum": [
                            "Vivian",
                            "Serena",
                            "Uncle_Fu",
                            "Dylan",
                            "Eric",
                            "Ryan",
                            "Aiden",
                            "Ono_Anna",
                            "Sohee",
                        ],
                    },
                },
                "required": [
                    "name",
                    "description",
                    "visual_prompt",
                    "personality",
                    "voice_profile",
                    "voice_direction",
                    "qwen3_speaker",
                ],
                "additionalProperties": False,
            },
            "minItems": 1,
        }

        if expected_items is not None:
            characters["minItems"] = expected_items
            characters["maxItems"] = expected_items

        return {
            **cls._base_json_schema("object"),
            "properties": {
                "characters": characters,
            },
            "required": ["characters"],
            "additionalProperties": False,
        }

    @classmethod
    def _music_prompt_schema(cls) -> dict:
        return {
            **cls._base_json_schema("object"),
            "properties": {
                "prompt": {"type": "string"},
                "mood": {"type": "string"},
                "tempo": {"type": "string"},
                "instruments": {"type": "array", "items": {"type": "string"}},
                "duration_seconds": {"type": "integer", "minimum": 5, "maximum": 300},
            },
            "required": ["prompt", "mood", "tempo", "instruments", "duration_seconds"],
            "additionalProperties": False,
        }

    @classmethod
    def _dialogue_scenes_schema(cls, expected_items, allowed_speakers=None) -> dict:
        speaker_schema = {"type": "string"}
        if allowed_speakers:
            speaker_schema["enum"] = allowed_speakers

        dialogues = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "script_index": {"type": "integer", "minimum": 0},
                    "speaker": speaker_schema,
                    "line": {"type": "string"},
                    "qwen3_speaker": {
                        "type": "string",
                        "enum": [
                            "Vivian",
                            "Serena",
                            "Uncle_Fu",
                            "Dylan",
                            "Eric",
                            "Ryan",
                            "Aiden",
                            "Ono_Anna",
                            "Sohee",
                        ],
                    },
                    "start_offset_seconds": {"type": "number", "minimum": 0, "maximum": 20},
                },
                "required": [
                    "script_index",
                    "speaker",
                    "line",
                    "qwen3_speaker",
                    "start_offset_seconds",
                ],
                "additionalProperties": False,
            },
        }

        if expected_items is not None:
            dialogues["minItems"] = expected_items
            dialogues["maxItems"] = expected_items

        return {
            **cls._base_json_schema("object"),
            "properties": {
                "dialogues": dialogues,
            },
            "required": ["dialogues"],
            "additionalProperties": False,
        }

    @classmethod
    def _generic_object_schema(cls) -> dict:
        return {
            **cls._base_json_schema("object"),
            "additionalProperties": True,
        }

    @classmethod
    def _json_schema_for_job(cls, job_type: str, num_scenes, expected_items, inputs=None) -> dict:
        inputs = inputs or {}
        allowed_characters = cls._normalize_allowed_values(
            inputs.get("allowed_characters") or inputs.get("allowed_character_names")
        )
        allowed_speakers = cls._allowed_speakers_for_inputs(inputs, allowed_characters)

        if job_type == "script_generation":
            return cls._script_scene_schema(num_scenes, allowed_characters, allowed_speakers)
        if job_type == "generation_style":
            return cls._generation_style_schema()
        if job_type == "activity_scenarios":
            return cls._activity_scenarios_schema(expected_items, allowed_characters)
        if job_type == "character_profiles":
            return cls._character_profiles_schema(expected_items)
        if job_type == "dialogue_scenes":
            return cls._dialogue_scenes_schema(expected_items, allowed_speakers)
        if job_type == "music_prompt":
            return cls._music_prompt_schema()
        return cls._generic_object_schema()

    @classmethod
    def _validate_generated_json(
        cls,
        generated_text: str,
        job_type: str,
        num_scenes,
        expected_items,
        allowed_characters=None,
        allowed_speakers=None,
    ):
        parsed_json = json.loads(generated_text)
        allowed_character_keys = {
            cls._text_key(value): value for value in (allowed_characters or [])
        }
        allowed_speaker_keys = {
            cls._text_key(value): value for value in (allowed_speakers or [])
        }

        if job_type == "generation_style":
            if not isinstance(parsed_json, dict):
                raise ValueError(f"Generated JSON is not an object. Got: {type(parsed_json).__name__}")
            for field_name in ("name", "description", "prompt_prefix"):
                field_text = str(parsed_json.get(field_name) or "")
                cleaned_text = cls._repair_style_abbreviations(
                    cls._sanitize_visual_story_text(field_text)
                )
                if cleaned_text != field_text:
                    logging.info(
                        "Sanitized style %s from %r to %r.",
                        field_name,
                        field_text,
                        cleaned_text,
                    )
                    parsed_json[field_name] = cleaned_text
                if not cleaned_text:
                    raise ValueError(f"Style {field_name} is empty after cleanup.")
            return parsed_json

        if job_type == "script_generation":
            if not isinstance(parsed_json, list):
                raise ValueError(f"Generated JSON is not an array. Got: {type(parsed_json).__name__}")
            actual_count = len(parsed_json)
            if num_scenes is not None and actual_count != num_scenes:
                raise ValueError(f"Expected exactly {num_scenes} scenes but got {actual_count}.")
            seen_scene_names = set()
            forbidden_title_terms = ("credit", "black screen", "cutscene")
            for index, scene in enumerate(parsed_json):
                if not isinstance(scene, dict):
                    raise ValueError(f"Scene {index + 1} is not an object.")
                name = str(scene.get("name") or "").strip()
                if cls._is_generic_scene_title(name):
                    name = cls._scene_title_from_content(scene, index, seen_scene_names)
                    scene["name"] = name
                    logging.info(
                        "Repaired generic script scene title for scene %s to %r.",
                        index + 1,
                        name,
                    )
                if cls._is_generic_scene_title(name):
                    raise ValueError(f"Scene {index + 1} has a generic title: {name!r}.")
                if len(name.split()) > 7 or len(name) > 70:
                    name = cls._scene_title_from_content(scene, index, seen_scene_names)
                    scene["name"] = name
                    logging.info(
                        "Repaired long script scene title for scene %s to %r.",
                        index + 1,
                        name,
                    )
                normalized_name = " ".join(name.lower().split())
                if normalized_name in seen_scene_names:
                    original_name = name
                    name = cls._scene_title_from_content(scene, index, seen_scene_names)
                    scene["name"] = name
                    normalized_name = " ".join(name.lower().split())
                    logging.info(
                        "Repaired duplicate script scene title for scene %s from %r to %r.",
                        index + 1,
                        original_name,
                        name,
                    )
                seen_scene_names.add(normalized_name)
                if any(term in normalized_name for term in forbidden_title_terms):
                    raise ValueError(f"Scene {index + 1} is not a visual story moment: {name!r}.")
                characters = scene.get("characters")
                if not isinstance(characters, list) or not characters:
                    raise ValueError(f"Scene {index + 1} is missing a non-empty characters array.")
                if allowed_character_keys:
                    for character in characters:
                        character_name = str(character or "").strip()
                        if cls._text_key(character_name) not in allowed_character_keys:
                            raise ValueError(
                                f"Scene {index + 1} uses unknown character {character_name!r}; "
                                f"allowed characters are: {', '.join(allowed_characters)}."
                            )
                description_text = str(scene.get("description") or "")
                image_prompt_text = str(scene.get("image_prompt") or "")
                if "black screen" in description_text.casefold() or "black screen" in image_prompt_text.casefold():
                    raise ValueError(f"Scene {index + 1} uses a black screen instead of renderable imagery.")
                for field_name, field_text in (
                    ("description", description_text),
                    ("image_prompt", image_prompt_text),
                ):
                    cleaned_text = cls._sanitize_visual_story_text(field_text)
                    if cleaned_text != field_text:
                        logging.info(
                            "Sanitized copied visual-rule text from scene %s %s.",
                            index + 1,
                            field_name,
                        )
                        scene[field_name] = cleaned_text
                    if not cleaned_text:
                        raise ValueError(f"Scene {index + 1} {field_name} is empty after cleanup.")

                description_text = str(scene.get("description") or "")
                image_prompt_text = str(scene.get("image_prompt") or "")
                for field_name, field_text, max_length in (
                    ("description", description_text, 520),
                    ("image_prompt", image_prompt_text, 1100),
                ):
                    if len(field_text) > max_length:
                        raise ValueError(
                            f"Scene {index + 1} {field_name} is too long ({len(field_text)} chars); "
                            "use concise positive visual direction."
                        )
                    forbidden_fragment = cls._forbidden_visual_rule_fragment(field_text)
                    if forbidden_fragment:
                        raise ValueError(
                            f"Scene {index + 1} {field_name} repeats a visual rule phrase "
                            f"{forbidden_fragment!r}; describe only visible story content."
                        )
                    forbidden_text_cue = cls._forbidden_rendered_text_cue(field_text)
                    if forbidden_text_cue:
                        raise ValueError(
                            f"Scene {index + 1} {field_name} still contains {forbidden_text_cue}; "
                            "use blank shapes, glowing objects, physical reactions, or abstract unreadable color seals."
                        )
                dialogue = scene.get("dialogue")
                if not isinstance(dialogue, list):
                    raise ValueError(f"Scene {index + 1} is missing a dialogue array.")
                for line_index, line in enumerate(dialogue):
                    if not isinstance(line, dict):
                        raise ValueError(f"Scene {index + 1} dialogue {line_index + 1} is not an object.")
                    speaker = str(line.get("speaker") or "").strip()
                    dialogue_line = str(line.get("line") or "").strip()
                    if not speaker or not dialogue_line:
                        raise ValueError(f"Scene {index + 1} dialogue {line_index + 1} is missing speaker or line.")
                    if allowed_speaker_keys and cls._text_key(speaker) not in allowed_speaker_keys:
                        raise ValueError(
                            f"Scene {index + 1} dialogue {line_index + 1} uses unknown speaker {speaker!r}; "
                            f"allowed speakers are: {', '.join(allowed_speakers)}."
                        )
            return parsed_json

        if job_type == "activity_scenarios":
            if not isinstance(parsed_json, dict):
                raise ValueError(f"Generated JSON is not an object. Got: {type(parsed_json).__name__}")
            scenarios = parsed_json.get("scenarios")
            if not isinstance(scenarios, list):
                raise ValueError("Generated activity scenario output is missing a scenarios array.")
            if expected_items is not None and len(scenarios) != expected_items:
                raise ValueError(f"Expected exactly {expected_items} scenarios but got {len(scenarios)}.")
            for index, scenario in enumerate(scenarios):
                if not isinstance(scenario, dict):
                    raise ValueError(f"Scenario {index + 1} is not an object.")
                character_name = str(scenario.get("character_name") or "").strip()
                if allowed_character_keys and cls._text_key(character_name) not in allowed_character_keys:
                    raise ValueError(
                        f"Scenario {index + 1} uses unknown character {character_name!r}; "
                        f"allowed characters are: {', '.join(allowed_characters)}."
                    )
                for field_name in ("scene_prompt", "image_edit_prompt", "video_prompt"):
                    field_text = str(scenario.get(field_name) or "")
                    cleaned_text = cls._sanitize_visual_story_text(field_text)
                    if cleaned_text != field_text:
                        logging.info(
                            "Sanitized copied visual-rule text from scenario %s %s.",
                            index + 1,
                            field_name,
                        )
                        scenario[field_name] = cleaned_text
                        field_text = cleaned_text
                    if not field_text:
                        raise ValueError(f"Scenario {index + 1} {field_name} is empty after cleanup.")
                    forbidden_fragment = cls._forbidden_visual_rule_fragment(field_text)
                    if forbidden_fragment:
                        raise ValueError(
                            f"Scenario {index + 1} {field_name} repeats a visual rule phrase "
                            f"{forbidden_fragment!r}; describe only visible story content."
                        )
                    forbidden_text_cue = cls._forbidden_rendered_text_cue(field_text)
                    if forbidden_text_cue:
                        raise ValueError(
                            f"Scenario {index + 1} {field_name} still contains {forbidden_text_cue}; "
                            "use blank shapes, glowing objects, physical reactions, or abstract unreadable color seals."
                        )
            return parsed_json

        if job_type == "character_profiles":
            if not isinstance(parsed_json, dict):
                raise ValueError(f"Generated JSON is not an object. Got: {type(parsed_json).__name__}")
            characters = parsed_json.get("characters")
            if not isinstance(characters, list):
                raise ValueError("Generated character profile output is missing a characters array.")
            if not characters:
                raise ValueError("Generated character profile output is empty.")
            if expected_items is not None and len(characters) != expected_items:
                raise ValueError(f"Expected exactly {expected_items} characters but got {len(characters)}.")
            return parsed_json

        if job_type == "dialogue_scenes":
            if not isinstance(parsed_json, dict):
                raise ValueError(f"Generated JSON is not an object. Got: {type(parsed_json).__name__}")
            dialogues = parsed_json.get("dialogues")
            if not isinstance(dialogues, list):
                raise ValueError("Generated dialogue output is missing a dialogues array.")
            if expected_items is not None and len(dialogues) != expected_items:
                raise ValueError(f"Expected exactly {expected_items} dialogue lines but got {len(dialogues)}.")
            for index, dialogue in enumerate(dialogues):
                if not isinstance(dialogue, dict):
                    raise ValueError(f"Dialogue {index + 1} is not an object.")
                speaker = str(dialogue.get("speaker") or "").strip()
                if allowed_speaker_keys and cls._text_key(speaker) not in allowed_speaker_keys:
                    raise ValueError(
                        f"Dialogue {index + 1} uses unknown speaker {speaker!r}; "
                        f"allowed speakers are: {', '.join(allowed_speakers)}."
                    )
            return parsed_json

        if not isinstance(parsed_json, dict):
            raise ValueError(f"Generated JSON is not an object. Got: {type(parsed_json).__name__}")

        return parsed_json

    @staticmethod
    def _is_transient_ollama_error(exc: Exception) -> bool:
        text = str(exc).lower()
        transient_markers = (
            "remote end closed connection",
            "connection aborted",
            "connection reset",
            "connection refused",
            "temporarily unavailable",
            "timeout",
            "timed out",
        )
        return any(marker in text for marker in transient_markers)

    @staticmethod
    def _positive_int(value, default=None):
        try:
            parsed = int(value)
            return parsed if parsed > 0 else default
        except (TypeError, ValueError):
            return default

    @classmethod
    def _context_limit_for_job(cls, inputs: dict) -> int:
        """Default to an 8GB-safe long-form context window unless the caller opts up."""
        override = (
            cls._positive_int(inputs.get("max_context_tokens"))
            or cls._positive_int(inputs.get("num_ctx"))
            or cls._positive_int(os.getenv("OPENFORK_LLM_MAX_CONTEXT_TOKENS"))
        )
        if override:
            return max(4096, min(32768, override))
        return 12288

    def process(self):
        if not self.job:
            self._fail_job(f"Job object is None for LLMJobProcessor. Cannot proceed.")
            return

        api_base = os.getenv("LLM_API_BASE", "http://127.0.0.1:11434")

        logging.info(f"Waiting for Ollama service to be ready at {api_base}...")

        if not self._wait_for_ollama(api_base):
            return

        inputs = self.job.get("inputs", {})
        # Default comes from the worker startup script, which selects Qwen3 4B
        # for lower VRAM and Gemma 4 12B for 16GB+ clients.
        default_model = os.getenv(
            "OPENFORK_LLM_MODEL",
            os.getenv("OPENFORK_LLM_BASE_MODEL", "qwen3:4b"),
        )
        model_name = inputs.get("model") or default_model
            
        system_prompt = inputs.get("system_prompt", "You are a helpful assistant.")
        temperature = inputs.get("temperature", 0.7)
        max_tokens = inputs.get("max_tokens", 16000)

        seed = inputs.get("seed")
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
            logging.info(f"No seed provided, using random seed: {seed}")

        if not self._ensure_model_available(api_base, model_name):
            return

        self._generate_text(api_base, model_name, system_prompt, temperature, max_tokens, seed)

    def _wait_for_ollama(self, api_base: str, timeout: int = 60) -> bool:
        """Wait for Ollama service to be ready."""
        for attempt in range(timeout):
            if self.is_cancelled():
                logging.info("Shutdown event received during Ollama readiness check.")
                return False
            try:
                response = requests.get(f"{api_base}/api/tags", timeout=2)
                response.raise_for_status()
                logging.info(f"Ollama service is ready! (attempt {attempt + 1}/{timeout})")
                return True
            except requests.exceptions.RequestException as e:
                if attempt % 10 == 0:
                    logging.debug(f"Ollama not ready yet (attempt {attempt + 1}/{timeout}): {e}")
                time.sleep(0.5)

        logging.error(f"Ollama service failed to become ready within {timeout} seconds at {api_base}")
        self._fail_job("Ollama service not ready")
        return False

    def _ensure_model_available(self, api_base: str, model_name: str) -> bool:
        """Check if model exists and pull if needed."""
        try:
            logging.info(f"Checking if model {model_name} is available...")
            tags_response = requests.get(f"{api_base}/api/tags", timeout=5)
            tags_response.raise_for_status()
            tags_data = tags_response.json()

            available_models = [m.get("name", "") for m in tags_data.get("models", [])]
            logging.info(f"Available models: {available_models}")

            if model_name not in available_models:
                logging.info(f"Model {model_name} not found. Pulling it now...")
                pull_payload = {"name": model_name, "stream": False}
                pull_response = requests.post(f"{api_base}/api/pull", json=pull_payload, timeout=600)
                pull_response.raise_for_status()
                logging.info(f"Successfully pulled model {model_name}")
            else:
                logging.info(f"Model {model_name} is already available")

            return True
        except Exception as e:
            self._fail_job(f"Failed to verify/pull model {model_name}: {e}")
            return False

    def _generate_text(
        self, api_base: str, model_name: str, system_prompt: str, temperature: float, max_tokens: int, seed: int
    ):
        """Generate text using Ollama with structured output for reliable JSON."""
        try:
            inputs = self.job.get("inputs", {})
            num_scenes = inputs.get("num_scenes")
            job_type = inputs.get("job_type") or ("script_generation" if num_scenes is not None else "general")
            expected_items = inputs.get("expected_items")
            allowed_character_names = self._normalize_allowed_values(
                inputs.get("allowed_characters") or inputs.get("allowed_character_names")
            )
            allowed_dialogue_speakers = self._allowed_speakers_for_inputs(
                inputs,
                allowed_character_names,
            )
            json_schema = self._json_schema_for_job(job_type, num_scenes, expected_items, inputs)
            context_limit = self._context_limit_for_job(inputs)

            # Dynamically size the context window to fit the requested output.
            # num_ctx must be >= (input tokens + max_tokens).  We add a 2048-token
            # buffer for the system/user prompts and round up to the next power-of-two
            # for efficiency in the attention implementation.
            raw_ctx = max_tokens + 2048
            num_ctx = min(
                context_limit,
                max(4096, int(2 ** math.ceil(math.log2(raw_ctx)))),
            )
            logging.info(
                f"Context window sizing: max_tokens={max_tokens}, num_ctx={num_ctx}, "
                f"context_limit={context_limit}, job_type={job_type}, "
                f"num_scenes={num_scenes}, expected_items={expected_items}"
            )

            logging.info(f"Calling Ollama chat API with structured output at {api_base}/api/chat")
            response = None
            parsed_json = None
            generated_text = ""
            validation_error = None
            content_attempts = 3

            for content_attempt in range(1, content_attempts + 1):
                retry_max_tokens = int(max_tokens * 1.15) + 512
                attempt_max_tokens = min(
                    MAX_LLM_OUTPUT_TOKENS,
                    max_tokens
                    if content_attempt == 1
                    else max(retry_max_tokens, max_tokens + 768),
                )
                attempt_num_ctx = min(
                    context_limit,
                    max(
                        4096,
                        int(2 ** math.ceil(math.log2(attempt_max_tokens + 2048))),
                    ),
                )
                max_predict_for_context = max(512, attempt_num_ctx - 2048)
                if attempt_max_tokens > max_predict_for_context:
                    logging.info(
                        "Capping LLM max output tokens from %s to %s for "
                        "num_ctx=%s.",
                        attempt_max_tokens,
                        max_predict_for_context,
                        attempt_num_ctx,
                    )
                    attempt_max_tokens = max_predict_for_context
                attempt_temperature = temperature if content_attempt == 1 else min(temperature, 0.35)
                attempt_seed = seed if content_attempt == 1 else seed + content_attempt
                retry_guard = ""

                if content_attempt > 1:
                    validation_feedback = (
                        f" Validation error to fix: {validation_error}."
                        if validation_error
                        else ""
                    )
                    story_guard = ""
                    if job_type == "script_generation":
                        story_guard = (
                            " Every scene must be a concrete visual story beat that can be rendered as image/video. "
                            "Every JSON field must describe positive story content only; do not write rule phrases "
                            "about text, logos, captions, watermarks, credits, title cards, black screens, or "
                            "meta-production moments."
                        )
                        if allowed_character_names:
                            story_guard += (
                                " Use only these exact character names in characters arrays: "
                                f"{', '.join(allowed_character_names)}. "
                                "Use only those names or Narrator as dialogue speakers."
                            )
                    elif job_type in {"activity_scenarios", "dialogue_scenes"} and allowed_character_names:
                        story_guard = (
                            " Use only these exact established character names where a character or speaker is required: "
                            f"{', '.join(allowed_character_names)}."
                        )
                    retry_guard = (
                        "\n\nThe previous output failed JSON validation. Return complete valid JSON only, "
                        "with no markdown, no commentary, no trailing prose, and every string, array, and object closed. "
                        "Keep field values concise enough to fit the token budget."
                        f"{validation_feedback}{story_guard}"
                    )

                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt + retry_guard},
                        {"role": "user", "content": self.positive_prompt + retry_guard},
                    ],
                    "stream": False,
                    "think": False,
                    "format": json_schema,
                    "options": {
                        "temperature": attempt_temperature,
                        "num_predict": attempt_max_tokens,
                        "seed": attempt_seed,
                        "num_ctx": attempt_num_ctx,
                    },
                }

                response = None
                max_attempts = 4
                for attempt in range(1, max_attempts + 1):
                    response_holder: list = [None]
                    error_holder: list = [None]

                    def _do_request():
                        try:
                            response_holder[0] = requests.post(
                                f"{api_base}/api/chat", json=payload, timeout=1200
                            )
                        except Exception as exc:
                            error_holder[0] = exc

                    req_thread = threading.Thread(target=_do_request, daemon=True)
                    req_thread.start()
                    while req_thread.is_alive():
                        req_thread.join(timeout=1.0)
                        if self.is_cancelled():
                            logging.info("Shutdown event received during LLM generation. Aborting.")
                            return

                    if error_holder[0]:
                        if (
                            attempt < max_attempts
                            and self._is_transient_ollama_error(error_holder[0])
                        ):
                            delay = min(30, 5 * attempt)
                            logging.warning(
                                "Ollama chat request failed during startup/model load "
                                "(attempt %s/%s): %s. Retrying in %ss.",
                                attempt,
                                max_attempts,
                                error_holder[0],
                                delay,
                            )
                            self.shutdown_event.wait(delay)
                            if self.is_cancelled():
                                logging.info("Shutdown event received during LLM retry delay. Aborting.")
                                return
                            continue
                        raise error_holder[0]

                    response = response_holder[0]
                    try:
                        response.raise_for_status()
                        break
                    except requests.exceptions.RequestException as exc:
                        if attempt < max_attempts and self._is_transient_ollama_error(exc):
                            delay = min(30, 5 * attempt)
                            logging.warning(
                                "Ollama chat returned a transient error "
                                "(attempt %s/%s): %s. Retrying in %ss.",
                                attempt,
                                max_attempts,
                                exc,
                                delay,
                            )
                            self.shutdown_event.wait(delay)
                            if self.is_cancelled():
                                logging.info("Shutdown event received during LLM retry delay. Aborting.")
                                return
                            continue
                        raise

                result = response.json()
                generated_text = result.get("message", {}).get("content", "")

                if not generated_text:
                    self._fail_job(f"Ollama returned empty response. Full result: {result}")
                    return

                try:
                    parsed_json = self._validate_generated_json(
                        generated_text,
                        job_type,
                        num_scenes,
                        expected_items,
                        allowed_character_names,
                        allowed_dialogue_speakers,
                    )
                    generated_text = json.dumps(parsed_json, ensure_ascii=False, indent=2)
                    validation_error = None
                    break
                except (json.JSONDecodeError, ValueError) as exc:
                    validation_error = exc
                    preview = generated_text[:500].replace("\n", " ")
                    logging.warning(
                        "Generated content failed validation on attempt %s/%s for %s: %s. "
                        "done_reason=%s, length=%s, preview=%s",
                        content_attempt,
                        content_attempts,
                        job_type,
                        exc,
                        result.get("done_reason"),
                        len(generated_text),
                        preview,
                    )
                    if content_attempt >= content_attempts:
                        break

            if parsed_json is None:
                self._fail_job(f"Generated content is not valid JSON: {validation_error}")
                return

            if isinstance(parsed_json, list):
                logging.info(
                    f"Generation complete. Generated {len(parsed_json)} item(s). Length: {len(generated_text)} chars."
                )
            elif job_type == "activity_scenarios":
                logging.info(
                    "Generation complete. Generated %s scenario(s). Length: %s chars.",
                    len(parsed_json.get("scenarios", [])),
                    len(generated_text),
                )
            else:
                logging.info(
                    f"Generation complete for {job_type}. Length: {len(generated_text)} chars."
                )

            output_filename = f"{self.job_id}_{job_type}.txt"
            output_path = os.path.join(self.cache_dir, output_filename)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(generated_text)

            storage_path = self.orchestrator_service.upload_output(output_path, self.job_id, "text/plain")

            if self.is_cancelled():
                logging.info("Shutdown event set after LLM generation completed. Skipping job completion.")
                return

            if storage_path:
                completion_metadata = self.job.get("completion_metadata") or {}
                completion_metadata = dict(completion_metadata)
                completion_metadata.update({"model": model_name, "job_type": job_type})
                self.orchestrator_service.update_job_status(
                    self.job_id,
                    "completed",
                    storage_path=storage_path,
                    completion_metadata=completion_metadata,
                )
            else:
                self._fail_job("Failed to upload generated script.")

            if os.path.exists(output_path):
                os.remove(output_path)

        except Exception as e:
            self._fail_job(f"Text generation failed: {e}")
