import os
import json
import sqlite3
from datetime import datetime, UTC
from typing import Dict, Any, Tuple
import unicodedata
import re
import html
import requests

from flask import Flask, render_template, request, jsonify, g, redirect, url_for

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

try:
    from detoxify import Detoxify
except Exception:  # Lazy import fallback at runtime if not available at build time
    Detoxify = None  # type: ignore

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # make language detection deterministic
except Exception:
    detect = None  # type: ignore

# Optional translators
try:
    from google.cloud import translate_v2 as gcloud_translate  # Official API
except Exception:
    gcloud_translate = None  # type: ignore

try:
    from googletrans import Translator as GoogleFreeTranslator  # Unofficial free fallback
except Exception:
    GoogleFreeTranslator = None  # type: ignore

try:
    from transformers import MarianMTModel, MarianTokenizer
except Exception:  # optional; will raise only if translation is attempted
    MarianMTModel = None  # type: ignore
    MarianTokenizer = None  # type: ignore


DATABASE_PATH = os.path.join(os.path.dirname(__file__), "app.db")


def create_app() -> Flask:
    app = Flask(__name__)

    # ---------- Database helpers ----------
    def get_db() -> sqlite3.Connection:
        db = getattr(g, "_database", None)
        if db is None:
            db = g._database = sqlite3.connect(DATABASE_PATH)
            db.row_factory = sqlite3.Row
        return db

    def init_db() -> None:
        conn = sqlite3.connect(DATABASE_PATH)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    language TEXT,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    scores_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    @app.teardown_appcontext
    def close_connection(exception: Exception | None) -> None:  # noqa: ARG001
        db = getattr(g, "_database", None)
        if db is not None:
            db.close()

    # ---------- Model loading ----------
    app.model = None  # type: ignore[attr-defined]

    def load_model_if_needed() -> None:
        if getattr(app, "model", None) is None:
            if Detoxify is None:
                raise RuntimeError(
                    "Detoxify is not installed. Please install dependencies from requirements.txt"
                )
            app.model = Detoxify("multilingual")  # type: ignore[attr-defined]

    # ---------- Translation loading ----------
    # Supported Indic→English models via Helsinki-NLP MarianMT
    INDIC_TO_EN_MODELS: Dict[str, str] = {
        "hi": "Helsinki-NLP/opus-mt-hi-en",
        "bn": "Helsinki-NLP/opus-mt-bn-en",
        "ta": "Helsinki-NLP/opus-mt-ta-en",
        "te": "Helsinki-NLP/opus-mt-te-en",
        "ml": "Helsinki-NLP/opus-mt-ml-en",
        "kn": "Helsinki-NLP/opus-mt-kn-en",
        "mr": "Helsinki-NLP/opus-mt-mr-en",
        "gu": "Helsinki-NLP/opus-mt-gu-en",
        "pa": "Helsinki-NLP/opus-mt-pa-en",
        "ur": "Helsinki-NLP/opus-mt-ur-en",
    }
    app.translation_models: Dict[str, Tuple[MarianMTModel, MarianTokenizer]] = {}  # type: ignore[attr-defined]

    # Cache translators
    app._gcloud_translate_client = None  # type: ignore[attr-defined]
    app._google_free_translator = None  # type: ignore[attr-defined]

    def get_gcloud_client():
        if gcloud_translate is None:
            return None
        if getattr(app, "_gcloud_translate_client", None) is None:
            app._gcloud_translate_client = gcloud_translate.Client()  # type: ignore[attr-defined]
        return app._gcloud_translate_client

    def get_google_free_translator():
        global GoogleFreeTranslator
        if GoogleFreeTranslator is None:
            # Try lazy import
            try:
                from googletrans import Translator as _GFT  # type: ignore
                GoogleFreeTranslator = _GFT  # type: ignore
            except Exception:
                return None
        if getattr(app, "_google_free_translator", None) is None:
            try:
                app._google_free_translator = GoogleFreeTranslator()  # type: ignore[attr-defined]
            except Exception:
                return None
        return app._google_free_translator

    def get_translator(lang_code: str) -> Tuple[MarianMTModel, MarianTokenizer]:  # type: ignore[valid-type]
        if lang_code in app.translation_models:  # type: ignore[attr-defined]
            return app.translation_models[lang_code]  # type: ignore[attr-defined]
        if MarianMTModel is None or MarianTokenizer is None:
            raise RuntimeError("Translation dependencies not available")
        model_name = INDIC_TO_EN_MODELS[lang_code]
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        app.translation_models[lang_code] = (model, tokenizer)  # type: ignore[attr-defined]
        return model, tokenizer

    def translate_to_english(text: str, src_lang: str) -> str | None:
        if not text or src_lang == "en":
            return None
        # 0) Google Cloud API key (REST) if provided
        api_key = (
            os.environ.get("GOOGLE_CLOUD_TRANSLATE_API_KEY")
            or os.environ.get("GOOGLE_TRANSLATE_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if api_key:
            try:
                resp = requests.get(
                    "https://translation.googleapis.com/language/translate/v2",
                    params={
                        "key": api_key,
                        "q": text,
                        "source": src_lang,
                        "target": "en",
                        "format": "text",
                    },
                    timeout=12,
                )
                if resp.ok:
                    data = resp.json()
                    translations = data.get("data", {}).get("translations", [])
                    if translations:
                        translated = html.unescape(translations[0].get("translatedText", "")).strip()
                        if translated:
                            return translated
            except Exception:
                pass
        # 1) Google Cloud Translation (ADC service account)
        use_cloud = os.environ.get("GOOGLE_TRANSLATE_USE_CLOUD", "0") in {"1", "true", "True"}
        if use_cloud and gcloud_translate is not None:
            try:
                client = get_gcloud_client()
                if client is not None:
                    result = client.translate(text, source_language=src_lang, target_language="en")
                    translated = html.unescape(result.get("translatedText", "")).strip()
                    if translated:
                        return translated
            except Exception:
                pass
        # 2) Unofficial googletrans fallback
        try:
            free_t = get_google_free_translator()
            if free_t is not None:
                res = free_t.translate(text, src=src_lang, dest="en")
                translated = (res.text or "").strip()
                if translated:
                    return translated
        except Exception:
            pass
        # 3) deep-translator fallback
        try:
            from deep_translator import GoogleTranslator as DeepGT  # type: ignore
            translated = DeepGT(source=src_lang, target="en").translate(text)
            if translated:
                return str(translated).strip()
        except Exception:
            pass
        # 4) MarianMT fallback for Indic languages
        if src_lang in INDIC_TO_EN_MODELS:
            try:
                model, tokenizer = get_translator(src_lang)
                batch = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
                gen = model.generate(**batch, max_length=512, num_beams=4)
                out = tokenizer.batch_decode(gen, skip_special_tokens=True)
                return out[0] if out else None
            except Exception:
                return None
        return None


    def detect_language(text: str) -> str | None:
        if not text.strip():
            return None
        # Script heuristic for Telugu (Unicode range 0C00–0C7F)
        if re.search(r"[\u0C00-\u0C7F]", text):
            return "te"
        try:
            if detect is None:
                return None
            return detect(text)
        except Exception:
            return None

    def infer_scores(text: str) -> Dict[str, float]:
        load_model_if_needed()
        assert app.model is not None  # for type checker
        normalized_text = unicodedata.normalize("NFC", text)
        scores: Dict[str, float] = app.model.predict(normalized_text)  # type: ignore[attr-defined]
        return {k: float(v) for k, v in scores.items()}

    def decide_label(scores: Dict[str, float], language: str | None) -> tuple[str, float]:
        toxicity = scores.get("toxicity", 0.0)
        identity_attack = scores.get("identity_attack", 0.0)
        insult = scores.get("insult", 0.0)
        threat = scores.get("threat", 0.0)
        obscene = scores.get("obscene", 0.0)
        severe = scores.get("severe_toxicity", 0.0)
        sexual_explicit = scores.get("sexual_explicit", 0.0)

        # Core scores
        hate_core = max(
            identity_attack * 1.2,
            threat * 1.2,
            severe * 1.0,
            toxicity * 0.8,
        )
        abuse_core = max(
            obscene * 1.1,
            sexual_explicit * 1.1,
            insult * 0.7,
            toxicity * 0.6,
        )

        # Language-specific thresholds
        hate_threshold_by_lang: Dict[str, float] = {
            "en": 0.50, "es": 0.50, "it": 0.50, "pt": 0.50, "ru": 0.50, "tr": 0.50,
            # Indic
            "hi": 0.45, "bn": 0.45, "ta": 0.45, "te": 0.40, "ml": 0.45, "kn": 0.45, "mr": 0.45, "gu": 0.45, "pa": 0.45, "ur": 0.45,
        }
        abuse_threshold_by_lang: Dict[str, float] = {
            "en": 0.35, "es": 0.35, "it": 0.35, "pt": 0.35, "ru": 0.35, "tr": 0.35,
            # Slightly lower for Indic to catch obscene phrases
            "hi": 0.25, "bn": 0.25, "ta": 0.25, "te": 0.18, "ml": 0.25, "kn": 0.25, "mr": 0.25, "gu": 0.25, "pa": 0.25, "ur": 0.25,
        }
        hate_th = hate_threshold_by_lang.get(language or "", 0.50)
        abuse_th = abuse_threshold_by_lang.get(language or "", 0.35)

        if hate_core >= hate_th:
            label = "HATE"
            confidence = hate_core
        elif abuse_core >= abuse_th:
            label = "ABUSIVE"
            confidence = abuse_core
        else:
            label = "NOT_HATE"
            confidence = 1.0 - max(hate_core, abuse_core)

        return label, float(confidence)

    def label_severity_rank(label: str) -> int:
        return {"NOT_HATE": 0, "ABUSIVE": 1, "HATE": 2}.get(label, 0)

    # ---------- Routes ----------
    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    @app.route("/history")
    def history() -> str:
        db = get_db()
        rows = db.execute(
            "SELECT id, text, language, label, confidence, scores_json, created_at "
            "FROM analyses ORDER BY id DESC LIMIT 50"
        ).fetchall()
        entries = [
            {
                "id": r["id"],
                "text": r["text"],
                "language": r["language"],
                "label": r["label"],
                "confidence": r["confidence"],
                "scores": json.loads(r["scores_json"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]
        return render_template("history.html", entries=entries)

    @app.route("/api/analyses", methods=["GET"])
    def api_analyses() -> Any:
        db = get_db()
        rows = db.execute(
            "SELECT id, text, language, label, confidence, scores_json, created_at "
            "FROM analyses ORDER BY id DESC LIMIT 100"
        ).fetchall()
        data = [
            {
                "id": r["id"],
                "text": r["text"],
                "language": r["language"],
                "label": r["label"],
                "confidence": r["confidence"],
                "scores": json.loads(r["scores_json"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]
        return jsonify(data)

    @app.route("/analyze", methods=["POST"]) 
    def analyze() -> Any:
        payload_text = request.form.get("text") or (request.json or {}).get("text")  # type: ignore[union-attr]
        if not payload_text or not payload_text.strip():
            return jsonify({"error": "No text provided."}), 400

        text = payload_text.strip()
        try:
            language = detect_language(text)
            orig_scores = infer_scores(text)
            orig_label, orig_conf = decide_label(orig_scores, language)

            translation_used = False
            translated_text = None
            translated_scores = None
            translated_label = None
            translated_conf = None

            if language and language != "en":
                translated_text = translate_to_english(text, language)
                if translated_text:
                    translated_scores = infer_scores(translated_text)
                    translated_label, translated_conf = decide_label(translated_scores, "en")
                    translation_used = True

            # Choose stricter outcome (HATE > ABUSIVE > NOT_HATE); on tie, pick higher confidence
            best_label = orig_label
            best_conf = orig_conf
            best_scores = orig_scores
            used_translation = False

            if translated_label is not None:
                if label_severity_rank(translated_label) > label_severity_rank(orig_label) or (
                    label_severity_rank(translated_label) == label_severity_rank(orig_label)
                    and translated_conf is not None and translated_conf > orig_conf
                ):
                    best_label = translated_label
                    best_conf = float(translated_conf or 0.0)
                    best_scores = translated_scores or best_scores
                    used_translation = True

            db = get_db()
            db.execute(
                "INSERT INTO analyses (text, language, label, confidence, scores_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    text,
                    language,
                    best_label,
                    float(best_conf),
                    json.dumps(best_scores),
                    datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
                ),
            )
            db.commit()

            response_payload: Dict[str, Any] = {
                "text": text,
                "language": language,
                "label": best_label,
                "confidence": float(best_conf),
                "scores": best_scores,
                "translation_used": used_translation,
            }
            if translation_used:
                response_payload.update(
                    {
                        "translated_text": translated_text,
                        "translated_scores": translated_scores,
                        "translated_label": translated_label,
                        "translated_confidence": translated_conf,
                    }
                )

            return jsonify(response_payload)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Initialize DB on startup
    with app.app_context():
        init_db()

    return app


app = create_app()

if __name__ == "__main__":
    # Run development server
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 3000)), debug=True) 