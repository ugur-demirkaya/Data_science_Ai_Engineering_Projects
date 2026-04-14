import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(_ENV_PATH)


def get_groq_advice(prediction_label, confidence, api_key=None):
    from groq import Groq

    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.startswith("gsk-your"):
        return "Groq API anahtarı bulunamadı. Lütfen .env dosyasını kontrol edin."

    client = Groq(api_key=api_key)
    prompt = _build_prompt(prediction_label, confidence)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Groq API hatası: {str(e)}"


def get_gemini_advice(prediction_label, confidence, api_key=None):
    import google.generativeai as genai

    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.startswith("your-"):
        return "Gemini API anahtarı bulunamadı. Lütfen .env dosyasını kontrol edin."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = _build_prompt(prediction_label, confidence)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API hatası: {str(e)}"


def get_treatment_advice(prediction_label, confidence, provider="groq", api_key=None):
    if provider == "gemini":
        result = get_gemini_advice(prediction_label, confidence, api_key)
        if "API hatası" in result:
            return get_groq_advice(prediction_label, confidence)
        return result
    else:
        result = get_groq_advice(prediction_label, confidence, api_key)
        if "API hatası" in result:
            return get_gemini_advice(prediction_label, confidence)
        return result


def _build_prompt(prediction_label, confidence):
    if isinstance(prediction_label, (int, float)):
        turkish_label = "Tümör VAR" if prediction_label >= 0.5 else "Tümör YOK"
    else:
        turkish_label = "Tümör VAR" if "VAR" in str(prediction_label) else "Tümör YOK"
    return f"""Sen bir nöroloji uzmanısın. Bir beyin MRI görüntüsünden AI modeli aşağıdaki sonucu tespit etti:

- Tahmin: {turkish_label}
- Güven Oranı: %{confidence * 100:.1f}

Lütfen aşağıdaki başlıklar altında bilgi ver:
1. Bu Sonucun Anlamı
2. Önerilen Tıbbi Prosedürler
3. Olası Tedavi Yöntemleri
4. Önemli Uyarılar
5. Sonraki Adımlar

Yanıtını Türkçe ver ve hastaya yönelik anlaşılır bir dil kullan.

UYARI: Bu bir AI tahminidir, kesin tanı değildir. Her zaman bir nörolog/onkolog ile görüşülmelidir."""
