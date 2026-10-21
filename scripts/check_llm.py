"""Quick check: does OPENAI_API_KEY work, and which models can it call? python -m scripts.check_llm"""
from app.core.config import settings

def main():
    if not settings.OPENAI_API_KEY:
        print("OPENAI_API_KEY is not set in .env"); return
    from openai import OpenAI
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    for name in (settings.REPORT_MODEL, settings.SPORT_CHECK_MODEL):
        try:
            r = client.chat.completions.create(model=name, messages=[{"role": "user", "content": "Reply with just: ok"}], max_completion_tokens=20)
            print(f"OK   {name}: {r.choices[0].message.content!r}")
        except Exception as e:
            print(f"FAIL {name}: {e}")

if __name__ == "__main__":
    main()
