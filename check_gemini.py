
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise SystemExit(
        "ERROR: GEMINI_API_KEY was not found in the root .env file."
    )

try:
    client = genai.Client(api_key=api_key)

    print("\nChecking models available to your API key...\n")

    available = []

    for model in client.models.list():
        name = getattr(model, "name", "") or ""
        actions = getattr(model, "supported_actions", None) or []

        if "generateContent" in actions:
            clean_name = name.removeprefix("models/")
            available.append(clean_name)
            print(f"Available: {clean_name}")

    if not available:
        print("\nNo text-generation models were returned.")
        print("Check your API key, project access, and API configuration.")
        raise SystemExit(1)

    # Prefer the configured model if it is actually available.
    preferred = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    selected = next(
        (m for m in available if m == preferred),
        available[0],
    )

    print(f"\nTesting model: {selected}")

    response = client.models.generate_content(
        model=selected,
        contents="Reply with exactly: Gemini connection successful"
    )

    print("\nSUCCESS")
    print(response.text)
    print(f"\nSet GEMINI_MODEL={selected} in your .env file.")

except Exception as exc:
    print(f"\nGEMINI ERROR: {type(exc).__name__}: {exc}")
    print("\nCheck the API key, model access, and API quota.")