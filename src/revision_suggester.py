"""
=============================================================
AI Requirement Revision Suggester
=============================================================

Uses Ollama + Qwen2.5 to rewrite software requirements
based on predicted requirement smells and SHAP keywords.

Author:
    Muhammad Arsalan
=============================================================
"""

from ollama import chat
from predict_requirement import get_predicted_smells
from explainability import get_shap_keywords
# ==========================================================
# Configuration
# ==========================================================

MODEL_NAME = "qwen2.5:7b"

# ==========================================================
# Revision Function
# ==========================================================

def suggest_revision(requirement, predicted_smells, shap_keywords):
    """
    Generate an improved version of a requirement.

    Parameters
    ----------
    requirement : str
        Original requirement.

    predicted_smells : list
        Predicted requirement smell labels.

    shap_keywords : list
        Important words identified by SHAP.

    Returns
    -------
    str
        Improved requirement.
    """

    smell_text = ", ".join(predicted_smells)

    keyword_text = ", ".join(shap_keywords)

    prompt = f"""
You are a senior Software Requirements Engineer.

Your task is to rewrite software requirements according to the ISO/IEC/IEEE 29148 standard.

Original Requirement:
{requirement}

Detected Requirement Smells:
{smell_text}

Important Words:
{keyword_text}

Requirements for rewriting:

- Preserve the original meaning.
- Remove ambiguity.
- Remove subjectivity.
- Make the requirement measurable whenever possible.
- Use precise and professional language.
- Use "shall" where appropriate.
- Keep the requirement concise.
- Do NOT ask questions.
- Do NOT write in ALL CAPITAL LETTERS.
- Do NOT add explanations.
- Do NOT use bullet points.
- Return ONLY one rewritten requirement sentence.

Improved Requirement:
"""

    try:

        response = chat(
            model=MODEL_NAME,
            options={
                "temperature": 0.2
            },
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response["message"]["content"].strip()

    except Exception as e:

        return f"Error: {e}"


# ==========================================================
# Example
# ==========================================================

def main():

    requirement = input(
        "\nEnter Requirement:\n\n"
    )

    predicted_smells = get_predicted_smells(
        requirement
    )

    # If no smells are detected, no revision is needed
    if not predicted_smells:

        print("=" * 60)
        print("Original Requirement")
        print("=" * 60)
        print(requirement)

        print("\n" + "=" * 60)
        print("Detected Smells")
        print("=" * 60)
        print("No requirement smells detected.")

        print("\nNo revision is required because the requirement is already well-formed.")

        return

    all_keywords = get_shap_keywords(
        requirement
    )

    shap_keywords = []

    for smell in predicted_smells:

        shap_keywords.extend(
            all_keywords.get(smell, [])
        )

    # Remove duplicates while preserving order
    shap_keywords = list(
        dict.fromkeys(shap_keywords)
    )

    revised_requirement = suggest_revision(
        requirement,
        predicted_smells,
        shap_keywords,
    )

    print("=" * 60)
    print("Original Requirement")
    print("=" * 60)
    print(requirement)

    print("\n" + "=" * 60)
    print("Detected Smells")
    print("=" * 60)

    if predicted_smells:
        for smell in predicted_smells:
            print(f"✓ {smell}")
    else:
        print("No requirement smells detected.")

    print("\n" + "=" * 60)
    print("SHAP Keywords")
    print("=" * 60)

    if shap_keywords:
        print(", ".join(shap_keywords))
    else:
        print("No important keywords identified.")

    print("\n" + "=" * 60)
    print("Improved Requirement")
    print("=" * 60)
    print(revised_requirement)


if __name__ == "__main__":
    main()