import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any
from src.logger import logger
import hashlib
import json

load_dotenv()

_client = None
_cache = {}

MODEL_FAST = "gpt-5-nano"


# CLIENT
def _get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY missing")
        _client = OpenAI(api_key=key)
    return _client


def _cache_key(data: dict) -> str:
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


# PATIENT RECOMMENDATION
def generate_patient_recommendation(
    risk_score: float,
    top_factors: List[Dict[str, Any]],
    patient_info: Dict[str, Any],
) -> str:

    try:
        risk_level = (
            "HIGH" if risk_score > 0.6
            else "MEDIUM" if risk_score > 0.35
            else "LOW"
        )

        factors = "\n".join(
            f"{f['feature']}={f['patient_value']:.1f} "
            f"(impact {f['shap_value']:+.2f})"
            for f in top_factors[:5]  # token control
        )

        payload = {
            "risk": risk_score,
            "factors": factors,
            "patient": patient_info,
        }

        key = _cache_key(payload)
        if key in _cache:
            logger.info("LLM cache hit")
            return _cache[key]

        prompt = f"""
Patient Risk Analysis

Risk Score: {risk_score:.1%} ({risk_level})
Age: {patient_info.get('age')}
Medications: {patient_info.get('num_medications')}
Hospital Days: {patient_info.get('time_in_hospital')}

Top Factors:
{factors}

Return markdown:

### Risk Summary
(one sentence)

### Key Drivers
(bullets)

### Recommended Actions
(2-3 practical interventions)
"""

        client = _get_client()

        response = client.responses.create(
            model=MODEL_FAST,
            input=prompt,
            temperature=0.2,
            max_output_tokens=250,
        )

        result = response.output_text.strip()

        _cache[key] = result
        return result

    except Exception as e:
        logger.error(str(e))
        return f"LLM unavailable: {str(e)}"


# CHATBOT
def chat_with_data(
    messages: List[Dict[str, str]],
    context_stats: Dict[str, Any],
) -> str:

    try:
        ctx = context_stats

        # Trim history
        messages = messages[-6:]

        context = f"""
Dataset: {ctx.get('dataset')}
Target: {ctx.get('target')}
Model: {ctx.get('model')}
AUC: {ctx.get('auc_roc')}
Test Size: {ctx.get('test_set_size')}
High Risk: {ctx.get('high_risk_pct')}%
Avg Risk: {ctx.get('avg_readmission_probability')}%
Top Features: {", ".join(ctx.get("top_10_features_by_importance", []))}
"""

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are a clinical ML analyst. "
                    "Answer ONLY using provided dataset facts. "
                    "Never invent statistics. "
                    "Be concise and use markdown."
                ),
            },
            {"role": "system", "content": context},
            *messages,
        ]

        client = _get_client()

        response = client.responses.create(
            model=MODEL_FAST,
            input=prompt_messages,
            temperature=0.3,
            max_output_tokens=300,
        )

        return response.output_text.strip()

    except Exception as e:
        logger.error(str(e))
        return f"⚠️ Chat unavailable: {str(e)}"