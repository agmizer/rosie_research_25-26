from transformers import AutoTokenizer, AutoModelForCausalLM
import json


class BaseLLM:
    """Base class that handles model loading and text generation."""

    def __init__(self, model_name: str, system_prompt: str, max_new_tokens: int = 512):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def _build_prompt(self, user_input: str) -> str:
        """Build a chat-style prompt. Override in subclasses if needed."""
        return (
            f"### System:\n{self.system_prompt}\n\n"
            f"### User:\n{user_input}\n\n"
            f"### Assistant:\n"
        )

    def generate(self, user_input: str) -> str:
        """Tokenize, generate, and decode a response."""
        prompt = self._build_prompt(user_input)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )
        # Decode only the newly generated tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class TutorLLM(BaseLLM):
    """Generates pedagogical responses that guide without giving answers."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a patient tutor helping a student learn. "
        "Never give the final answer directly. Instead, ask guiding questions, "
        "give hints, explain underlying concepts, and walk the student through "
        "the reasoning process step by step."
    )

    def __init__(self, model_name: str, system_prompt: str | None = None, **kwargs):
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            **kwargs,
        )

    def respond(self, context_json: dict) -> str:
        """Generate a tutor response from the preprocessed context object."""
        user_input = json.dumps(context_json)
        return self.generate(user_input)


class VerifierLLM(BaseLLM):
    """Reviews a tutor response and decides whether it violates guidelines."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a response verifier for an educational tutoring system. "
        "You will receive a student query and a proposed tutor response. "
        "Determine whether the tutor response gives away the answer directly. "
        "Respond with exactly 'PASS' if the response is acceptable, or 'FAIL' "
        "followed by a brief reason if it gives away the answer."
    )

    def __init__(self, model_name: str, system_prompt: str | None = None, **kwargs):
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            **kwargs,
        )

    def verify(self, student_query: str, tutor_response: str) -> dict:
        """Return {'passed': bool, 'reason': str}."""
        user_input = (
            f"Student query: {student_query}\n\n"
            f"Tutor response: {tutor_response}"
        )
        raw = self.generate(user_input)
        passed = raw.upper().startswith("PASS")
        return {"passed": passed, "raw": raw}


class EvaluatorLLM(BaseLLM):
    """Reads a full conversation and grades the tutor on five categories."""

    CATEGORIES = [
        "avoided_giving_direct_answer",
        "asked_guiding_questions",
        "explained_underlying_concepts",
        "encouraged_student_reasoning",
        "maintained_supportive_tone",
    ]

    DEFAULT_SYSTEM_PROMPT = (
        "You are an evaluator for a tutoring chatbot. You will receive a full "
        "conversation between a student and a tutor. Grade the tutor on each of "
        "the following categories with YES or NO:\n"
        "1. avoided_giving_direct_answer\n"
        "2. asked_guiding_questions\n"
        "3. explained_underlying_concepts\n"
        "4. encouraged_student_reasoning\n"
        "5. maintained_supportive_tone\n\n"
        "Respond with exactly one JSON object mapping each category to true/false."
    )

    def __init__(self, model_name: str, system_prompt: str | None = None, **kwargs):
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            **kwargs,
        )

    def evaluate(self, conversation: list[dict]) -> dict:
        """
        Evaluate a conversation.

        Args:
            conversation: List of {"role": "student"|"tutor", "content": str} dicts.

        Returns:
            Dict mapping each category to True/False, plus the raw output.
        """
        formatted = "\n".join(
            f"{turn['role'].capitalize()}: {turn['content']}"
            for turn in conversation
        )
        raw = self.generate(formatted)

        # Try to parse the JSON from the response
        try:
            grades = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: return raw output so the caller can handle it
            grades = {cat: None for cat in self.CATEGORIES}

        return {"grades": grades, "raw": raw}