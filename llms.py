import json
import re
from openai import OpenAI
from context import TutorContext, VerifierContext

class BaseLLM:
    """Base class that handles model calls via OpenAI-compatible API."""

    def __init__(self, model_name: str, system_prompt: str, max_new_tokens: int = 200):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens

        self.client = OpenAI(
            base_url="http://dh-dgxh100-2.hpc.msoe.edu:8001/v1",
            api_key="unused"
        )

    def generate(self, LLM_input: str, system_prompt: str | None = None) -> str:
        """Send request to the chat completion API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt or self.system_prompt},
                {"role": "user", "content": LLM_input},
            ],
            max_tokens=self.max_new_tokens,
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()


class TutorLLM(BaseLLM):
    """Generates pedagogical responses that guide without giving answers."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a patient tutor helping a student learn. "
        "Do not do the student's work for them — instead, guide them "
        "through the reasoning so they reach the answer themselves. "
        "Only address the specific thing the student asked about. "
        "After responding, ask the student to take the next step themselves. "
        "Never jump ahead to later steps or the final answer on their behalf. "
        "If the subject involves mathematical expressions, format them "
        "using LaTeX notation with $ delimiters for inline math and $$ for display math."
    )

    def __init__(self, model_name: str, system_prompt: str | None = None, **kwargs):
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            **kwargs,
        )

    def respond(self, context: TutorContext) -> str:
        """Generate a tutor response from the preprocessed context object."""
        full_system_prompt = self.system_prompt + context.to_system_prompt()
        return self.generate(context.student_query, system_prompt=full_system_prompt)


class VerifierLLM(BaseLLM):
    """Reviews a tutor response and decides whether it violates guidelines."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a response verifier for an educational tutoring system. "
        "You will receive a student query and a proposed tutor response. "
        "Your job is to determine whether the tutor does the student's work "
        "for them before the student has attempted it themselves.\n\n"
        "IMPORTANT: Use the conversation history to judge whether the student "
        "has already worked through the problem. If the conversation shows the "
        "student arrived at a conclusion through their own effort over previous "
        "turns and is now presenting their result for confirmation, the tutor "
        "confirming or denying that result is ACCEPTABLE and should PASS.\n\n"
        "Explaining general definitions, formulas, or concepts that the student "
        "needs in order to approach the problem is ACCEPTABLE and should PASS. "
        "The student's work is applying those concepts to their specific problem. "
        "A FAIL means the tutor provided a solution, answer, or analysis the "
        "student had not yet attempted or worked toward. A PASS means the tutor "
        "guided the student or appropriately responded to the student's own work.\n\n"
        "Respond with exactly 'PASS' if the response is acceptable, or 'FAIL' "
        "followed by a brief reason."
    )

    def __init__(self, model_name: str, system_prompt: str | None = None, **kwargs):
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            **kwargs,
        )

    def verify(self, context: VerifierContext) -> dict:
        """Return {'passed': bool, 'reason': str}."""
        full_system_prompt = self.system_prompt + context.to_system_prompt()
        raw = self.generate(context.tutor_response, system_prompt=full_system_prompt)
        passed = "PASS" in raw.upper()
        return {"passed": passed, "reason": raw}


class EvaluatorLLM(BaseLLM):
    """Reads a full conversation and grades the tutor on five categories."""

    CATEGORIES = [
        "avoided_doing_the_students_work",
        "asked_guiding_questions",
        "explained_underlying_concepts",
        "encouraged_student_reasoning",
        "maintained_supportive_tone",
    ]

    DEFAULT_SYSTEM_PROMPT = (
        "You are an evaluator for a tutoring chatbot. You will receive a full "
        "conversation between a student and a tutor. Grade the tutor on each of "
        "the following categories with YES or NO:\n"
        "1. avoided_doing_the_students_work — the tutor did not produce the "
        "student's deliverable (solution, essay, analysis, etc.) for them\n"
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
            # LLM may wrap JSON in markdown fences or extra text; extract it
            match = re.search(r'\{[^{}]*\}', raw)
            if match:
                try:
                    grades = json.loads(match.group())
                except json.JSONDecodeError:
                    grades = {cat: None for cat in self.CATEGORIES}
            else:
                grades = {cat: None for cat in self.CATEGORIES}

        return {"grades": grades, "raw": raw}