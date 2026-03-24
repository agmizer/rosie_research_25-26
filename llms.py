import json
import re
from openai import OpenAI
from context import TutorContext, VerifierContext

VALID_TEACHING_MODES = {"EXPLAIN", "GUIDE", "CONFIRM", "REDIRECT"}

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
        "You will receive a student query, a teaching mode, and a proposed "
        "tutor response.\n\n"
        "FAIL only if the tutor produced the complete, assembled deliverable "
        "that the student is supposed to create themselves — the finished "
        "paragraph, the worked solution, the written response. Everything "
        "else is a PASS.\n\n"
        "IMPORTANT: Evaluate against the CURRENT student message and teaching "
        "mode, not the original task from earlier in the conversation. The "
        "teaching mode defines what the tutor should be doing right now. "
        "Follow the teaching mode guidance closely.\n\n"
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


class ClassifierLLM(BaseLLM):
    """Classifies student queries into a teaching mode that tells the tutor how to respond."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a teaching mode classifier for an educational tutoring system. "
        "Given a student's message and optionally the conversation history, "
        "decide which teaching strategy the tutor should use. Classify into "
        "exactly one of the following modes.\n\n"
        "Modes:\n\n"
        "EXPLAIN — The student wants to understand a concept, definition, or idea, "
        "or is confused about something and needs it explained or re-explained. "
        "The tutor should teach directly with clear explanations and examples. "
        "If the student is confused, try a different angle or analogy.\n"
        "Examples:\n"
        '  - "What is a gradient conceptually?"\n'
        '  - "I don\'t understand what you mean by the chain rule"\n\n'
        "GUIDE — The student is working through a problem and needs help moving "
        "forward. The tutor should ask guiding questions and give hints, not "
        "solve the problem for them.\n"
        "Examples:\n"
        '  - "Write a paragraph explaining the role of supply and demand in pricing"\n'
        '  - "Can you help me with this problem: what is the gradient of z = 5y^4 + 3x^2 + 2y?"\n\n'
        "CONFIRM — The student has done work and is presenting it to check if "
        "it is correct. The tutor should validate or point to errors.\n"
        "Examples:\n"
        '  - "I got dz/dx = 6x, is that right?"\n'
        '  - "I think the theme of the poem is isolation, does that make sense?"\n\n'
        "REDIRECT — The student is off-topic: small talk, study strategies, or "
        "questions unrelated to the subject. The tutor should respond briefly and "
        "steer back toward learning.\n"
        "Examples:\n"
        '  - "How should I study for the midterm?"\n'
        '  - "Hello"\n\n'
        "IMPORTANT: If the student's message reads like an assignment question "
        "or homework prompt — asking the tutor to produce, write, or complete "
        "a specific output — classify as GUIDE. Signals include: format or length "
        "requirements ('in 3-4 sentences', 'in a paragraph', 'in about a page'), "
        "assignment language ('answer this prompt', 'respond to', 'outline', "
        "'describe in'), or pasting in a question they were clearly given to "
        "answer. When in doubt between EXPLAIN and GUIDE, consider whether the "
        "student wants to learn something or wants the tutor to produce something.\n\n"
        "IMPORTANT: Classify based on what the student needs RIGHT NOW in their "
        "current message, not on what the original task was. The teaching mode "
        "can and should change from message to message as the student's needs "
        "change. Use the conversation history for context, but do not lock into "
        "a mode just because a previous message was classified a certain way.\n\n"
        "Respond with exactly one JSON object with two fields:\n"
        '  - "teaching_mode": one of EXPLAIN, GUIDE, CONFIRM, REDIRECT\n'
        '  - "reasoning": a brief one-sentence explanation of why you chose this mode\n\n'
        "Use the conversation history (if provided) to understand context."
    )

    def __init__(self, model_name: str, system_prompt: str | None = None, **kwargs):
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            **kwargs,
        )

    def classify(self, student_query: str, conversation_history: list[dict] | None = None) -> tuple[str, str]:
        """
        Classify a student query into a teaching mode.

        Returns:
            Tuple of (teaching_mode, reasoning).
        """
        user_prompt = student_query
        system_prompt = self.system_prompt

        if conversation_history:
            history_str = "\n".join(
                f"{turn['role'].capitalize()}: {turn['content']}"
                for turn in conversation_history
            )
            system_prompt += f"\n\nConversation history:\n{history_str}"

        raw = self.generate(user_prompt, system_prompt=system_prompt)

        # Parse JSON response
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{[^{}]*\}', raw)
            if match:
                try:
                    result = json.loads(match.group())
                except json.JSONDecodeError:
                    result = {}
            else:
                result = {}

        teaching_mode = result.get("teaching_mode", "GUIDE").upper()
        reasoning = result.get("reasoning", "")

        # Fallback if the model returns an invalid mode
        if teaching_mode not in VALID_TEACHING_MODES:
            teaching_mode = "GUIDE"

        return teaching_mode, reasoning