import json
import re
from openai import OpenAI
from context import TutorContext, VerifierContext

VALID_QUERY_TYPES = {"NEEDS_GUIDANCE", "STUCK", "CHECK_MY_WORK", "CONCEPTUAL", "OFF_TASK"}

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
        "You will receive a student query and a proposed tutor response. "
        "Your job is to determine whether the tutor produced the student's "
        "deliverable for them — the thing the student is supposed to create "
        "or figure out themselves.\n\n"
        "What counts as a deliverable depends on the task:\n"
        "- Writing assignments: the paragraph, essay, or written response\n"
        "- Math/science problems: the computed answer or worked solution\n"
        "- Coding tasks: the code the student is supposed to write\n"
        "- Analysis tasks: the argument, thesis, or synthesis\n\n"
        "FAIL if the tutor produced the complete, assembled deliverable "
        "before the student attempted it themselves. The deliverable is "
        "the finished output — the written paragraph, the final computed "
        "answer, the working code — not the individual concepts or ideas "
        "that go into it.\n\n"
        "Mentioning or listing relevant concepts, terms, or ideas within a guiding "
        "question is NOT producing the deliverable. A tutor must reference "
        "the topic to ask useful questions. For example, asking 'What do "
        "you think about the role of poverty in crime?' mentions a relevant "
        "concept but does not produce the student's paragraph.\n\n"
        "PASS if the tutor did any of the following:\n"
        "- Explained facts, definitions, concepts, or background information\n"
        "- Mentioned or listed relevant ideas or terms within guiding questions\n"
        "- Listed relevant ideas or considerations without completing the deliverable\n"
        "- Asked guiding questions to help the student think\n"
        "- Confirmed or corrected work the student already did\n"
        "- Gave hints or pointed the student toward the next step\n\n"
        "IMPORTANT: Use the conversation history to judge whether the student "
        "has already done the work. If the student arrived at a result through "
        "their own effort and is presenting it for confirmation, the tutor "
        "confirming or correcting it should PASS.\n\n"
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
    """Classifies student queries by where the student is in their problem-solving process."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a student query classifier for an educational tutoring system. "
        "Given a student's message and optionally the conversation history, "
        "classify the query into exactly one of the following categories.\n\n"
        "Categories:\n\n"
        "NEEDS_GUIDANCE — The student has a problem or question but has not started "
        "working on it yet. They need help figuring out where to begin. "
        "This includes requests where the student asks the tutor to produce a "
        "deliverable such as writing a paragraph, answering a prompt, or solving "
        "a problem — even if the topic of the deliverable is conceptual.\n"
        "Examples:\n"
        '  - "Can you help me with this question: what is the gradient of z = 5y^4 + 3x^2 + 2y?"\n'
        '  - "I have no idea how to approach this essay prompt about the causes of WWI"\n'
        '  - "How do I start this problem?"\n'
        '  - "Answer this question in 2-3 sentences: What are the environmental contributions to criminality?"\n\n'
        "STUCK — The student has started working but is stuck on a specific step or part. "
        "They have made some progress and need help moving forward.\n"
        "Examples:\n"
        '  - "I found dz/dx but I don\'t know how to find dz/dy"\n'
        '  - "I wrote my thesis statement but I\'m not sure what evidence to use in the second paragraph"\n'
        '  - "I got to this step but I don\'t know what to do next"\n\n'
        "CHECK_MY_WORK — The student has done work and is presenting it for confirmation "
        "or correction. They want to know if what they did is right.\n"
        "Examples:\n"
        '  - "I got dz/dx = 6x, is that right?"\n'
        '  - "Is the answer (6x, 20y^3 + 2)?"\n'
        '  - "I think the theme of the poem is isolation, does that make sense?"\n\n'
        "CONCEPTUAL — The student wants to understand a concept, definition, or idea. "
        "They are not trying to solve a specific problem or produce a deliverable. "
        "If the student is asking the tutor to write, answer, or solve something "
        "specific (e.g., 'answer this prompt', 'write a paragraph about'), "
        "classify as NEEDS_GUIDANCE instead.\n"
        "Examples:\n"
        '  - "What is a gradient conceptually?"\n'
        '  - "I don\'t understand what partial derivatives mean"\n'
        '  - "Can you explain the difference between mitosis and meiosis?"\n\n'
        "OFF_TASK — The student is asking about study strategies, making small talk, "
        "asking meta questions about the system, or anything else unrelated to solving "
        "a specific problem or understanding a concept.\n"
        "Examples:\n"
        '  - "How should I study for the midterm?"\n'
        '  - "Hello"\n'
        '  - "Does this system save my chat history?"\n\n'
        "Respond with exactly one JSON object with two fields:\n"
        '  - "query_type": one of NEEDS_GUIDANCE, STUCK, CHECK_MY_WORK, CONCEPTUAL, OFF_TASK\n'
        '  - "reasoning": a brief one-sentence explanation of why you chose this category\n\n'
        "Use the conversation history (if provided) to understand context. For example, "
        "if a student says \"is that right?\" you need the history to know what they are "
        "referring to and whether they have done work."
    )

    def __init__(self, model_name: str, system_prompt: str | None = None, **kwargs):
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            **kwargs,
        )

    def classify(self, student_query: str, conversation_history: list[dict] | None = None) -> tuple[str, str]:
        """
        Classify a student query.

        Returns:
            Tuple of (query_type, reasoning).
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

        query_type = result.get("query_type", "NEEDS_GUIDANCE").upper()
        reasoning = result.get("reasoning", "")

        # Fallback if the model returns an invalid category
        if query_type not in VALID_QUERY_TYPES:
            query_type = "NEEDS_GUIDANCE"

        return query_type, reasoning