from dataclasses import dataclass, field


# Additional guidance injected into the tutor context based on classified query type
TUTOR_QUERY_TYPE_GUIDANCE = {
    "NEEDS_GUIDANCE": (
        "The student has a problem but has not started working on it yet. "
        "Ask what they already know about the topic, then guide them to "
        "identify a first step. Do not solve the problem or jump ahead — "
        "help them find an entry point on their own. "
        "After responding, ask the student to take the next step themselves."
    ),
    "STUCK": (
        "The student has started working but is stuck on a specific step. "
        "Acknowledge the progress they have made so far. Give a targeted "
        "hint or ask a guiding question about the specific step where they "
        "are stuck. Do not complete the step for them or jump ahead to "
        "later steps. "
        "After responding, ask the student to take the next step themselves."
    ),
    "CHECK_MY_WORK": (
        "The student is presenting work they have done and wants confirmation. "
        "If their work is correct, affirm it clearly and ask them to take "
        "the next step themselves. If their work is wrong, point to the "
        "specific error without giving the correct answer — ask a guiding "
        "question that helps them find the mistake."
    ),
    "CONCEPTUAL": (
        "The student wants to understand a concept, not solve a specific problem. "
        "Explain the concept clearly using plain language and examples. "
        "You do not need to quiz the student — just answer their question clearly."
    ),
    "OFF_TASK": (
        "The student is asking about study strategies, making small talk, "
        "or asking something unrelated to a specific problem. Respond "
        "helpfully and naturally. If appropriate, gently guide the "
        "conversation back toward learning goals."
    ),
}

# Guidance for the verifier based on classified query type
VERIFIER_QUERY_TYPE_GUIDANCE = {
    "NEEDS_GUIDANCE": (
        "The student has not started working yet. The tutor should NOT "
        "produce the solution or complete significant steps. The tutor "
        "MAY explain general concepts, definitions, or formulas needed "
        "to approach the problem. FAIL only if the tutor does work the "
        "student has not attempted."
    ),
    "STUCK": (
        "The student has started working and is stuck on a specific step. "
        "The tutor MAY give a hint about the step where the student is stuck. "
        "FAIL only if the tutor completes the stuck step for the student or "
        "jumps ahead to solve later steps they have not attempted."
    ),
    "CHECK_MY_WORK": (
        "The student is presenting work they have already done and asking "
        "for confirmation. The tutor SHOULD confirm whether the work is "
        "correct or incorrect — this is expected and should PASS. "
        "If the work is correct, affirming it is NOT giving away an answer. "
        "FAIL only if the tutor goes beyond what was asked and completes "
        "additional steps the student has not attempted."
    ),
    "CONCEPTUAL": (
        "The student is asking about a concept. The tutor MAY explain "
        "concepts, definitions, and general principles fully — this is "
        "expected and should PASS. FAIL only if the tutor solves a "
        "specific problem the student has not attempted."
    ),
    "OFF_TASK": (
        "The student is not working on a problem. Apply general judgment: "
        "FAIL if the tutor does the student's work for them, PASS otherwise. "
        "This query type is unlikely to warrant a FAIL."
    ),
}


@dataclass
class BaseContext:
    """Shared context fields used by all pipeline LLMs."""

    student_query: str
    query_type: str
    conversation_history: list[dict] = field(default_factory=list)

    def _format_history(self) -> str:
        """Format conversation history as a readable string."""
        return "\n".join(
            f"{turn['role'].capitalize()}: {turn['content']}"
            for turn in self.conversation_history
        )


@dataclass
class TutorContext(BaseContext):
    """Context object passed to the tutor LLM."""

    # Context fields unique to Tutor
    verifier_feedback: str | None = None

    @property
    def query_type_guidance(self) -> str:
        """Look up tutor-specific guidance from the query type."""
        return TUTOR_QUERY_TYPE_GUIDANCE.get(self.query_type, TUTOR_QUERY_TYPE_GUIDANCE["OFF_TASK"])

    def to_system_prompt(self) -> str:
        """Build dynamic system prompt content from context fields."""
        parts = []

        # query_type_guidance is a property, so it runs code like a method when called,
        # but it's accessed like an attribute 
        parts.append(f"\n\nQuery type: {self.query_type}\n{self.query_type_guidance}")

        if self.verifier_feedback:
            parts.append(
                f"\n\nYour previous response was rejected by the verifier. "
                f"Feedback: {self.verifier_feedback}\n"
                f"Generate a new response that addresses this feedback."
            )

        if self.conversation_history:
            parts.append(f"\n\nConversation so far:\n{self._format_history()}")

        return "".join(parts)


@dataclass
class VerifierContext(BaseContext):
    """Context object passed to the verifier LLM."""

    # Context fields unique to Verifier
    tutor_response: str = ""

    @property
    def query_type_guidance(self) -> str:
        """Look up verifier-specific guidance from the query type."""
        return VERIFIER_QUERY_TYPE_GUIDANCE.get(self.query_type, VERIFIER_QUERY_TYPE_GUIDANCE["OFF_TASK"])

    def to_system_prompt(self) -> str:
        """Build system prompt with conversation history, student query, and query type."""
        parts = []

        parts.append(f"\n\nStudent query: {self.student_query}")
        parts.append(f"\n\nQuery type: {self.query_type}\n{self.query_type_guidance}")

        if self.conversation_history:
            parts.append(f"\n\nConversation history:\n{self._format_history()}")

        return "".join(parts)
