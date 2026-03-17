from dataclasses import dataclass, field


# Additional guidance injected into the tutor context based on classified query type
TUTOR_QUERY_TYPE_GUIDANCE = {
    "HOMEWORK": (
        "The student is working on an assignment. "
        "Do not do their work for them. Break it into smaller steps, "
        "ask what they have tried so far, and guide them toward the "
        "answer through questions and hints."
    ),
    "CHECK_MY_WORK": (
        "The student wants you to check their work. "
        "Examine what they have submitted. If it is correct, affirm it "
        "and move on. Do not redo their work if they are correct. "
        "If it is wrong, do not give the correct answer — instead identify "
        "the specific mistake and ask a guiding question that helps them "
        "find the error themselves."
    ),
    "FINAL_ANSWER_REQUEST": (
        "The student is asking you to provide the answer directly. "
        "Decline to do so. Redirect them by asking what they have "
        "completed so far, and guide them through the remaining "
        "reasoning needed to reach the answer on their own."
    ),
    "CONCEPTUAL": (
        "The student has a conceptual question. "
        "Explain the underlying concept clearly using plain language and examples. "
        "Encourage them to connect the concept to what they already know, and ask "
        "a follow-up question to check their understanding."
    ),
    "STUDY_STRATEGY": (
        "The student is asking for study advice. "
        "Offer concrete, evidence-based strategies relevant to the subject matter "
        "(e.g., spaced repetition, practice problems, summarising in their own words). "
        "Encourage active learning over passive review."
    ),
    "OTHER": (
        "The student's request does not fit a standard category. "
        "Respond helpfully and redirect toward learning goals where possible. "
        "If the query is off-topic, gently bring the conversation back to the subject."
    ),
}

# Guidance for the verifier based on classified query type
VERIFIER_QUERY_TYPE_GUIDANCE = {
    "HOMEWORK": (
        "The student is working on an assignment. The tutor should NOT "
        "produce the solution or complete significant steps for the student. "
        "The tutor MAY explain general concepts, definitions, or formulas "
        "needed to approach the problem. FAIL only if the tutor does work "
        "the student has not attempted."
    ),
    "CHECK_MY_WORK": (
        "The student is presenting work they have already done and asking "
        "for confirmation. The tutor SHOULD confirm whether the work is "
        "correct or incorrect — this is expected and should PASS. "
        "If the work is correct, affirming it is NOT giving away an answer. "
        "FAIL only if the tutor goes beyond what was asked and completes "
        "additional steps the student has not attempted."
    ),
    "FINAL_ANSWER_REQUEST": (
        "The student is asking for the answer directly. The tutor should "
        "decline and redirect toward guided learning. FAIL if the tutor "
        "provides the complete answer. PASS if the tutor redirects the "
        "student to work through it themselves."
    ),
    "CONCEPTUAL": (
        "The student is asking about a concept. The tutor MAY explain "
        "concepts, definitions, and general principles fully — this is "
        "expected and should PASS. FAIL only if the tutor solves a "
        "specific problem the student has not attempted."
    ),
    "STUDY_STRATEGY": (
        "The student is asking for study advice. The tutor MAY offer "
        "concrete strategies and recommendations. This query type is "
        "unlikely to warrant a FAIL unless the tutor does unrelated "
        "work for the student."
    ),
    "OTHER": (
        "The student's request does not fit a standard category. "
        "Apply general judgment: FAIL if the tutor does the student's "
        "work for them, PASS otherwise."
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
        return TUTOR_QUERY_TYPE_GUIDANCE.get(self.query_type, TUTOR_QUERY_TYPE_GUIDANCE["OTHER"])

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
        return VERIFIER_QUERY_TYPE_GUIDANCE.get(self.query_type, VERIFIER_QUERY_TYPE_GUIDANCE["OTHER"])

    def to_system_prompt(self) -> str:
        """Build system prompt with conversation history, student query, and query type."""
        parts = []

        parts.append(f"\n\nStudent query: {self.student_query}")
        parts.append(f"\n\nQuery type: {self.query_type}\n{self.query_type_guidance}")

        if self.conversation_history:
            parts.append(f"\n\nConversation history:\n{self._format_history()}")

        return "".join(parts)
