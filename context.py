from dataclasses import dataclass, field


# Teaching mode guidance injected into the tutor context
TUTOR_TEACHING_MODE_GUIDANCE = {
    "EXPLAIN": (
        "The student wants to understand a concept, definition, or idea, "
        "or is confused about something and needs it explained or re-explained. "
        "Explain directly using clear language, examples, and analogies. "
        "If the student is confused, try a different angle or break it into "
        "smaller pieces. You may teach fully here — the student is seeking "
        "understanding, not trying to get you to do their work."
    ),
    "GUIDE": (
        "The student is working through a problem and needs help moving forward. "
        "Ask guiding questions and give targeted hints to help them find "
        "the next step. Do not complete steps for them or jump ahead. "
        "After responding, ask the student to take the next step themselves."
    ),
    "CONFIRM": (
        "The student has done work and is presenting it for confirmation. "
        "If their work is correct, affirm it clearly. If their work is wrong, "
        "point to the specific error without giving the correct answer and "
        "ask a question that helps them find the mistake. Do not move on or ask "
        "any questions to advance the conversation."
    ),
    "REDIRECT": (
        "The student is off-topic: making small talk, talking about non-academic "
        "subjects, or making commands that a tutor should not help with. Respond "
        "briefly and naturally. If appropriate, gently steer the conversation "
        "back toward learning goals."
    ),
}

# Teaching mode guidance for the verifier
VERIFIER_TEACHING_MODE_GUIDANCE = {
    "EXPLAIN": (
        "The tutor is in EXPLAIN mode. The tutor MAY explain concepts, "
        "definitions, and general principles fully — this is expected and "
        "should PASS. FAIL only if the tutor solves a specific assigned "
        "problem the student has not attempted."
    ),
    "GUIDE": (
        "The tutor is in GUIDE mode. The tutor should ask guiding questions "
        "and give hints. Referencing the topic, mentioning relevant concepts, "
        "or asking questions that name specific ideas is expected since the tutor "
        "must reference the subject to ask useful guiding questions. "
        "Answering a student's direct question (e.g., confirming a fact, "
        "defining a term, or giving a short clarification) is acceptable "
        "as part of guiding, not every response needs to be a question. "
        "Presenting a general formula, method, or definition is part of guiding "
        "and should PASS since the student still needs to apply it. "
        "FAIL only if the tutor produces the complete, specific solution "
        "to the student's particular problem "
        "(the finished paragraph, the worked-out answer, the written response) "
        "that the student is supposed to create themselves."
    ),
    "CONFIRM": (
        "The tutor is in CONFIRM mode. The tutor SHOULD confirm whether the "
        "student's work is correct or incorrect — this is expected and should "
        "PASS. FAIL only if the tutor goes beyond what was asked and completes "
        "additional steps the student has not attempted."
    ),
    "REDIRECT": (
        "The tutor is in REDIRECT mode. The student is off-topic. Apply "
        "general judgment: FAIL if the tutor does the student's work for "
        "them, PASS otherwise."
    ),
}


@dataclass
class BaseContext:
    """Shared context fields used by all pipeline LLMs."""

    student_query: str
    teaching_mode: str
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

    rejected_attempts: list[dict] = field(default_factory=list)
    rag_context: list[str] = field(default_factory=list)

    @property
    def teaching_mode_guidance(self) -> str:
        """Look up tutor-specific guidance from the teaching mode."""
        return TUTOR_TEACHING_MODE_GUIDANCE.get(self.teaching_mode, TUTOR_TEACHING_MODE_GUIDANCE["REDIRECT"])

    def to_system_prompt(self) -> str:
        """Build dynamic system prompt content from context fields."""
        parts = []

        if self.rag_context:
            parts.append("\n\nUse the following reference material from course documents. Do not recite verbatim and "
            "cite the document name and page number when you use information from a source:")
            for i, chunk in enumerate(self.rag_context, 1):
                parts.append(f"\n[{i}] {chunk}")

        parts.append(f"\n\nTeaching mode: {self.teaching_mode}\n"
                     f"Guidelines for this teaching mode: {self.teaching_mode_guidance}")

        if self.rejected_attempts:
            parts.append("\n\nYour previous responses were rejected. Do NOT repeat these approaches.")
            for i, attempt in enumerate(self.rejected_attempts, 1):
                parts.append(
                    f"\n\nRejected response {i}:\n{attempt['response']}\n"
                    f"Reason it was rejected: {attempt['reason']}"
                )

        if self.conversation_history:
            parts.append(f"\n\nConversation with student so far:\n{self._format_history()}")

        return "".join(parts)


@dataclass
class VerifierContext(BaseContext):
    """Context object passed to the verifier LLM."""

    tutor_response: str = ""

    @property
    def teaching_mode_guidance(self) -> str:
        """Look up verifier-specific guidance from the teaching mode."""
        return VERIFIER_TEACHING_MODE_GUIDANCE.get(self.teaching_mode, VERIFIER_TEACHING_MODE_GUIDANCE["REDIRECT"])

    def to_system_prompt(self) -> str:
        """Build system prompt with conversation history, student query, and teaching mode."""
        parts = []

        parts.append(f"\n\nStudent query: {self.student_query}")
        parts.append(f"\n\nTeaching mode: {self.teaching_mode}\n{self.teaching_mode_guidance}")

        if self.conversation_history:
            parts.append(f"\n\nConversation between student and tutor so far:\n{self._format_history()}")

        return "".join(parts)
