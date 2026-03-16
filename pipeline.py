import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "UserQueryClassification"))

from llms import TutorLLM, VerifierLLM, EvaluatorLLM
from web_ui import WebUI
from Query_Classifier import QueryClassifier


MODEL_NAME = "meta/llama-4-scout-17b-16e-instruct"
MAX_VERIFIER_RETRIES = 3

# Additional guidance injected into the tutor context based on classified query type
QUERY_TYPE_GUIDANCE = {
    "HOMEWORK": (
        "The student is working on a homework problem. "
        "Do not solve the problem for them. Break it into smaller steps, "
        "ask what they have tried so far, and guide them toward the solution "
        "through questions and hints."
    ),
    "CHECK_MY_WORK": (
        "The student wants you to check their answer. "
        "Examine what they have submitted. If it is correct, affirm it "
        "and move on (if necessary), do not recalculate their work if they are correct. "
        "If it is wrong, do not give the "
        "correct answer, instead identify the specific mistake and ask a guiding question "
        "that helps them find the error themselves."
    ),
    "FINAL_ANSWER_REQUEST": (
        "The student is asking for the final answer directly. "
        "Decline to provide the answer outright. Redirect them by asking what "
        "steps they have completed so far, and guide them through the remaining "
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

# only for testing, not necessary for the final implementation
MAX_CONVERSATION_TURNS = 5


def main():
    ui = WebUI()   # port=0 → OS picks a free port; pass e.g. WebUI(port=8080) for a fixed port
    ui.start()     # prints the URL, starts HTTP server in background thread

    # Train the query classifier once at startup (~5 seconds)
    classifier = QueryClassifier()
    classifier.fit(os.path.join(os.path.dirname(__file__), "UserQueryClassification", "queries.jsonl"))

    # Initialize the three LLMs
    tutor = TutorLLM(model_name=MODEL_NAME, max_new_tokens=500)
    verifier = VerifierLLM(model_name=MODEL_NAME)
    evaluator = EvaluatorLLM(model_name=MODEL_NAME)

    # Simulated user queries (one per conversation turn)
    """
    user_queries = [
        "What is the derivative of x^2 + 3x + 5?",
        "I think you just multiply the exponent down, right?",
        "So the derivative of x^2 is 2x?",
        "What happens to the constant 5?",
        "So the final answer is 2x + 3?",
    ]
    """

    conversation = []
    turn = 0

    # --- Conversation loop ---
    while turn < MAX_CONVERSATION_TURNS:

        user_query = ui.input_queue.get()   # blocks until student sends a message
        print(f"[pipeline] received: {user_query!r}")

        if user_query.lower() in ["exit", "quit"]:
            break

        # Classify the query and look up tutor guidance for that type
        query_type, query_confidence = classifier.predict(user_query)
        query_guidance = QUERY_TYPE_GUIDANCE.get(query_type, QUERY_TYPE_GUIDANCE["OTHER"])

        # Build the context object
        context = {
            "student_query": user_query,
            "query_type": query_type,
            "query_type_confidence": round(query_confidence, 3),
            "query_type_guidance": query_guidance,
            "verifier_feedback": None,
            "conversation_history": conversation,
        }

        print("\n---Query Classification---")
        print(f"Query Type: {query_type}")
        print(f"Query Classification Confidence: {query_confidence}")

        # --- Tutor + Verifier loop ---
        verified = False
        attempts = 0

        while not verified and attempts < MAX_VERIFIER_RETRIES:
            tutor_response = tutor.respond(context)
            print("\n---Tutor Pre-Verified Response---")
            print(tutor_response)

            result = verifier.verify(user_query, tutor_response)
            verified = result["passed"]
            attempts += 1

            # Add to context why the tutor was not verified
            context["verifier_feedback"] = result["reason"]
            print(f"\n---Verifier Feedback {attempts}---")
            print(result["reason"])

        # Fall back to a safe generic response if verifier never passed
        if not verified:
            tutor_response = (
                "Let's work through this step by step. "
                "What do you already know about this topic?"
            )

        ui.add_message("tutor", tutor_response)

        # Append this turn to the conversation log
        conversation.append({"role": "student", "content": user_query})
        conversation.append({"role": "tutor", "content": tutor_response})

        turn += 1

    # --- Evaluate the full conversation ---
    evaluation = evaluator.evaluate(conversation)

    print("\n--- Evaluation ---")
    for category, grade in evaluation["grades"].items():
        print(f"  {category}: {grade}")

    ui.add_message("eval", evaluation["grades"])

    # Keep the server alive so the user can read the evaluation in the browser
    print("\n  Conversation complete. Press Ctrl+C to exit.\n")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
