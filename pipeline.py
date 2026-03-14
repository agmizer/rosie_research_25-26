from llms import TutorLLM, VerifierLLM, EvaluatorLLM
from web_ui import WebUI


MODEL_NAME = "meta/llama-4-scout-17b-16e-instruct"
MAX_VERIFIER_RETRIES = 3

# only for testing, not necessary for the final implementation
MAX_CONVERSATION_TURNS = 5


def main():
    ui = WebUI()   # port=0 → OS picks a free port; pass e.g. WebUI(port=8080) for a fixed port
    ui.start()     # prints the URL, starts HTTP server in background thread

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

        # Build the context object (placeholder for prompt building step)
        context = {
            "student_query": user_query,
            "verifier_feedback": None,
            "conversation_history": conversation,
        }

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
