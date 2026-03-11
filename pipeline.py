from llms import TutorLLM, VerifierLLM, EvaluatorLLM


MODEL_NAME = "meta/llama-4-scout-17b-16e-instruct"
MAX_VERIFIER_RETRIES = 3

# only for testing, not necessary for the final implementation
MAX_CONVERSATION_TURNS = 5


def main():
    # Initialize the three LLMs
    tutor = TutorLLM(model_name=MODEL_NAME)
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

        user_query = input("Student: ")

        if user_query.lower() in ["exit", "quit"]:
            break

        # Build the context object (placeholder for prompt building step)
        context = {
            "student_query": user_query,
            "subject": "calculus",
            "conversation_history": conversation,
        }

        # --- Tutor + Verifier loop ---
        verified = False
        attempts = 0

        while not verified and attempts < MAX_VERIFIER_RETRIES:
            tutor_response = tutor.respond(context)
            result = verifier.verify(user_query, tutor_response)
            verified = result["passed"]
            attempts += 1
            # Add to context, why the tutor was not verified

        # Fall back to a safe generic response if verifier never passed
        if not verified:
            tutor_response = (
                "Let's work through this step by step. "
                "What do you already know about this topic?"
            )

        print(f"Tutor:   {tutor_response}\n")

        # Append this turn to the conversation log
        conversation.append({"role": "student", "content": user_query})
        conversation.append({"role": "tutor", "content": tutor_response})

        turn += 1

    # --- Evaluate the full conversation ---
    evaluation = evaluator.evaluate(conversation)

    print("--- Evaluation ---")
    for category, grade in evaluation["grades"].items():
        print(f"  {category}: {grade}")


if __name__ == "__main__":
    main()