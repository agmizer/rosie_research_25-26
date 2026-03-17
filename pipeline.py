import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "UserQueryClassification"))

from llms import TutorLLM, VerifierLLM, EvaluatorLLM
from web_ui import WebUI
from Query_Classifier import QueryClassifier
from context import TutorContext, VerifierContext


MODEL_NAME = "meta/llama-4-scout-17b-16e-instruct"
MAX_VERIFIER_RETRIES = 3

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

    # --------------- Conversation loop ---------------
    while turn < MAX_CONVERSATION_TURNS:

        user_query = ui.input_queue.get()   # blocks until student sends a message
        print(f"[pipeline] received: {user_query!r}")

        if user_query.lower() in ["exit", "quit"]:
            break

        # Classify the query and build the context object
        query_type, query_confidence = classifier.predict(user_query)

        tutor_context = TutorContext(
            student_query=user_query,
            query_type=query_type,
            conversation_history=conversation,
        )

        print("\n---Query Classification---")
        print(f"Query Type: {query_type}")
        print(f"Query Classification Confidence: {query_confidence}")


        # ----------- Tutor + Verifier loop -----------
        verified = False
        attempts = 0

        while not verified and attempts < MAX_VERIFIER_RETRIES:
            tutor_response = tutor.respond(tutor_context)
            print("\n---Tutor Pre-Verified Response---")
            print(tutor_response)

            verifier_context = VerifierContext(
                student_query=user_query,
                query_type=query_type,
                conversation_history=conversation,
                tutor_response=tutor_response,
            )
            result = verifier.verify(verifier_context)
            verified = result["passed"]
            attempts += 1

            # Add to tutor context why the response was not verified
            tutor_context.verifier_feedback = result["reason"]
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


    # -------- Evaluate the full conversation --------
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
