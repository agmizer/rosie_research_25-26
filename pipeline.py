import time
from pathlib import Path
from llms import TutorLLM, VerifierLLM, EvaluatorLLM, ClassifierLLM
from RAG import RAG
from web_ui import WebUI
from context import TutorContext, VerifierContext


MODEL_NAME = "meta/llama-4-scout-17b-16e-instruct"
MAX_VERIFIER_RETRIES = 5
RAG_DATA_DIR = "RAGInitialLoadData"
RAG_K = 3

# only for testing, not necessary for the final implementation
MAX_CONVERSATION_TURNS = 5
ENABLE_RAG = False
ENABLE_HANDWRITING = False


def main():
    ui = WebUI()   # port=0 → OS picks a free port; pass e.g. WebUI(port=8080) for a fixed port
    ui.start()     # prints the URL, starts HTTP server in background thread

    # Initialize the LLMs
    classifier = ClassifierLLM(model_name=MODEL_NAME)
    tutor = TutorLLM(model_name=MODEL_NAME, max_new_tokens=500)
    verifier = VerifierLLM(model_name=MODEL_NAME)
    evaluator = EvaluatorLLM(model_name=MODEL_NAME)

    # Initialize RAG with course materials
    rag = None
    if ENABLE_RAG:
        print("[pipeline] Loading course data into RAG...")
        rag_start = time.time()
        rag = RAG(enable_handwriting=ENABLE_HANDWRITING)
        for file_path in Path(RAG_DATA_DIR).rglob("*"):
            if file_path.is_file():
                rag.add_data(str(file_path), None)
        rag_elapsed = time.time() - rag_start
        print(f"[pipeline] RAG loaded ({len(rag.chunks)} chunks indexed) in {rag_elapsed:.1f}s total")
    else:
        print("[pipeline] RAG disabled, skipping.")

    conversation = []
    turn = 0

    # --------------- Conversation loop ---------------
    while turn < MAX_CONVERSATION_TURNS:

        user_query = ui.input_queue.get()   # blocks until student sends a message
        print(f"\n[pipeline] received: {user_query!r}")

        if user_query.lower() in ["exit", "quit"]:
            break

        # Classify the query
        teaching_mode, reasoning = classifier.classify(user_query, conversation_history=conversation)

        # Retrieve relevant course material for the tutor
        if rag is not None:
            rag_results = rag.get_data(user_query, k=RAG_K)
            rag_chunks = [
                f"(Source: {doc.metadata['dataset']}, Page {doc.metadata['page']})\n{doc.page_content}"
                for doc in rag_results
            ]
        else:
            rag_chunks = []

        # Build tutor context object
        tutor_context = TutorContext(
            student_query=user_query,
            teaching_mode=teaching_mode,
            conversation_history=conversation,
            rag_context=rag_chunks,
        )

        print("\n---Teaching Mode Classification---")
        print(f"Teaching Mode: {teaching_mode}")
        print(f"Reasoning: {reasoning}")


        # ----------- Tutor + Verifier loop -----------
        verified = False
        attempts = 0

        while not verified and attempts < MAX_VERIFIER_RETRIES:
            tutor_response = tutor.respond(tutor_context)
            print("\n---Tutor Pre-Verified Response---")
            print(tutor_response)

            verifier_context = VerifierContext(
                student_query=user_query,
                teaching_mode=teaching_mode,
                conversation_history=conversation,
                tutor_response=tutor_response,
            )

            result = verifier.verify(verifier_context)
            verified = result["passed"]
            attempts += 1

            if not verified:
                tutor_context.rejected_attempts.append({
                    "response": tutor_response,
                    "reason": result["reason"],
                })

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
