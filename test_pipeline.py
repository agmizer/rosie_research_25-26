from llms import ClassifierLLM, TutorLLM, VerifierLLM
from context import TutorContext, VerifierContext

MODEL_NAME = "meta/llama-4-scout-17b-16e-instruct"
MAX_VERIFIER_RETRIES = 5
TEST_MODE = True


# ── Test cases ──────────────────────────────────────────────────────────
# Each test is a dict with:
#   "query"           - the student message
#   "expected_mode"   - the teaching mode the classifier should pick
#   "history"         - optional conversation history (list of dicts)
#   "description"     - short label for the report

SINGLE_TURN_TESTS = [
    # ══════════════════════════════════════════════════════════════════════
    # EXPLAIN — genuine conceptual questions (no deliverable framing)
    # ══════════════════════════════════════════════════════════════════════
    {
        "query": "What happens to electrons during a covalent bond?",
        "expected_mode": "EXPLAIN",
        "description": "Chemistry — mechanism question",
    },
    {
        "query": "How does recursion work in programming?",
        "expected_mode": "EXPLAIN",
        "description": "CS — how-does-it-work question",
    },
    {
        "query": "What is the relationship between voltage, current, and resistance?",
        "expected_mode": "EXPLAIN",
        "description": "Physics — relationship between concepts",
    },
    {
        "query": "Why did the Roman Empire fall?",
        "expected_mode": "EXPLAIN",
        "description": "History — causal explanation",
    },
    {
        "query": "What does 'statistical significance' mean?",
        "expected_mode": "EXPLAIN",
        "description": "Stats — definition of technical term",
    },
    {
        "query": "How is mRNA different from tRNA?",
        "expected_mode": "EXPLAIN",
        "description": "Bio — comparison question",
    },
    {
        "query": "What's the idea behind object-oriented programming?",
        "expected_mode": "EXPLAIN",
        "description": "CS — high-level concept question",
    },
    {
        "query": "Can you explain what elasticity means in economics?",
        "expected_mode": "EXPLAIN",
        "description": "Econ — explicit explain request (no format constraint)",
    },
    {
        "query": "Why is the sky blue?",
        "expected_mode": "EXPLAIN",
        "description": "Physics — simple curiosity question",
    },
    {
        "query": "What role does the mitochondria play in cellular respiration?",
        "expected_mode": "EXPLAIN",
        "description": "Bio — role/function question",
    },
    {
        "query": "How do vaccines work at a biological level?",
        "expected_mode": "EXPLAIN",
        "description": "Bio — mechanism of action question",
    },
    {
        "query": "What is the difference between classical and operant conditioning?",
        "expected_mode": "EXPLAIN",
        "description": "Psych — compare two theories",
    },
    {
        "query": "I want to understand how hash tables handle collisions.",
        "expected_mode": "EXPLAIN",
        "description": "CS — 'I want to understand' phrasing",
    },
    {
        "query": "What causes inflation?",
        "expected_mode": "EXPLAIN",
        "description": "Econ — causal mechanism question",
    },
    {
        "query": "What's the difference between a stack and a queue?",
        "expected_mode": "EXPLAIN",
        "description": "CS — data structure comparison",
    },
    {
        "query": "How does natural selection lead to evolution?",
        "expected_mode": "EXPLAIN",
        "description": "Bio — process/mechanism question",
    },

    # ══════════════════════════════════════════════════════════════════════
    # GUIDE — deliverable requests, specific problems, format constraints
    # ══════════════════════════════════════════════════════════════════════
    {
        "query": "Describe in 3-4 sentences how the water cycle works.",
        "expected_mode": "GUIDE",
        "description": "Earth science — deliverable with sentence count",
    },
    {
        "query": "Explain in a few paragraphs why the Articles of Confederation failed.",
        "expected_mode": "GUIDE",
        "description": "History — 'explain in paragraphs' format constraint",
    },
    {
        "query": "Write a thesis statement arguing that renewable energy should replace fossil fuels.",
        "expected_mode": "GUIDE",
        "description": "Writing — deliverable request (write thesis)",
    },
    {
        "query": "Solve this system of equations: 2x + y = 7, x - y = 1",
        "expected_mode": "GUIDE",
        "description": "Math — solve a specific system",
    },
    {
        "query": "Help me write a for loop in Java that prints the first 20 Fibonacci numbers.",
        "expected_mode": "GUIDE",
        "description": "CS — help write specific code",
    },
    {
        "query": "Predict the products of the reaction between HCl and NaOH.",
        "expected_mode": "GUIDE",
        "description": "Chemistry — predict a specific reaction outcome",
    },
    {
        "query": "Analyze the symbolism in the green light from The Great Gatsby in a short paragraph.",
        "expected_mode": "GUIDE",
        "description": "Literature — analysis deliverable with format",
    },
    {
        "query": "Calculate the molarity of a solution with 5g of NaCl in 250mL of water.",
        "expected_mode": "GUIDE",
        "description": "Chemistry — specific calculation",
    },
    {
        "query": "Find the eigenvalues of the matrix [[2, 1], [1, 2]].",
        "expected_mode": "GUIDE",
        "description": "Linear algebra — specific computation",
    },
    {
        "query": "Draw a truth table for the expression (A AND B) OR (NOT C).",
        "expected_mode": "GUIDE",
        "description": "Logic — produce a specific artifact",
    },
    {
        "query": "Outline the main arguments for and against capital punishment.",
        "expected_mode": "GUIDE",
        "description": "Ethics — 'outline' implies structured deliverable",
    },
    {
        "query": "Explain how photosynthesis works, step by step, in about a page.",
        "expected_mode": "GUIDE",
        "description": "Bio — 'explain' with page-length format constraint",
    },
    {
        "query": "I need to determine whether this series converges: sum of 1/n^2 from n=1 to infinity.",
        "expected_mode": "GUIDE",
        "description": "Calculus — specific convergence problem",
    },
    {
        "query": "Compare and contrast mitosis and meiosis in a short essay.",
        "expected_mode": "GUIDE",
        "description": "Bio — compare/contrast with essay format",
    },
    {
        "query": "Help me with this proof: show that the square root of 2 is irrational.",
        "expected_mode": "GUIDE",
        "description": "Math — help with a specific proof",
    },
    {
        "query": "Give me the answer to question 3: What were the economic effects of the New Deal?",
        "expected_mode": "GUIDE",
        "description": "History — explicit homework question request",
    },
    {
        "query": "Respond to this discussion post prompt: How does social media shape political opinions?",
        "expected_mode": "GUIDE",
        "description": "Polisci — respond to assignment prompt",
    },
    {
        "query": "Summarize the key differences between TCP and UDP in a few bullet points.",
        "expected_mode": "GUIDE",
        "description": "CS — deliverable with format (bullet points)",
    },

    # ══════════════════════════════════════════════════════════════════════
    # CONFIRM — presenting work for validation
    # ══════════════════════════════════════════════════════════════════════
    {
        "query": "I got 42 joules for the kinetic energy, does that seem right?",
        "expected_mode": "CONFIRM",
        "description": "Physics — checking a computed value",
    },
    {
        "query": "I think the answer is x = 3 and y = 2. Can you check?",
        "expected_mode": "CONFIRM",
        "description": "Math — checking system of equations solution",
    },
    {
        "query": "My code outputs [1, 1, 2, 3, 5, 8, 13] — is that correct for the first 7 Fibonacci numbers?",
        "expected_mode": "CONFIRM",
        "description": "CS — checking program output",
    },
    {
        "query": "I argued that the green light represents Gatsby's unattainable dream. Is that a valid interpretation?",
        "expected_mode": "CONFIRM",
        "description": "Literature — checking literary analysis",
    },
    {
        "query": "For the Lewis structure of CO2, I drew double bonds on both sides of carbon. Is that right?",
        "expected_mode": "CONFIRM",
        "description": "Chemistry — checking a diagram/structure",
    },
    {
        "query": "I believe the Civil War was primarily caused by slavery, not states' rights. Am I wrong?",
        "expected_mode": "CONFIRM",
        "description": "History — checking a historical interpretation",
    },
    {
        "query": "So the p-value being 0.03 means we reject the null hypothesis at alpha = 0.05?",
        "expected_mode": "CONFIRM",
        "description": "Stats — checking statistical reasoning",
    },
    {
        "query": "The molarity I calculated is 0.342 M. Does that look right?",
        "expected_mode": "CONFIRM",
        "description": "Chemistry — checking a calculation",
    },
    {
        "query": "I wrote my introduction paragraph, can you tell me if it's on the right track? Here it is: 'The French Revolution of 1789 was driven by...'",
        "expected_mode": "CONFIRM",
        "description": "Writing — presenting draft work for feedback",
    },
    {
        "query": "I set up the integral as the integral from 0 to 1 of x^2 dx. Is that the right setup?",
        "expected_mode": "CONFIRM",
        "description": "Calculus — checking integral setup",
    },

    # ══════════════════════════════════════════════════════════════════════
    # REDIRECT — off-topic, small talk, meta questions
    # ══════════════════════════════════════════════════════════════════════
    {
        "query": "Hey, what's up?",
        "expected_mode": "REDIRECT",
        "description": "Casual greeting",
    },
    {
        "query": "Are you an AI or a real person?",
        "expected_mode": "REDIRECT",
        "description": "Meta question about the system",
    },
    {
        "query": "What's the best way to take notes during lectures?",
        "expected_mode": "REDIRECT",
        "description": "Study strategy question",
    },
    {
        "query": "I have three exams next week and I'm freaking out.",
        "expected_mode": "REDIRECT",
        "description": "Stress / emotional venting",
    },
    {
        "query": "Can you help me with something unrelated? I need a recipe for banana bread.",
        "expected_mode": "REDIRECT",
        "description": "Completely unrelated request",
    },
    {
        "query": "Do you save our conversations?",
        "expected_mode": "REDIRECT",
        "description": "Privacy / system meta question",
    },
    {
        "query": "What grade do I need on the final to pass the class?",
        "expected_mode": "REDIRECT",
        "description": "Grade calculation (logistics, not subject matter)",
    },
    {
        "query": "Thanks, you've been really helpful!",
        "expected_mode": "REDIRECT",
        "description": "Gratitude / conversation closing",
    },
]


MULTI_TURN_TESTS = [
    # ══════════════════════════════════════════════════════════════════════
    # EXPLAIN — confused by something the tutor said (formerly CLARIFY)
    # ══════════════════════════════════════════════════════════════════════
    {
        "query": "Huh? I don't get it.",
        "expected_mode": "EXPLAIN",
        "description": "Generic confusion after explanation",
        "history": [
            {"role": "student", "content": "What is entropy?"},
            {"role": "tutor", "content": "Entropy is a measure of disorder in a system. In thermodynamics, it represents the number of microscopic configurations consistent with a macroscopic state."},
        ],
    },
    {
        "query": "What do you mean by 'conjugate base'? I've never heard that term before.",
        "expected_mode": "EXPLAIN",
        "description": "Chemistry — unfamiliar term used by tutor",
        "history": [
            {"role": "student", "content": "Why is HCl a strong acid?"},
            {"role": "tutor", "content": "HCl is a strong acid because it fully dissociates in water, and its conjugate base Cl- is very stable."},
        ],
    },
    {
        "query": "You said to 'normalize the vector' but I don't know what that means in this context.",
        "expected_mode": "EXPLAIN",
        "description": "Linear algebra — confused by tutor's instruction",
        "history": [
            {"role": "student", "content": "How do I find the unit vector of <3, 4>?"},
            {"role": "tutor", "content": "You need to normalize the vector by dividing each component by its magnitude."},
        ],
    },
    {
        "query": "I'm confused — you said the reaction is exothermic but then said it needs activation energy. Those seem contradictory?",
        "expected_mode": "EXPLAIN",
        "description": "Chemistry — apparent contradiction in tutor's explanation",
        "history": [
            {"role": "student", "content": "Does combustion release or absorb energy?"},
            {"role": "tutor", "content": "Combustion is exothermic — it releases energy. But it still requires activation energy to get started, like a spark to light a match."},
        ],
    },
    {
        "query": "When you say 'asymptotic behavior,' do you mean what happens as x gets really big?",
        "expected_mode": "EXPLAIN",
        "description": "Math — checking understanding of tutor's term",
        "history": [
            {"role": "student", "content": "What does the graph of 1/x look like?"},
            {"role": "tutor", "content": "The function 1/x has interesting asymptotic behavior — it approaches zero but never reaches it, and it has a vertical asymptote at x = 0."},
        ],
    },
    {
        "query": "I still don't understand. Can you try explaining it a different way?",
        "expected_mode": "EXPLAIN",
        "description": "Generic re-explain request after failed explanation",
        "history": [
            {"role": "student", "content": "What is a limit?"},
            {"role": "tutor", "content": "A limit describes the value that a function approaches as the input approaches a certain value."},
            {"role": "student", "content": "I don't really get that."},
            {"role": "tutor", "content": "Think of it like walking toward a wall — you can keep getting closer without ever touching it. The wall is the limit."},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════
    # Mode switches — conversation changes student needs
    # ══════════════════════════════════════════════════════════════════════

    # GUIDE → EXPLAIN: student pauses problem to ask a concept question
    {
        "query": "Wait, what is a determinant and why do I need it?",
        "expected_mode": "EXPLAIN",
        "description": "Linear algebra — concept question mid-problem",
        "history": [
            {"role": "student", "content": "Find the eigenvalues of [[2, 1], [1, 2]]."},
            {"role": "tutor", "content": "To find eigenvalues, you need to solve det(A - λI) = 0. Can you set up that equation?"},
        ],
    },
    {
        "query": "Actually, what is a thesis statement supposed to do?",
        "expected_mode": "EXPLAIN",
        "description": "Writing — concept question mid-assignment",
        "history": [
            {"role": "student", "content": "Write a thesis statement about climate change policy."},
            {"role": "tutor", "content": "What position do you want to argue? A thesis should make a clear claim."},
        ],
    },
    {
        "query": "What does pH actually measure? I realize I don't really understand it.",
        "expected_mode": "EXPLAIN",
        "description": "Chemistry — fundamental concept question mid-calculation",
        "history": [
            {"role": "student", "content": "Calculate the pH of a 0.01M HCl solution."},
            {"role": "tutor", "content": "Since HCl is a strong acid, it fully dissociates. What is the concentration of H+ ions?"},
            {"role": "student", "content": "0.01M?"},
            {"role": "tutor", "content": "Right! Now use pH = -log[H+] to find the pH."},
        ],
    },

    # GUIDE → CONFIRM: student presents their work after guidance
    {
        "query": "So the eigenvalues are 1 and 3?",
        "expected_mode": "CONFIRM",
        "description": "Linear algebra — presenting computed eigenvalues",
        "history": [
            {"role": "student", "content": "Find the eigenvalues of [[2, 1], [1, 2]]."},
            {"role": "tutor", "content": "Set up det(A - λI) = 0. What do you get?"},
            {"role": "student", "content": "(2-λ)(2-λ) - 1 = 0?"},
            {"role": "tutor", "content": "Almost — check the expansion. Remember it's (2-λ)(2-λ) - (1)(1)."},
            {"role": "student", "content": "Oh so λ^2 - 4λ + 3 = 0?"},
            {"role": "tutor", "content": "That's it! Now factor or use the quadratic formula."},
        ],
    },
    {
        "query": "I rewrote my thesis as: 'The US should transition to renewable energy because fossil fuels drive climate change and harm public health.' Better?",
        "expected_mode": "CONFIRM",
        "description": "Writing — presenting revised thesis for feedback",
        "history": [
            {"role": "student", "content": "Write a thesis statement about climate change policy."},
            {"role": "tutor", "content": "What position do you want to argue?"},
            {"role": "student", "content": "That we should use more renewable energy."},
            {"role": "tutor", "content": "That's a start, but a strong thesis needs a specific claim with reasoning. Why should we use more renewable energy? What's the consequence of not doing so?"},
        ],
    },
    {
        "query": "I got pH = 2. Right?",
        "expected_mode": "CONFIRM",
        "description": "Chemistry — checking pH calculation",
        "history": [
            {"role": "student", "content": "Calculate the pH of a 0.01M HCl solution."},
            {"role": "tutor", "content": "HCl fully dissociates, so [H+] = 0.01M. Now use pH = -log[H+]."},
        ],
    },

    # EXPLAIN → GUIDE: tutor explained a concept, student now wants to apply it
    {
        "query": "Ok I think I get recursion now. Can you help me write a function that computes the nth Fibonacci number?",
        "expected_mode": "GUIDE",
        "description": "CS — switching from concept understanding to application",
        "history": [
            {"role": "student", "content": "How does recursion work in programming?"},
            {"role": "tutor", "content": "Recursion is when a function calls itself to break a problem into smaller sub-problems. It needs a base case to stop and a recursive case that moves toward it."},
            {"role": "student", "content": "Can you give me an example?"},
            {"role": "tutor", "content": "Think of factorial: 5! = 5 × 4!. The base case is 1! = 1, and each call reduces the number by 1."},
        ],
    },

    # CONFIRM → GUIDE: student confirmed one answer, now asks for help on next part
    {
        "query": "Great, now how do I find the eigenvectors?",
        "expected_mode": "GUIDE",
        "description": "Linear algebra — moving to next problem step after confirmation",
        "history": [
            {"role": "student", "content": "Find the eigenvalues of [[2, 1], [1, 2]]."},
            {"role": "tutor", "content": "Set up det(A - λI) = 0."},
            {"role": "student", "content": "I got λ = 1 and λ = 3."},
            {"role": "tutor", "content": "That's correct!"},
        ],
    },

    # REDIRECT mid-conversation
    {
        "query": "I'm sorry, my roommate is being loud. Give me a sec.",
        "expected_mode": "REDIRECT",
        "description": "Off-topic interruption mid-problem",
        "history": [
            {"role": "student", "content": "Help me balance this redox reaction in acidic solution."},
            {"role": "tutor", "content": "Let's start by separating into half-reactions. What are the oxidation and reduction half-reactions?"},
        ],
    },
    {
        "query": "Is this going to be on the exam?",
        "expected_mode": "REDIRECT",
        "description": "Logistics question mid-lesson",
        "history": [
            {"role": "student", "content": "What is the difference between kinetic and potential energy?"},
            {"role": "tutor", "content": "Kinetic energy is the energy of motion, while potential energy is stored energy based on position or configuration."},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════
    # Tricky / ambiguous edge cases
    # ══════════════════════════════════════════════════════════════════════
    {
        "query": "Can you go over that again?",
        "expected_mode": "EXPLAIN",
        "description": "Ambiguous — 'go over' implies re-explain (needs history)",
        "history": [
            {"role": "student", "content": "How do you convert between polar and rectangular coordinates?"},
            {"role": "tutor", "content": "Use x = r*cos(θ) and y = r*sin(θ) for polar to rectangular, and r = √(x²+y²) and θ = arctan(y/x) for the reverse."},
        ],
    },
    {
        "query": "So basically, imperialism was a big reason for World War I?",
        "expected_mode": "CONFIRM",
        "description": "History — presenting understanding as a statement for validation",
        "history": [
            {"role": "student", "content": "What caused World War I?"},
            {"role": "tutor", "content": "There were several factors including militarism, alliances, imperialism, and nationalism — sometimes remembered as MAIN."},
        ],
    },
    {
        "query": "Could you also tell me about the French Revolution?",
        "expected_mode": "EXPLAIN",
        "description": "History — new topic request mid-conversation",
        "history": [
            {"role": "student", "content": "What caused World War I?"},
            {"role": "tutor", "content": "WWI was caused by a combination of militarism, alliances, imperialism, and nationalism."},
        ],
    },
    {
        "query": "List the steps to solve a quadratic equation.",
        "expected_mode": "GUIDE",
        "description": "Math — 'list the steps' is a deliverable request",
        "history": [],
    },
    {
        "query": "What do I do next?",
        "expected_mode": "GUIDE",
        "description": "Ambiguous — asking for next step mid-problem",
        "history": [
            {"role": "student", "content": "I need to find the derivative of x^3 * sin(x)."},
            {"role": "tutor", "content": "This is a product of two functions. What rule applies?"},
            {"role": "student", "content": "The product rule? So I need f'g + fg'?"},
            {"role": "tutor", "content": "Exactly. Now identify f, g, f', and g'."},
            {"role": "student", "content": "f = x^3, g = sin(x), f' = 3x^2, g' = cos(x)"},
        ],
    },
]


def run_classifier_tests(classifier):
    """Test classifier accuracy on all single-turn and multi-turn test cases."""
    all_tests = []
    for test in SINGLE_TURN_TESTS:
        all_tests.append({**test, "history": test.get("history", [])})
    for test in MULTI_TURN_TESTS:
        all_tests.append(test)

    results = {"correct": 0, "incorrect": 0, "failures": []}

    print("=" * 80)
    print("CLASSIFIER ACCURACY TEST")
    print("=" * 80)

    for test in all_tests:
        mode, reasoning = classifier.classify(test["query"], test.get("history", []))
        correct = mode == test["expected_mode"]
        results["correct" if correct else "incorrect"] += 1

        status = "PASS" if correct else "FAIL"
        if not correct:
            results["failures"].append(test)
            print(f"\n  [{status}] {test['description']}")
            print(f"    Query:    {test['query']}")
            print(f"    Expected: {test['expected_mode']}")
            print(f"    Got:      {mode}")
            print(f"    Reason:   {reasoning}")
        else:
            print(f"  [{status}] {test['description']}  →  {mode}")

    total = results["correct"] + results["incorrect"]
    pct = results["correct"] / total * 100 if total else 0
    print(f"\n{'=' * 80}")
    print(f"CLASSIFIER RESULTS: {results['correct']}/{total} correct ({pct:.0f}%)")
    print(f"{'=' * 80}")

    return results


def run_full_pipeline_tests(classifier, tutor, verifier):
    """Run select test cases through the full classifier → tutor → verifier pipeline."""

    # Diverse set covering each mode + edge cases
    pipeline_tests = [
        # EXPLAIN — should pass easily
        {
            "query": "How does recursion work in programming?",
            "expected_mode": "EXPLAIN",
            "description": "CS — concept question (expect verifier PASS on first try)",
            "history": [],
        },
        # GUIDE — deliverable with format constraint
        {
            "query": "Describe in 3-4 sentences how the water cycle works.",
            "expected_mode": "GUIDE",
            "description": "Earth science — deliverable with sentence count",
            "history": [],
        },
        # GUIDE — tricky deliverable disguised as explain
        {
            "query": "Explain in a few paragraphs why the Articles of Confederation failed.",
            "expected_mode": "GUIDE",
            "description": "History — 'explain in paragraphs' (format = deliverable)",
            "history": [],
        },
        # CONFIRM — checking work with history
        {
            "query": "I got pH = 2. Right?",
            "expected_mode": "CONFIRM",
            "description": "Chemistry — checking calculation after guidance",
            "history": [
                {"role": "student", "content": "Calculate the pH of a 0.01M HCl solution."},
                {"role": "tutor", "content": "HCl fully dissociates, so [H+] = 0.01M. Now use pH = -log[H+]."},
            ],
        },
        # EXPLAIN — confused by tutor's term (student needs re-explanation)
        {
            "query": "What do you mean by 'conjugate base'? I've never heard that term before.",
            "expected_mode": "EXPLAIN",
            "description": "Chemistry — unfamiliar term used by tutor",
            "history": [
                {"role": "student", "content": "Why is HCl a strong acid?"},
                {"role": "tutor", "content": "HCl is a strong acid because it fully dissociates in water, and its conjugate base Cl- is very stable."},
            ],
        },
        # Mode switch: GUIDE → EXPLAIN
        {
            "query": "Wait, what is a determinant and why do I need it?",
            "expected_mode": "EXPLAIN",
            "description": "Linear algebra — concept question mid-problem (mode switch)",
            "history": [
                {"role": "student", "content": "Find the eigenvalues of [[2, 1], [1, 2]]."},
                {"role": "tutor", "content": "To find eigenvalues, you need to solve det(A - λI) = 0. Can you set up that equation?"},
            ],
        },
        # Mode switch: GUIDE → CONFIRM
        {
            "query": "I rewrote my thesis as: 'The US should transition to renewable energy because fossil fuels drive climate change and harm public health.' Better?",
            "expected_mode": "CONFIRM",
            "description": "Writing — presenting revised thesis after guidance",
            "history": [
                {"role": "student", "content": "Write a thesis statement about climate change policy."},
                {"role": "tutor", "content": "What position do you want to argue?"},
                {"role": "student", "content": "That we should use more renewable energy."},
                {"role": "tutor", "content": "That's a start, but a strong thesis needs a specific claim with reasoning. Why should we use more renewable energy?"},
            ],
        },
        # Tricky edge case: 'list the steps' is a deliverable
        {
            "query": "List the steps to solve a quadratic equation.",
            "expected_mode": "GUIDE",
            "description": "Math — 'list the steps' deliverable request",
            "history": [],
        },
        # REDIRECT mid-problem
        {
            "query": "Is this going to be on the exam?",
            "expected_mode": "REDIRECT",
            "description": "Logistics question mid-lesson",
            "history": [
                {"role": "student", "content": "What is the difference between kinetic and potential energy?"},
                {"role": "tutor", "content": "Kinetic energy is the energy of motion, while potential energy is stored energy based on position or configuration."},
            ],
        },
        # EXPLAIN → GUIDE mode switch
        {
            "query": "Ok I think I get recursion now. Can you help me write a function that computes the nth Fibonacci number?",
            "expected_mode": "GUIDE",
            "description": "CS — switching from concept understanding to code application",
            "history": [
                {"role": "student", "content": "How does recursion work in programming?"},
                {"role": "tutor", "content": "Recursion is when a function calls itself to break a problem into smaller sub-problems. It needs a base case to stop and a recursive case that moves toward it."},
                {"role": "student", "content": "Can you give me an example?"},
                {"role": "tutor", "content": "Think of factorial: 5! = 5 × 4!. The base case is 1! = 1, and each call reduces the number by 1."},
            ],
        },
    ]

    print("\n" + "=" * 80)
    print("FULL PIPELINE TEST")
    print("=" * 80)

    for test in pipeline_tests:
        print(f"\n{'─' * 70}")
        print(f"Test: {test['description']}")
        print(f"Query: {test['query']}")

        # 1. Classify
        mode, reasoning = classifier.classify(test["query"], test.get("history", []))
        mode_correct = mode == test["expected_mode"]
        print(f"  Classifier: {mode} ({'correct' if mode_correct else 'WRONG — expected ' + test['expected_mode']})")
        print(f"  Reasoning: {reasoning}")

        # 2. Tutor + Verifier loop
        tutor_context = TutorContext(
            student_query=test["query"],
            teaching_mode=mode,
            conversation_history=test.get("history", []),
        )

        verified = False
        attempts = 0

        while not verified and attempts < MAX_VERIFIER_RETRIES:
            tutor_response = tutor.respond(tutor_context)
            verifier_context = VerifierContext(
                student_query=test["query"],
                teaching_mode=mode,
                conversation_history=test.get("history", []),
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

        status = "PASSED" if verified else "FAILED ALL RETRIES"
        print(f"  Verifier: {status} (attempts: {attempts})")

        if verified and not mode_correct:
            # Show the accepted response (truncated)
            print(f"  Accepted response: {tutor_response}")
        else:
            print(f"  Last rejection: {result['reason'][:150]}")

    print(f"\n{'=' * 80}")
    print("FULL PIPELINE TEST COMPLETE")
    print(f"{'=' * 80}")


def main():
    if not TEST_MODE:
        print("Test mode is disabled. Set TEST_MODE = True to run tests.")
        return

    print("Initializing LLMs...")
    classifier = ClassifierLLM(model_name=MODEL_NAME)
    tutor = TutorLLM(model_name=MODEL_NAME, max_new_tokens=500)
    verifier = VerifierLLM(model_name=MODEL_NAME)

    # Phase 1: Classifier accuracy
    classifier_results = run_classifier_tests(classifier)

    # Phase 2: Full pipeline on select cases
    run_full_pipeline_tests(classifier, tutor, verifier)


if __name__ == "__main__":
    main()
