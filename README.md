# Sal — The AI Tutor

> *A tutor, not a do-er.*

Sal is a multi-component AI tutoring pipeline built for classroom integration. Rather than completing work for students, Sal guides them through problems using questions, hints, and course-specific material — the way a real tutor would.

---

## Overview

This repository contains the full source code for Sal, including:

- **The 4-stage tutoring pipeline** — Classifier, Context Object, Tutor LLM, and Verifier LLM
- **A RAG system** — ingests and retrieves course materials (PDFs, PowerPoints, handwritten notes) using MiniLM-L6-v2 + FAISS
- **A handwriting recognition model** *(see note below)*

The codebase is intended for students, professors, or researchers who want to run, test, or extend Sal in their own environment.

`pipeline.py` is the main entry point. It wires together all four components and manages the Tutor–Verifier retry loop until a passing response is produced and returned to the student.

---

## Running Instructions

### 1. Set Up Virtual Environment

```bash
# Create the virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Activate it (Mac/Linux)
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Sal Pipeline

```bash
python pipeline.py
```
*RAG is disabbled in this code, because loading in the default class information takes about 2 minutes on a teaching node. If you want to enable RAG, change the ENABLE_RAG boolean in pipeline.py*

*If you want a speed increase, you can request a GPU. You would also need to go into the requirements.txt and change faiss-cpu to faiss-gpu*


### 3. View Sal in Web Browser
    Use the local host link provided in the output of running pipeline.py

---

## Example Output

Screenshots of example interactions are located in the `example_outputs/` folder.

---

## Submission Elements

| Item | Link |
|------|------|
| Video | [Watch here]() |
| Single Page Overview | [View here](https://msoe365-my.sharepoint.com/:w:/g/personal/sterritts_msoe_edu/IQB0Gx7qrUEjQrTW_3r-SKExAW8id207yrkjAMAr0ZgpBoM?e=udPaIf)|

---

## Handwriting Model

> **Note:** This version of Sal has the handwriting recognition model dissabled. It significantly increases execution time, and we wanted this code to be easily runnable for anyone who wants to try Sal.