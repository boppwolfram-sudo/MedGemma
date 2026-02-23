### Project name

DIRECT: Deterministic & Cost-Aware Clinical Triage using MedGemma 4B

### Your team

[Your Name/Team Member 1] - [Specialty, e.g., Machine Learning Engineer] - [Role, e.g., AI Architecture & Prompt Engineering]

[Team Member 2, if applicable] - [Specialty, e.g., Clinical Informatician] - [Role, e.g., Clinical Pathway Design]

### Problem statement

The integration of generative AI into clinical triage is currently bottlenecked by two critical flaws: diagnostic hallucination and financial inefficiency.

Instruction-tuned Large Language Models (LLMs) are optimized for conversational verbosity. When presented with clinical vignettes, their propensity to generate sprawling differential diagnoses rather than discrete, machine-parseable conclusions renders them dangerous for autonomous downstream routing. Furthermore, standard LLMs lack inherent Cost-Awareness. They routinely recommend high-cost, high-friction diagnostic tests (like MRIs) before exhausting low-cost, high-value symptom inquiries.

Simultaneously, human physicians face severe time constraints that limit their ability to conduct exhaustive patient histories. This time deficit compromises diagnostic accuracy and exacerbates physician burnout. If we can solve the AI hallucination problem, we can automate the exhaustive questioning phase, reclaiming physician time, reducing over-testing, and mitigating the malpractice liability associated with AI misdiagnosis.

### Overall solution:

Our solution, DIRECT (Deterministic Inference Routing Engine for Clinical Triage), transforms MedGemma 1.5 4B-IT from a conversational agent into a deterministic, Cost-Aware Clinical Triage backend.

DIRECT operates as the mathematical "brain" behind a hospital's Digital Front Door app. While a standard UI gathers the patient's chief complaint, DIRECT runs silently in the background. To bypass conversational biases, we discretize diagnoses into strict Multiple-Choice Questions (MCQs) mapped to a decision tree. By using explicit formatting constraints at the end of the prompt, MedGemma evaluates the clinical context and outputs a single, `<answer>`-tagged classification token.

If DIRECT achieves high confidence, it immediately routes the patient to the safest care setting (e.g., ER vs. telehealth). If the engine is uncertain, it calculates the Expected Information Gain (EIG) to determine the single most statistically valuable, low-cost question the UI should ask the patient next. This mathematically mimics a senior clinician's reasoning. Finally, the results are presented to the physician via a visual dashboard showing exact probabilities, entirely bypassing the need to read dense, AI-generated paragraphs.

### Technical details

Our application establishes clinical safety and product feasibility by bypassing "black-box" text generation in favor of raw mathematical extraction and cost-sensitive heuristics.

**Deterministic Decoding (HAI-DEF):** To guarantee reliable routing, the MedGemma inference engine is configured statelessly with Greedy Decoding (`do_sample=False`) and physical token limits (`max_new_tokens=10`). Output is extracted via Regex, safely bridging the semantic language model with our visual dashboard.

**Multimodal Integration (MedSigLIP):** When triage requires visual data (e.g., a patient uploads a photo of a dermatological lesion or visible inflammation), the images are preprocessed via the MedSigLIP encoder. The rigid text prompt anchors these dense visual embeddings back to the discrete MCQ options, preventing the vision-encoder from generating unconstrained captions.

**Logit Calibration & The Safety Threshold:** At each node, raw output logits ($z$) are extracted and scaled via Temperature ($T > 1$) to penalize inherent LLM overconfidence using the formula: $P(y_i | x) = \frac{\exp(z_i / T)}{\sum_{j} \exp(z_j / T)}$. If the winning diagnostic branch fails to clear a strict confidence threshold ($P < 0.70$), the system immediately halts the automated triage.

**Expected Information Gain (EIG):** When confidence falls below the threshold, the engine calculates the expected reduction in Shannon Entropy ($H(S) = - \sum P(x_i) \log_2 P(x_i)$) against the direct financial/temporal cost of candidate tests. The system explicitly selects the next step based on optimal utility ($EIG = H(S_{current}) - \mathbb{E}[H(S_{next})]$), ensuring expensive procedures are only recommended when low-cost queries are exhausted.

**Intra-Visit Backtracking:** A mandatory "Contradiction/None of the Above" token is included at every node. If new patient inputs contradict the current pathway, this token's probability spikes, triggering an immediate algorithmic backtrack to prevent misdiagnosis.

**Edge Deployment Ready:** Because the system operates via rapid, single-pass classification turns on the lightweight 4-billion-parameter MedGemma architecture, it is highly compute-efficient. It functions without continuous cloud API calls, allowing for offline deployment on standard hospital hardware (e.g., 4-bit quantized weights) while automatically ensuring HIPAA/GDPR data privacy compliance.
