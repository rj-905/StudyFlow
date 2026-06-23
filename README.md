#  **StudyFlow: Multi-Agent Personalized Learning Companion**

**StudyFlow** is a stateful, multi-agent AI system designed to act as a highly personalized learning companion. By leveraging advanced agent orchestration and hybrid execution, StudyFlow automates the end-to-end learning lifecycle: from curriculum planning and content curation to synthesis and mathematically rigorous spaced-repetition assessment.

This project was built as a capstone submission, demonstrating complex Agent-to-Agent (A2A) communication, vector-based semantic memory (RAG), and dynamic model routing.

##  **Key Features**

*  **Multi-Agent Orchestration:** A specialized team of AI agents working together via a LangGraph state machine.  
*  **Hybrid Execution Model:** Balances sequential reasoning (planning) with parallel task execution (concurrently scraping video and academic texts) for maximum speed.  
*  **Intelligent Spaced Repetition (SM-2):** Integrates the proven Ebbinghaus SuperMemo-2 algorithm to dynamically schedule quiz reviews based on user performance.  
*  **Semantic Memory (RAG):** Uses ChromaDB and text-embedding-004 to give agents long-term contextual awareness of past study sessions.  
*  **Dynamic Model Routing:** Cost-optimized LLM routing between heavy reasoners (gemini-2.5-pro) and high-speed classifiers (gemini-3.5-flash).  
*  **Interactive CLI & Automated Delivery:** Real-time asynchronous streaming in the terminal, with automated hand-off of Markdown study guides to the user's local system.

##  **System Architecture & Execution Flow**

StudyFlow operates on a **hybrid execution** LangGraph model to ensure optimal context passing without sacrificing speed:

1. **Orchestration (Sequential):** The system evaluates the user's requested topic against known prerequisites.  
2. **Parallel Sourcing (Fan-Out):** Once validated, multiple agents (Lecture, Reading) are spawned concurrently to scour different media types without blocking each other.  
3. **Synthesis (Fan-In/Sequential):** Sourced context is aggregated and funneled into a specialized Notes Agent to create a unified study guide.  
4. **Assessment (Sequential):** A Quiz Agent evaluates the synthesized notes to generate baseline questions and schedule future reviews.

##  **The Multi-Agent Ecosystem**

StudyFlow separates concerns into specialized personas, each equipped with specific tools for its domain:

*  **The Orchestrator:** The brain of the operation. Analyzes the target topic, identifies knowledge prerequisites, and formulates a structured study plan. Acts as the gatekeeper.  
*  **The Lecture Agent:** Visual and auditory content curator. Searches YouTube for relevant video lectures. Scores, ranks, and retrieves the highest-quality educational video content.  
*  **The Reading Agent:** Academic content curator. Scours the web for academic papers, articles, and textbook excerpts, filtering for credibility and relevance.  
*  **The Notes Agent:** The Synthesizer. Ingests raw, unstructured data gathered by sourcing agents and distills it into a comprehensive, beautifully formatted Markdown study guide using RAG.  
*  **The Quiz Agent:** The Assessor. Generates an initial assessment quiz immediately after synthesis and runs interactive, terminal-based review sessions when the user is due for practice.

##  **Memory & Context Management**

StudyFlow features a dual-memory system to combat AI amnesia and track student progress:

* **Relational State (SQLite):** Tracks topics, SM-2 scheduling variables (Ease Factor, Interval, Repetitions), quiz attempts, and user performance over time.  
* **Semantic Memory (ChromaDB):** Acts as a vector store for embeddings. Allows the Notes Agent to perform RAG against past study guides, explicitly weaving prior knowledge into new summaries.

##  **Dynamic Model Routing**

To optimize speed, cost, and rate limits, StudyFlow implements a centralized model routing mechanism based on computational weight:

* TaskWeight.HEAVY **(gemini-2.5-pro)**: Used for complex reasoning (Orchestrator planning, Notes synthesis, Quiz generation).  
* TaskWeight.LIGHT **(gemini-3.5-flash)**: Used for lightweight, high-speed tasks (grading user input, ranking search results, classification).  
* TaskWeight.EMBEDDING **(text-embedding-004)**: Dedicated to converting text into high-dimensional semantic vectors.

##  **Getting Started**

### **Prerequisites**

* Python 3.11+  
* Gemini API Key
* YouTube Data API v3 Key
* Google Custom Search JSON API


### **Installation**

1. **Clone the repository:**  
   git clone \[https://github.com/rj-905/studyflow.git\](https://github.com/rj-905/studyflow.git)  
   cd studyflow

2. **Set up a virtual environment:**  
   python \-m venv venv  
   source venv/bin/activate  \# On Windows use \`venv\\Scripts\\activate\`

3. **Install dependencies:**  
   pip install \-r requirements.txt

4. **Configure Environment Variables:**  
   * Copy the example .env file: cp .env.example .env  
   * Open .env and add your Google Gemini API Key and other necessary search API credentials.

### **Usage**

Run the interactive CLI companion:

python -m studyflow.py

Follow the terminal prompts to input a new study topic or initiate an SM-2 scheduled review session\!

*Built for the AI Agents Capstone Project.*
