# EssayWriter AI

A multi-agent essay writing system built with **LangGraph** and **LangChain**, exposed via a **FastAPI** web API.  
This project demonstrates a modular workflow where multiple AI agents collaborate to generate, critique, and refine essays.

---

## Features

- **Multi-Agent Workflow** with LangGraph:
  - **Planner Agent**: Generates a high-level essay outline based on the user-provided task.
  - **Research Agents**: Use TavilySearch to gather relevant information for both planning and critique stages.
  - **Generator Agent**: Produces the first draft of the essay based on the plan and research content.
  - **Reflection Agent**: Critiques drafts and suggests improvements.
  - **Revision Loop**: Drafts are iteratively refined until the maximum number of revisions is reached.

- **FastAPI Backend**: Exposes a `/api/essay` endpoint that accepts a task and returns the generated essay.
- **SQLite Memory Checkpointing**: Tracks agent state and workflow progress.
- **Configurable Prompts**: Allows easy modification of planning, writing, research, and reflection behavior.

<img width="942" height="457" alt="graph" src="https://github.com/user-attachments/assets/63a0c19a-c7ca-4d81-8f6d-341ac9236270" />

---

## Installation

1. Clone the repository:
  ```bash
    git clone https://github.com/your-username/EssayWriter.git
    cd EssayWriter
2. Create a virtual environment:
  python -m venv myenv
  source myenv/bin/activate  # Linux / Mac
  myenv\Scripts\activate     # Windows

3. Install dependencies:
  pip install -r requirements.txt

4. Create a .env file in the project root and add:
  GOOGLE_API_KEY=your_google_api_key_here
  TAVILY_SEARCH_API_KEY=your_tavily_api_key_here


