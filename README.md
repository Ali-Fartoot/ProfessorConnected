# 🎓 ProfessorConnected: An API for Discovering Similar Professors

## **📚 Overview**

ProfessorConnected is an API-driven system designed to identify professors with similar research interests based on their publications on [arXiv](https://arxiv.org/). The system crawls and processes professors' papers, extracts key information, and uses this data to find related academics by chromaDB vector search.

## **🔍 How It Works**

The core of ProfessorConnected involves a multi-stage process:

1.  ### **🕸️ Crawling**
    The system begins by crawling papers associated with a given professor's name from arXiv.

2.  ### **📄 Paper Processing**
    Each paper undergoes detailed analysis, focusing on key sections:
    
    - **📝 Title Analysis:** The title is extracted and processed using the arXiv API.
    - **📖 Text Analysis:** The introduction and conclusion sections are summarized using a Large Language Model (LLM), specifically **Mistral-Nemo 14B** by default.
    - **🔑 Keyword Extraction:** Keywords are extracted using three methods:
        - **🧩 N-gram Based:** Identifies keywords based on the frequency of word sequences.
        - **🤖 BERT Based:** Uses a BERT model to extract contextual keywords.
        - **🧠 LLM Based:** Leverages an LLM to extract relevant keywords.
    - **🔄 LLM Revision:** All extracted keywords are refined and finalized using an LLM.

3.  ### **📈 Embedding**
    The processed title, summaries, and revised keywords from each paper are concatenated and embedded to create a vector representation.

4.  ### **🔗 Hybrid Search**
    When a user searches for a professor, the system performs a hybrid search:
    
    - **🔍 Retrieval:** Retrieves potentially related professors.
    - **📊 Ranking:** Ranks them based on the cosine similarity of their embeddings.
    - **🔄 Re-ranking:** Re-ranks them by adding a score based on the overlap of statics keywords between the query and the retrieved professor's papers.

## **🖼️ Diagram of Embedding Process**

```mermaid
graph LR
    A[Input: Professor Name] --> B[arXiv Crawler]
    q[Query] --> K
    B --> C[Extract Data]
    
    subgraph "Paper Processing"
        C -->|Title| D[Title Analysis]
        C -->|Figures Caption| O[Figures Caption]
        C -->|Introduction, Conclusion| E[Summarizer]
        C -->|Introduction, Conclusion| F[Keyword Extraction]
        
        subgraph "Keyword Methods"
            O --> F
            F --> G[N-gram]
            F --> H[BERT]
            F --> I[LLM]
            G & H & I --> J[LLM Revision]
        end
    end
    
    D & J & E --> L[Paper Embeddings]
    
    J --> K[Calculate Keywords Score]
    L --> M[Vector Database]
    q --> M
    K -->|Re-rank| M
    M -->|Retrived after Re-rank| N[Hybrid Search]
```

## **🚀 Getting Started**

Follow these steps to set up and run ProfessorConnected:

1.  ### **🧩 LLM Model Download**
    Download a GGUF format LLM model and place it in the `models` directory.

2.  ### **⚙️ Environment Setup**
    - **Create a virtual environment:**  
      ```bash
      python3 -m venv venv
      ```
    - **Install dependencies:**  
      ```bash
      make install
      ```
    - **Run the server:**  
      ```bash
      make run
      ```  
      This will start both the FastAPI server on port `8000` and the `llm-cpp-server` on port `5333`.
    - **💻 GPU Acceleration:**  
      To use GPU acceleration (half of the model layers on the GPU), run:  
      ```bash
      make run USE_GPU=true
      ```

3.  ### **🗃️ Populate the Database**
    - The vector database is initially empty.
    - Run `cold_start.py` or `make cold-start` to add initial professor data to the database.

4.  ### **🛠️ Using the API**
    - The `scripts` folder contains scripts for various operations:
        - **➕ Adding a professor**
        - **🧹 Cleaning the dataset**
        - **🔍 Finding similar professors and visualizing the results**
    - You can modify the `curl` request parameters to customize your search:
        - `limit`: The number of professors to retrieve.
        - `min_similarity`: The minimum cosine similarity threshold for retrieving results.
        - **⚠️ Note:** Increasing the number of professors in the database may require adjusting the `min_similarity` threshold for more accurate results.
    - **⚡ Quick Test:** Use `make image-request` to quickly test the API and get visualized retrived professors as image.

## **Input request**
By running app by `make run` now you have three operations.

- Add: 
  ```bash
    #!/bin/bash
    # Add professor
    curl -X POST "http://localhost:8000/add_professor" \
         -H "Content-Type: application/json" \
         -d '{"professor_name": "A. Barresi", "number_of_articles": 3}'

  ```
- Delete: 
  ```bash
    #!/bin/bash
    # Cleanup database
    curl -X DELETE "http://localhost:8000/cleanup_database" 

  ```
- search:
  ```bash
    #!/bin/bash
    # Search without visualization
    curl -X POST "http://localhost:8000/search" \
        -H "Content-Type: application/json" \
        -d '{"professor_name": "Sergey Levine", "min_similarity": 0.1, "limit": 5}'

  ```

This tool is designed to collect and process articles based on specified parameters. The following parameters are available for customization:

- **`professor_name`**: The name of the professor to search for.
- **`number_of_articles`**: The number of articles to collect and process.
- **`min_similarity`**: The minimum cosine similarity score (ranging between 0 and 1) for filtering articles.
- **`limit`**: The maximum number of professors to retrieve data for.

By adjusting these parameters, you can tailor the tool to meet your specific research or data collection needs.

## **🖼️ Example Result**

The following image shows the result of a search for "Sergey Levine" after running the cold start mode:

![alt text](./examples/visualization.png)


The gist of three papers from **Manuel Cebrian**. (The whole papers will concate and embed together.)

```json
{
    "Manuel Cebrian": [
        {
            "title": "Mobilizing Waldo_ Evaluating Multimodal AI for Public Mobilization",
            "summary": "The Cambridge Analytica scandal and foreign misinformation campaigns highlight the influence of AI on public opinion, privacy, and democracy. This study evaluates GPT-4o's capabilities in analyzing complex social scenes using 'Where's Waldo?' images as proxies for real-world gatherings. Despite its strong language generation abilities, the model struggles with spatial reasoning and character identification, limiting its effectiveness in strategic planning. The findings underscore the need for advancements in multimodal integration and ethical guidelines to harness AI’s potential while mitigating risks.",
            "keywords": "AI IN PUBLIC ENGAGEMENT, SOCIAL MOBILIZATION, IMAGE"
        },
        {
            "title": "Conversational Complexity for Assessing Risk in Large Language Models",
            "summary": "The rapid development of Large Language Models (LLMs) has introduced new challenges, particularly in ensuring they don't produce harmful or unethical content. A key issue is that these systems often require multi-turn interactions to bypass safety measures and generate problematic outputs. This complexity necessitates the introduction of novel metrics like Conversational Length and Complexity for risk assessment. These measures can quantify how easily harmful outcomes are elicited, aiding in more precise LLM safety evaluations compared to traditional methods like red teaming. Algorithmic information theory provides the theoretical foundation for these concepts, offering a new perspective on conversational dynamics within LLMs. The paper explores this framework through empirical analysis of specific conversations and a large dataset from red-teaming efforts, aiming to improve our understanding and mitigation of LLM risks.",
            "keywords": "CONVERSATIONS, COMMUNICATION STRATEGY, LANGUAGE MODE"
        },
        {
            "title": "Supervision policies can shape long-term risk management in   general-purpose AI models",
            "summary": "The rapid adoption of Generative Pre-trained Transformer (GPT) models and other Large Language Models (LLMs) has introduced both opportunities and challenges, particularly in cybersecurity, bias, privacy, misinformation, and harmful content generation. To address these risks, an ecosystem of risk reporting mechanisms, including community-driven platforms like Reddit and expert-led assessments, has emerged.\n\nCommunity involvement helps identify diverse perspectives on emerging issues, while crowdsourcing initiatives test AI systems for vulnerabilities. For instance, DEF CON's Generative AI Red Team events engage thousands to evaluate models from leading AI organizations. OpenAI's Preparedness Challenge similarly invites public input on potential risks, offering incentives like API credits.\n\nSupervisory entities will need efficient strategies to prioritize and address the growing volume of risk reports on GPAI models. Our study explores how these bodies can process and prioritize incoming reports using a simulation framework that examines various supervision policies: non-prioritised, random, priority-based, and diversity-prioritised.\n\nOur findings reveal that priority-based strategies effectively mitigate high-impact risks but might overlook community-driven insights. Diversity-prioritized approaches offer a balanced coverage but still prioritize significant issues. These dynamics highlight the need for inclusive risk management practices to ensure comprehensive oversight and avoid marginalizing user perspectives.\n\nUltimately, our study underscores the importance of developing effective, ethically sound supervision policies for GPAI models, balancing high-impact risk mitigation with the inclusion of diverse community inputs.",
            "keywords": "CROWDSOURCING, AI RISK MANAGEMENT"
        }
    ]
}
```

## **🧪 Testing Strategy**

ProfessorConnected is rigorously tested using two methods:

- **🔧 Component Testing:**  
  Each individual component of the system (as shown in the diagram) is tested in isolation to ensure correct functionality. The following processor are tested:
   - Crawler
   - key extractor
   - llm summarizer
   - visuliazer
   - Vector Search

- **🔄 End-to-End Testing:**  
  The entire system for each operation (ADD, DELETE, VISUALIZE(SEARCH)) is tested by sending a request and verifying the final output. 

- **✅ Test Execution:**  
  Run `make test` to execute all tests, covering both component and end-to-end testing.

## **📅 Project Timeline**

The project was developed over **15 days**:

- **🛠️ 3 days:** Architecture design and library selection.
- **🚀 7 days:** Deployment.
- **🧪 3 days:** Writing tests.
- **📄 2 days:** Documentation.

