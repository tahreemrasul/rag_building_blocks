# Retrieval-Augmented Generation (RAG) Pipeline

This project demonstrates how to build a Retrieval-Augmented Generation (RAG) pipeline using open-source tools. The pipeline includes **Index Creation**, **Retrieval**, and **Generation** to create a queryable knowledge system.

---

## Prerequisites
- Python 3.8 or higher
- Required Libraries:
  - `langchain`
  - `dotenv`
  - `sentence-transformers`
  - `chromadb`

---

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/rag-pipeline.git
   cd rag-pipeline```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Add Input File**: Place your input file (PDF) in the data/ directory.

## File Structure

├── data/
│   └── sample-document.pdf  # Add your document here
├── rag_building_blocks.py   # RAG pipeline implementation
├── requirements.txt         # List of dependencies
└── README.md                # Project documentation

## Usage
Run the RAG pipeline with the following command:
```bash
python rag_building_blocks.py
```
## License
This project is licensed under the `MIT` License. See the LICENSE file for more details.
