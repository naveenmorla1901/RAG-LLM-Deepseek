# Intelligent Loan Document Processing System

## Project Overview

This system implements an intelligent loan document processing assistant that helps analyze loan applications, policy documents, and historical loan data using RAG (Retrieval-Augmented Generation) with AWS Bedrock and ChromaDB.


### Core Features
- Document Processing (PDF, CSV)
- Semantic Search with AWS Bedrock Embeddings
- Vector Storage with ChromaDB
- Intelligent Question Answering with Multiple LLM Support
- Source Reference Tracking
- Fallback Responses for Connection Issues
- Progress Tracking and Caching
- Enhanced Error Handling

## Architecture

### Components
1. **Document Processor**
   - Handles PDF and CSV files
   - Splits documents into chunks
   - Generates embeddings using AWS Bedrock
   - Stores in ChromaDB
   - Added caching for better performance
   - Progress tracking for large files

2. **Query Engine**
   - Semantic search using Bedrock embeddings
   - Multi-LLM support (DeepSeek, GPT-4, Claude)
   - Source tracking and reference management
   - Fallback mechanisms for connection issues
   - Automatic retries with exponential backoff

3. **Vector Database**
   - Uses ChromaDB for vector storage
   - Supports incremental updates
   - Persistent storage
   - Enhanced error recovery

4. **Response System**
   - Clear distinction between document-based and model knowledge
   - Formatted output with source references
   - Backup responses for connection issues
   - Progress indicators for long operations

## Project Structure
```
loan_processor/
├── data/
│   └── raw/
│       ├── loan_applications/    # PDF loan applications
│       ├── policy_documents/     # PDF policy documents
│       └── training_data/        # CSV historical data
├── get_embedding_function.py     # AWS Bedrock embeddings
├── process_documents.py          # Document processing
├── query_data.py                # Query interface
├── config.py                    # Configuration
└── requirements.txt             # Dependencies
```
## Setup Instructions

### Prerequisites
- Python 3.9+
- AWS Account with Bedrock Access
- One or more of the following API keys:
  - DeepSeek API Key
  - OpenAI API Key (optional for GPT-4)
  - Anthropic API Key (optional for Claude)

### Installation

1. Create Virtual Environment:
```bash
conda create -n loan_processor python=3.9
conda activate loan_processor
```

2. Install Requirements:
```bash
pip install -r requirements.txt
```

3. Configure Environment Variables:
Create a `.env` file with:
```env
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-east-1
DEEPSEEK_API_KEY=your_deepseek_key
OPENAI_API_KEY=your_openai_key    # Optional
ANTHROPIC_API_KEY=your_claude_key # Optional
```

### AWS Bedrock Setup

1. Enable AWS Bedrock in your AWS account
2. Request access to "Titan Embeddings G1 - Text" model
3. Create IAM user with required permissions:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:ListFoundationModels"
            ],
            "Resource": "*"
        }
    ]
}
```
## Usage

### Running the System

Start the interactive interface:
```bash
# Using default DeepSeek model
python query_data.py

# Or specify a different model
python query_data.py --model gpt-4
python query_data.py --model claude-3
```

### Available Commands
- `/help` - Show help message
- `/analyze <id>` - Analyze specific application
- `/files` - List processed documents
- `/ingest <path>` - Process new document
- `/clear` - Clear screen
- `/exit` - Exit program

### Response Types

1. **Document-Based Response (📄)**
```
📄 DOCUMENT-BASED RESPONSE
Answer derived from analyzed documents
[Answer content]
[Document references]
```

2. **Model Knowledge Response (ℹ)**
```
ℹ MODEL KNOWLEDGE RESPONSE
No relevant documents found - Using model's base knowledge
[General knowledge answer]
```

3. **System Backup Response (⚠)**
```
⚠ SYSTEM BACKUP RESPONSE
Using built-in fallback knowledge
[Backup response content]
```

[Previous Features in Detail section remains the same]

## Enhanced Features

### Error Handling
- Automatic retries for API calls
- Exponential backoff strategy
- Fallback responses for connection issues
- Clear error messaging

### Performance Improvements
- Document processing progress tracking
- Embedding caching
- Batch processing optimization
- Response streaming support

### Multiple LLM Support
- DeepSeek Chat (default)
- GPT-4 (optional)
- Claude-3 (optional)
- Automatic fallback chain

### Response Enhancement
- Clear source attribution
- Formatted references
- Progress indicators
- Rich terminal interface

[Previous Troubleshooting section with additions:]

### Additional Troubleshooting

4. LLM Connection:
```
Error: Connection error with LLM service
Solution: System provides backup response, check API keys and connection
```

5. Progress Tracking:
```
Error: Process hanging without progress
Solution: Check terminal support for rich display features
```

[Rest of the sections remain the same]

## Dependencies
[Previous dependencies plus:]
- rich>=13.7.0
- tenacity>=8.2.3
- aiohttp>=3.9.1
- typing-extensions>=4.9.0
- colorama>=0.4.6
- watchdog>=3.0.0

## License
This project is licensed under the MIT License.