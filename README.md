# 📄 Semantic Search System for Transcript Q&A

A powerful semantic search system for timestamped transcripts that combines traditional keyword search with advanced AI-powered semantic understanding. Ask questions about your transcripts and get precise answers with relevant context.

## 🌟 Features

### Multiple Search Technologies
- **TF-IDF (Keyword Search)**: Fast, traditional keyword-based search
- **Sentence Transformers**: Advanced semantic search using `all-MiniLM-L6-v2` model  
- **Google Gemini AI**: State-of-the-art semantic search with AI-generated answers

### Dual Interface Options
- **🖥️ Command Line Interface**: Interactive terminal-based searching
- **🌐 Web Application**: User-friendly Streamlit interface with file uploads

### Smart Features
- **Intelligent Chunking**: Automatically groups transcript lines for optimal search
- **Timestamp Preservation**: Maintains original timing information for easy reference
- **Similarity Scoring**: Shows relevance scores for search results
- **Context-Aware Answers**: AI generates concise answers from retrieved transcript segments
- **Real-time Mode Switching**: Switch between search methods without restarting

## 📂 Project Structure

```
transcript-qa-search/
├── Output Screenshot/         # Screenshots of application outputs and demos
│   ├── cli_output/           # Command line interface screenshots
│   └── web_app_output/       # Web application screenshots
├── web_app/                   # Streamlit web application directory
│   ├── transcript_qna_web.py  # Web interface main script
│   └── .streamlit/            # Streamlit configuration
│       └── secrets.toml       # API keys and configuration secrets
├── .gitignore                 # Git ignore file
├── output.txt                 # Sample CLI output results and examples
├── README.md                  # Project documentation (this file)
├── requirements.txt           # Python dependencies
├── transcript.py              # CLI application
├── transcript.txt             # Sample timestamped transcript
└── transcript_doc.md.pdf      # Detailed documentation and design decisions
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- (Optional) Google Gemini API key for advanced AI features

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/amit11ki/Semantic-Search-System-for-Transcript-Q-A
cd Semantic-Search-System-for-Transcript-Q-A
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up Gemini API (Optional)**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

### Usage

#### 🖥️ Command Line Interface

**TF-IDF keyword search**
```bash
python transcript.py transcript.txt tfidf
```
**Semantic search with Sentence Transformers**
```bash
python transcript.py transcript.txt llm2
```
**AI-powered search with Gemini**
```bash
python transcript.py transcript.txt llm1
```

**Interactive Commands:**
- Type your question and press Enter
- `switch to tfidf` / `switch to llm1` / `switch to llm2` - Change search mode
- `8` - Exit application

#### 🌐 Web Interface

```bash
streamlit run web_app/transcript_qna_web.py
```

Then open your browser and:
1. Upload your transcript file (.txt format)
2. Configure search parameters in the sidebar
3. Enter your question and click "🔍 Search Transcript"

## 📋 Transcript Format

Your transcript should be in this timestamped format:

```
[00:12 - 00:17]  These are the terms which have confused a lot of people and if you two are one among them,
[00:17 - 00:19]  let me resolve it for you.
[00:19 - 00:24]  Well artificial intelligence is a broader umbrella under which machine learning and deep learning
```

**Format Requirements:**
- Timestamps in `[MM:SS - MM:SS]` or `[HH:MM:SS - HH:MM:SS]` format
- Each line should contain one timestamped segment
- Plain text encoding (UTF-8 preferred)

## 🔧 Configuration Options

### Web Interface Settings
- **Lines per chunk**: Group 1-15 consecutive lines (default: 5)
- **Top-K results**: Display 1-10 most relevant chunks (default: 3)
- **Search mode**: Choose from TF-IDF, Semantic, or AI-powered search

### Search Modes Explained

| Mode | Technology | Speed | Accuracy | AI Answer |
|------|-----------|-------|----------|-----------|
| **TF-IDF** | Keyword matching | ⚡ Fast | Good for exact terms | ❌ |
| **Sentence Transformer** | Semantic embeddings | 🔄 Medium | Understands context | ❌ |
| **Gemini AI** | Advanced embeddings + LLM | 🐌 Slower | Best semantic understanding | ✅ |

## 💡 Example Queries

- "What was discussed about the project timeline?"
- "Tell me about budget concerns"
- "What were the main action items?"
- "Who mentioned the quarterly results?"

## 🛠️ Technical Details

### Dependencies
- **Core**: `numpy`, `scikit-learn`, `sentence-transformers`
- **Web UI**: `streamlit`
- **AI Integration**: `google-generativeai`
- **Text Processing**: `re` (built-in)

### Performance Considerations
- **Chunking**: Larger chunks provide more context but may reduce precision
- **Embeddings**: Gemini embeddings are cached to avoid repeated API calls
- **Memory**: Sentence Transformer models are cached for efficiency

### Error Handling
- Graceful fallback when API keys are missing
- Robust file encoding detection (UTF-8/Latin-1)
- Input validation for malformed timestamps

## 🔒 Privacy & Security

- **Local Processing**: TF-IDF and Sentence Transformer modes run entirely locally
- **API Usage**: Gemini mode sends text to Google's API for processing
- **No Storage**: No transcript data is permanently stored by the application

## 📚 Documentation & Examples

- **📋 Process Documentation**: `transcript_doc.md.pdf` contains detailed design decisions, implementation details, and technical specifications
- **📄 Text Output Examples**: `output.txt` contains sample CLI results showing actual queries and responses from all three search modes
- **📸 CLI Examples**: `Output Screenshot/cli_output/` folder contains screenshots of command-line interface usage and results
- **🌐 Web Interface Examples**: `Output Screenshot/web_app_output/` folder contains screenshots of the Streamlit web application
- **🔧 Sample Data**: `transcript.txt` provides a working example of properly formatted transcript data
- **⚙️ Configuration**: `web_app/.streamlit/secrets.toml` for API key configuration (Streamlit apps)

## 🐛 Troubleshooting

### Common Issues

**"No valid chunks found"**
- Check transcript format matches the expected timestamp pattern
- Ensure file is not empty and properly encoded

**Gemini mode not working**
- Verify `GEMINI_API_KEY` environment variable is set
- Check API key has proper permissions
- Ensure internet connection is available

**Slow performance**
- Consider reducing chunk size for faster processing
- Use TF-IDF mode for quick keyword searches
- Check available system memory for large transcripts

## 📊 Example Output

```
Top relevant chunks:
1. [10:06 - 10:22]  data. This is a very distinctive part of deep learning, which makes it way ahead of traditional machine learning. Deep learning reduces the task of developing new feature extractor for every problem. Like in the case of CNN algorithm, it first try to learn the low level features of the
2. [11:00 - 11:17]  Let's take an example to understand this. Suppose you have a task of multiple object detection and your task is to identify what is the object and where it is present in the image. So let's see and compare how will you tackle this issue using the concept of machine learning and deep learning.
3. [13:55 - 14:12]  Machine learning uses algorithm to parse the data, learn from the data and make informed decision based on what it has learned. Fine. Now this deep learning structures algorithms and layers to create artificial neural network that can learn and make intelligent decisions on their own.

Generating answer with Gemini...

Answer: Deep learning uses structured algorithms and layers to create artificial neural networks that learn and make decisions independently, unlike traditional machine learning which relies on pre-defined feature extractors.  This allows deep learning to handle tasks like object detection more effectively.
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Sentence Transformers**: For providing excellent semantic search capabilities
- **Google Gemini**: For advanced AI-powered search and answer generation
- **Streamlit**: For the intuitive web interface framework

---

**Built with ❤️ for better transcript analysis and question answering**
- **Google Gemini**: For advanced AI-powered search and answer generation
- **Streamlit**: For the intuitive web interface framework

---

**Built with ❤️ for better transcript analysis and question answering**
