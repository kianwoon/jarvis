# Jarvis - AI Assistant Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-75.6%25-blue" alt="Python">
  <img src="https://img.shields.io/badge/TypeScript-22.3%25-blue" alt="TypeScript">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/FastAPI-0.104%2B-009688" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-19.1.4-61DAFB" alt="React">
</p>

Jarvis is an advanced AI assistant platform that combines multiple AI technologies to provide intelligent conversational capabilities with knowledge graph integration, document processing, and comprehensive observability.

## 🌟 Features

- **Multi-Model LLM Platform**: Integration with various language models for flexible AI responses
- **Knowledge Graph**: Neo4j-powered knowledge management for structured information
- **Vector Search**: Qdrant-based semantic search for intelligent document retrieval
- **Document Processing**: Support for PDFs, Word, PowerPoint, Excel, and images with OCR
- **MCP Integration**: Model Context Protocol for extensible tool integrations
- **Gmail Integration**: Email processing and management through MCP
- **Observability**: Langfuse integration for comprehensive analytics and monitoring
- **Caching**: Redis-powered caching for performance optimization
- **Analytics**: ClickHouse for deep analytics insights
- **Custom Workflows**: LangGraph-based workflow orchestration

## 🏗️ Architecture

Jarvis is built with a microservices architecture using Docker Compose:

### Core Services

- **App**: FastAPI backend server
- **PostgreSQL**: Primary database for user data, conversations, and workflows
- **Qdrant**: Vector database for semantic search
- **Redis**: Caching and session management
- **Neo4j**: Knowledge graph for structured relationships
- **ClickHouse**: Analytics storage for Langfuse
- **Langfuse**: Observability and monitoring platform
- **Qwen Embedder**: Custom embedding service
- **MCP Gmail**: Email integration service

### Tech Stack

**Backend:**
- FastAPI - Modern Python web framework
- LangChain & LangGraph - AI workflow orchestration
- Transformers & PyTorch - ML model integration
- PostgreSQL (psycopg2) - Relational database
- Neo4j - Graph database
- Qdrant - Vector database
- Redis - Caching

**Frontend:**
- React 19.1+ - UI framework
- Material-UI 7.1+ - Component library
- TypeScript - Type-safe development
- MCP SDK - Model Context Protocol integration

**AI/ML:**
- LLaMA Index - Document indexing
- Sentence Transformers - Text embeddings
- Qwen Agent - MCP integration
- Langfuse - Observability

## 📋 Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Node.js 18+
- 8GB+ RAM recommended

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/kianwoon/jarvis.git
cd jarvis
```

### 2. Configure Environment Variables

Copy the example environment file and configure it:

```bash
cp .env.docker .env
```

Edit `.env` with your configuration (refer to `.env.example` for available options).

Key environment variables:
```bash
# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=llm_platform

# Neo4j
NEO4J_PASSWORD=jarvis_neo4j_password

# Langfuse
NEXTAUTH_SECRET=your-secret-key
SALT=your-salt

# ClickHouse
CLICKHOUSE_PASSWORD=clickhouse_password
```

### 3. Start Services

```bash
docker-compose up -d
```

This will start all services including:
- FastAPI backend (port 8000)
- PostgreSQL (port 5432)
- Qdrant (port 6333)
- Redis (port 6379)
- Neo4j (port 7474 for browser, 7687 for bolt)
- ClickHouse (port 8123)
- Langfuse Web (port 3000)
- Qwen Embedder (port 8050)

### 4. Run Database Migrations

```bash
docker-compose exec app python run_migration.py
```

### 5. Access the Application

- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Langfuse Dashboard**: http://localhost:3000
- **Neo4j Browser**: http://localhost:7474

## 🔧 Development

### Backend Development

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development

The frontend is located in the `llm-ui` directory.

```bash
cd llm-ui
npm install
npm run dev
```

### Running Tests

```bash
# Python tests
pytest tests/

# With coverage
pytest --cov=app tests/
```

## 📁 Project Structure

```
jarvis/
├── app/                    # FastAPI application
│   ├── api/               # API routes
│   ├── core/              # Core functionality
│   ├── models/            # Database models
│   └── services/          # Business logic
├── llm-ui/                # React frontend
├── docker/                # Docker configurations
├── migrations/            # Database migrations
├── scripts/               # Utility scripts
├── simple_gmail_mcp/       # Gmail MCP integration
├── docker-compose.yml      # Service orchestration
├── main.py                # FastAPI entry point
└── requirements.txt       # Python dependencies
```

## 🔑 API Endpoints

### LLM Endpoints

- `POST /api/llm/chat` - Send chat messages to the AI assistant
- `POST /api/llm/workflow` - Execute AI workflows
- `POST /api/llm/search` - Semantic search in documents
- `GET /api/llm/conversations` - List conversations
- `GET /api/llm/conversations/{id}` - Get conversation details

### Knowledge Graph

- `POST /api/kg/query` - Query knowledge graph
- `POST /api/kg/create` - Create knowledge graph entities
- `GET /api/kg/visualize` - Get graph visualization data

### Document Processing

- `POST /api/documents/upload` - Upload documents
- `GET /api/documents/{id}` - Get document details
- `DELETE /api/documents/{id}` - Delete document

## 🔌 MCP Integration

Jarvis supports Model Context Protocol (MCP) for extending functionality:

### Available MCP Servers

- **Gmail MCP**: Email processing and management
- **Custom MCP Servers**: Located in `../MCP` directory

### Adding MCP Servers

1. Place MCP server in the designated MCP servers path
2. Configure in `.env`:
   ```bash
   MCP_SERVERS_PATH=/mcp-servers
   ```

## 📊 Observability

Jarvis uses Langfuse for comprehensive observability:

- **Traces**: Track LLM calls and workflow executions
- **Metrics**: Monitor performance and resource usage
- **Analytics**: ClickHouse-powered analytics for insights
- **Debugging**: Detailed logs and traces for troubleshooting

Access Langfuse at http://localhost:3000

## 🛠️ Configuration

### Environment Variables

See `.env.example` for all available configuration options.

### Knowledge Graph Settings

Configuration options in the database allow fine-tuning of knowledge graph behavior:
- Node relationship thresholds
- Extraction confidence levels
- Graph visualization settings

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_rag_logic.py

# Run with verbose output
pytest -v

# Generate coverage report
pytest --cov=app --cov-report=html
```

## 📈 Performance Optimization

- Redis caching for conversation context
- Vector similarity search with Qdrant
- Lazy loading for large documents
- Async/await for I/O operations
- Connection pooling for databases

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 for Python code
- Use TypeScript for frontend
- Add tests for new features
- Update documentation as needed
- Use meaningful commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋 Support

For issues and questions:
- Open an issue on GitHub
- Check the [Langfuse documentation](https://langfuse.com/docs)
- Review API documentation at `/docs` endpoint

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [LangChain](https://langchain.com/) - LLM application framework
- [Qdrant](https://qdrant.tech/) - Vector similarity search
- [Neo4j](https://neo4j.com/) - Graph database
- [Material-UI](https://mui.com/) - React component library
- [Langfuse](https://langfuse.com/) - LLM observability

## 📞 Contact

- **Author**: Kian Woon
- **GitHub**: [@kianwoon](https://github.com/kianwoon)

---

Made with ❤️ by Kian Woon
