# Genesys - Plataforma SaaS de Deep Q-Learning

Genesys é uma plataforma SaaS API-first para treinamento e inferência de agentes de Deep Q-Learning (DQN). A plataforma oferece uma arquitetura extensível com templates de ambiente, execução assíncrona de jobs e persistência completa de modelos.

## 🎯 Características Principais

- **API-first**: Toda a funcionalidade acessível via API REST
- **Treinamento Assíncrono**: Jobs de treino executam em workers separados via Celery
- **Sistema de Templates**: Ambientes plugáveis (GridWorld, Decision Optimization)
- **Persistência Completa**: SQLite para metadados, filesystem para artefatos
- **Versionamento de Modelos**: Controle de versões com ativação de modelos
- **Inferência via API**: Predições em tempo real com modelos treinados

## 🏗️ Arquitetura

```
genesys/
├── app/
│   ├── api/              # FastAPI routes
│   ├── core/             # Core logic
│   ├── db/               # Database session
│   ├── models/           # SQLAlchemy models
│   ├── rl/               # DQN implementation
│   ├── schemas/          # Pydantic schemas
│   ├── services/         # Business logic
│   ├── templates/        # Environment templates
│   ├── utils/            # Utilities
│   └── workers/          # Celery tasks
├── storage/              # Data persistence
├── tests/                # Test suite
├── main.py               # Application entry
└── requirements.txt      # Dependencies
```

## 🚀 Início Rápido

### 1. Instalação

```bash
# Clone o repositório
git clone <repository-url>
cd genesys

# Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instale dependências
pip install -r requirements.txt
```

### 2. Configuração

```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite as configurações conforme necessário
nano .env
```

### 3. Inicie os Serviços

#### Terminal 1 - Redis (Message Broker)
```bash
redis-server
```

#### Terminal 2 - API
```bash
python main.py
```

#### Terminal 3 - Celery Workers
```bash
celery -A app.workers.celery_app worker --loglevel=info
```

## 📡 API Endpoints

### Autenticação

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/v1/auth/register` | Registrar novo usuário |
| POST | `/v1/auth/login` | Login e obter token |
| GET | `/v1/auth/me` | Informações do usuário |

### Projetos

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/v1/projects` | Criar projeto |
| GET | `/v1/projects` | Listar projetos |
| GET | `/v1/projects/{id}` | Obter projeto |
| PATCH | `/v1/projects/{id}` | Atualizar projeto |
| DELETE | `/v1/projects/{id}` | Deletar projeto |

### Treinamento

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/v1/jobs/projects/{id}/train` | Criar job de treino |
| GET | `/v1/jobs` | Listar jobs |
| GET | `/v1/jobs/{id}` | Status do job |
| GET | `/v1/jobs/{id}/logs` | Logs do treino |
| POST | `/v1/jobs/{id}/cancel` | Cancelar job |

### Modelos

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/v1/models/projects/{id}/models` | Listar modelos |
| POST | `/v1/models/{id}/activate` | Ativar modelo |
| GET | `/v1/models/{id}/download` | Download do modelo |

### Inferência

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/v1/inference/projects/{id}/predict` | Predição |

### Templates

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/v1/templates` | Listar templates |
| GET | `/v1/templates/{name}` | Detalhes do template |

## � Web Frontend

A aplicação agora inclui uma interface web para usar a API diretamente no navegador.

- Acesse: `http://localhost:8000/ui`
- O frontend consome os endpoints existentes em `/v1`
- Não altera a versão da API ou os endpoints atuais

## �💡 Exemplos de Uso

### 1. Registrar Usuário

```bash
curl -X POST http://localhost:8000/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "email": "john@example.com",
    "password": "securepassword123"
  }'
```

### 2. Criar Projeto

```bash
curl -X POST http://localhost:8000/v1/projects \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "GridWorld Navigation",
    "description": "Agente de navegação em grid",
    "template_default": "grid_world"
  }'
```

### 3. Iniciar Treinamento

```bash
curl -X POST http://localhost:8000/v1/jobs/projects/PROJECT_ID/train \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "template": "grid_world",
    "config": {
      "state_size": 8,
      "action_space": [0, 1, 2, 3],
      "episodes": 500,
      "max_steps": 100,
      "gamma": 0.99,
      "learning_rate": 0.001,
      "epsilon_start": 1.0,
      "epsilon_end": 0.01,
      "epsilon_decay": 0.995,
      "batch_size": 64,
      "memory_size": 10000,
      "target_update_freq": 100
    }
  }'
```

### 4. Verificar Status do Treino

```bash
curl http://localhost:8000/v1/jobs/JOB_ID \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 5. Ativar Modelo

```bash
curl -X POST http://localhost:8000/v1/models/MODEL_ID/activate \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 6. Fazer Predição

```bash
curl -X POST http://localhost:8000/v1/inference/projects/PROJECT_ID/predict \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "state": [0.0, 0.0, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0]
  }'
```

## 🔧 Templates Disponíveis

### GridWorld

Ambiente de navegação em grade 2D com obstáculos.

**Configuração:**
```json
{
  "grid_size": [5, 5],
  "obstacle_count": 3,
  "max_steps": 100,
  "use_sensors": true,
  "sensor_range": 2
}
```

**Ações:** 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT

### DecisionOptimization

Ambiente genérico para otimização de decisões.

**Configuração:**
```json
{
  "state_size": 10,
  "action_space": [0, 1, 2],
  "max_steps": 100,
  "reward_type": "linear",
  "state_change_prob": 0.1,
  "noise_std": 0.0
}
```

**Tipos de Reward:** `linear`, `quadratic`, `sparse`, `custom`

## 📊 Monitoramento

### Logs do Worker

```bash
# Ver logs em tempo real
celery -A app.workers.celery_app worker --loglevel=info

# Ver fila de tarefas
celery -A app.workers.celery_app inspect active
```

### Métricas de Treino

Acesse as métricas detalhadas via API:

```bash
curl http://localhost:8000/v1/jobs/JOB_ID/metrics \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## 🧪 Testes

```bash
# Executar todos os testes
pytest

# Com cobertura
pytest --cov=app --cov-report=html
```

## 🚢 Deploy

### Docker (opcional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

### Produção

1. Use PostgreSQL em vez de SQLite
2. Configure Redis para persistência
3. Use múltiplos workers Celery
4. Configure nginx como reverse proxy
5. Use HTTPS

## 📚 Documentação Adicional

- [Documentação da API](http://localhost:8000/docs) - Swagger UI
- [ReDoc](http://localhost:8000/redoc) - Documentação alternativa

## 🤝 Contribuindo

1. Fork o repositório
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Crie um Pull Request

## 📝 Licença

MIT License - veja [LICENSE](LICENSE) para detalhes.

## 📧 Contato

Para suporte ou dúvidas, abra uma issue no GitHub.

---

**Genesys** - Plataforma de Deep Q-Learning API-First 🧠⚡
