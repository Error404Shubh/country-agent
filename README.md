#  Country Information Agent

A LangGraph agent that answers natural-language questions about countries. You ask it something like "What currency does Japan use?" and it figures out what you're asking, hits the REST Countries API, and gives you a clean answer. No database, no auth, no embeddings — just a three-step pipeline and a public API.

Built with FastAPI on the backend and a dark dev-tool UI on the frontend. Runs in Docker.

---

## How it works

The agent runs three steps in sequence every time you ask a question.

First it reads your question and works out which country you're asking about and what data you want — population, capital, currency, whatever. This is the `parse_intent` node, powered by Kimi K2.6 via NVIDIA NIM.

Then it calls the REST Countries API to actually fetch that data. It only pulls the fields you asked about, so the response stays lean and the final answer stays grounded.

Finally the `synthesize` node takes the raw data and writes a proper answer in plain English. It's explicitly told to use only what the API returned — no hallucinating figures it doesn't have.

If anything goes wrong at any step (bad country name, API down, model error), the graph short-circuits and returns a clean error message instead of blowing up.



## Installation

### What you need before starting

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine + Compose on Linux)
- A NVIDIA NIM API key — get one free at [build.nvidia.com](https://build.nvidia.com/moonshotai/kimi-k2.6), click **Get API Key**
- Git

That's it. No Python, no Node, nothing else required if you're running via Docker.

---

### Step 1 — Clone the repo

```bash
git clone https://github.com/YOUR_HANDLE/country-agent.git
cd country-agent
```

---

### Step 2 — Set up your environment file

```bash
cp .env.example .env
```

Open `.env` in any editor and fill in your API key:

```bash
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxx
```

Two things that will break it if you get them wrong:
- No quotes around the value — `nvapi-xxx` not `'nvapi-xxx'`
- No spaces around the equals sign — `KEY=value` not `KEY = value`

---

### Step 3 — Build and run

```bash
docker compose up --build
```

First run takes 30–60 seconds to pull the base image and install dependencies. Subsequent runs use the cache and start in a few seconds.

---

### Step 4 — Open it

| What | URL |
|------|-----|
| Web UI | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| Health check | http://localhost:8000/api/health |

---

### Verify it's working

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of Japan?"}'
```

You should get a JSON response back with `"answer"` containing something about Tokyo.

---

### Stopping it

```bash
# Stop but keep the container
docker compose stop

# Stop and remove the container
docker compose down
```

---

### Running without Docker (local Python)

If you'd rather run it directly:

```bash
# Requires Python 3.12+
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

export NVIDIA_API_KEY=nvapi-xxx  # Windows: set NVIDIA_API_KEY=nvapi-xxx

uvicorn app.main:app --reload --port 8000
```

---

### Troubleshooting

**401 Unauthorized in the logs**
Your API key is wrong or has quotes around it. Run `docker compose exec api printenv NVIDIA_API_KEY` — if the output has quote characters in it, remove them from your `.env` file.

**Port 8000 already in use**
Something else is running on that port. Either stop it, or change the port in `docker-compose.yml` from `"8000:8000"` to e.g. `"8080:8000"` and open `http://localhost:8080` instead.

**Build fails with dependency errors**
Make sure your `requirements.txt` uses `>=` version bounds, not pinned `==` versions. See the `requirements.txt` in the repo — it should already be correct.

**Container starts but UI doesn't load**
Check that the `static/` folder exists and contains `index.html`. The Dockerfile copies it in — if you moved or renamed it the container won't find it.