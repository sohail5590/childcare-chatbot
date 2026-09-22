# Childcare Chatbot Install Checklist

Use this checklist on a brand-new Windows machine with no VS Code, no Python, and no WSL setup.

This project is a Docker-based system. The app will not run unless the required files, folders, and environment variables are present before startup.

---

## 1. Confirm the machine requirements

Before installing anything, make sure the computer has:

- Windows 10 or Windows 11
- Administrator rights
- Internet access
- A GitHub account if the repo is private
- A browser for opening the app after startup
- No VS Code required

---

## 2. Install WSL 2

Open PowerShell as Administrator and run:

```powershell
wsl --install
```

Then restart the machine if prompted.

After restart, open Ubuntu/WSL and set up your Linux username and password.

Verify:

```bash
wsl --version
```

---

## 3. Install Git

Download Git here:

https://git-scm.com/downloads

Verify:

```bash
git --version
```

---

## 4. Install Docker Desktop

Download Docker Desktop here:

https://www.docker.com/products/docker-desktop/

Install it and start Docker Desktop.

Then enable WSL integration:

- Open Docker Desktop
- Go to Settings
- Open Resources
- Select WSL Integration
- Enable your Ubuntu distro
- Click Apply & Restart

Verify:

```bash
docker --version
docker compose version
```

> If Docker is not found in WSL, this is the step that is missing.

---

## 5. Install Python (recommended, not always strictly required)

Although this project runs through Docker containers, Python is useful for local debugging, validation, and compatibility checks.

Install Python from:

https://www.python.org/downloads/

During install, check:

- Add Python to PATH

Verify:

```bash
python --version
```

---

## 6. Clone the project from GitHub

Open the WSL terminal and run:

```bash
git clone <github-repo-url>
cd childcare-chatbot
```

Make sure you are in the project root before starting containers.

---

## 7. Confirm the project structure exists

From the project root, make sure these items exist:

```bash
ls
```

Required project items:

- docker-compose.yml
- backend/
- frontend/
- chat-feature/
- Data/
- config.json
- .env
- hf_cache/

If any of these are missing, the app will not start correctly.

### Important notes

- `docker-compose.yml` is required because it defines all app services.
- `config.json` is mounted into containers and is used for state configuration.
- `Data/` is used for uploaded and processed documents.
- `hf_cache/` is mounted into the chat backend and is used for model caching.
- `.env` must exist at the project root.

---

## 8. Create the required `.env` file

Create a file named `.env` in the project root.

At minimum, add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

This project uses OpenAI for document classification and answer generation. Without a valid API key, the app will fail during runtime.

You may also want to include any project-specific env values used by the repo, but the required key for startup is:

```env
OPENAI_API_KEY=...
```

It is also okay to copy any repo-provided sample env file if one exists.

---

## 9. Make sure Docker can access project files

The app relies on local folder mounts. Confirm the following folders exist before startup:

```bash
mkdir -p Data hf_cache
```

This ensures the mounted local folders exist before Docker starts.

---

## 10. Start the app from the project root

From the project root (the folder containing `docker-compose.yml`), run:

```bash
docker compose up --build -d
```

This builds the app images and starts all services.

---

## 11. Check whether the containers started successfully

Run:

```bash
docker compose ps
```

If something failed, check logs:

```bash
docker compose logs -f
```

---

## 12. Verify the expected app ports

Open the browser on Windows and use these URLs if the services are running:

- Frontend: http://localhost:8501
- Chat frontend: http://localhost:8601
- Backend: http://localhost:9000
- DB API: http://localhost:9200
- ChromaDB: http://localhost:8000
- PostgreSQL: localhost:5433

---

## 13. If you need a full reset

To remove all containers and volumes and restart from scratch:

```bash
docker compose down -v --remove-orphans
docker compose up --build -d
```

This resets database and vector storage.

---

## 14. Common issues to check before panicking

### Docker command not found in WSL

- Open Docker Desktop
- Go to Settings → Resources → WSL Integration
- Enable your Ubuntu distro
- Restart Docker Desktop

### Project files missing

Check that these exist in the repo root:

- docker-compose.yml
- config.json
- Data/
- hf_cache/
- .env

### App fails because OpenAI key is missing

Check `.env` and make sure it contains:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Containers fail to start

Check logs:

```bash
docker compose logs -f
```

---

## 15. Complete required setup summary

Before running this app, the machine must have:

- WSL 2 installed
- Ubuntu distro available in WSL
- Git installed
- Docker Desktop installed and running
- WSL integration enabled in Docker Desktop
- Python installed (recommended)
- A GitHub repo cloned locally
- `docker-compose.yml` present in the project root
- `config.json` present in the project root
- `Data/` folder present
- `hf_cache/` folder present
- `.env` file created with `OPENAI_API_KEY`
- Docker Compose started from the project root

---

## 16. Final startup command

```bash
cd childcare-chatbot
docker compose up --build -d
```

That is the main command needed to run the application.

---

## 17. No VS Code required

You can do the entire setup in the terminal without VS Code. A plain WSL terminal, Docker Desktop, Git, and the GitHub repo are enough to get this project running.
