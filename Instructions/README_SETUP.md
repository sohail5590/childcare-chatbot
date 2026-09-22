# Childcare Chatbot Setup Instructions

This guide explains everything a new developer or user needs in order to clone this app from GitHub and run it successfully on a different computer.

It assumes they do not already have Python, Docker, WSL, or VS Code installed.
It also assumes they are using a normal Windows machine and want to set everything up from scratch without an IDE.

---

## 1. What you need before starting

You will need:

- A Windows 10 or Windows 11 computer
- Internet access
- Administrator privileges on the machine for installing software
- No prior VS Code install
- No prior Python install
- No prior Docker install
- No prior WSL setup
- Git installed
- Docker Desktop installed
- WSL 2 enabled and configured
- A GitHub account if the repository is private

---

## 2. Install WSL 2

WSL is required because Docker Desktop on Windows works best with the Linux subsystem.

### Steps

1. Open PowerShell as Administrator.
2. Run:

```powershell
wsl --install
```

3. Restart the computer if prompted.
4. After restart, open Ubuntu from the Start menu.
5. Set up the Linux username and password when prompted.

### Verify WSL

Run:

```bash
wsl --version
```

If it works, WSL is installed successfully.

---

## 3. Install Git

### On Windows

1. Download Git from:
   https://git-scm.com/downloads
2. Run the installer.
3. Keep the default settings unless you know you need something different.
4. Finish installation.

### Verify Git

In PowerShell, Git Bash, or WSL:

```bash
git --version
```

If this prints a version number, Git is installed correctly.

---

## 4. Install Docker Desktop

1. Download Docker Desktop for Windows:
   https://www.docker.com/products/docker-desktop/
2. Install it.
3. Launch Docker Desktop.
4. When prompted, allow it to install WSL integration support if needed.
5. Wait for Docker Desktop to finish starting.

### Important

Docker Desktop must be running before you try to run any containers.

### Verify Docker

Open a WSL terminal and run:

```bash
docker --version
```

and:

```bash
docker compose version
```

If both commands work, Docker is ready.

> You do not need VS Code for this step. A plain WSL terminal is enough.

---

## 5. Enable WSL integration in Docker Desktop

This is often the missing step that causes the error:

```bash
The command 'docker' could not be found in this WSL 2 distro.
```

### Steps

1. Open Docker Desktop.
2. Go to Settings.
3. Open Resources.
4. Select WSL Integration.
5. Enable the WSL distro being used (for example, Ubuntu).
6. Click Apply & Restart.

After Docker restarts, reopen your WSL terminal and verify:

```bash
docker --version
```

---

## 6. Install Python (if needed)

This app is a Python-based project, and even though Docker handles most of the app runtime, local Python is still useful for debugging and compatibility.

### Install Python on Windows

1. Download Python from:
   https://www.python.org/downloads/
2. During install, check:
   - Add Python to PATH
   - Install launcher for all users
3. Finish installation.

### Verify Python

In WSL or PowerShell:

```bash
python --version
```

or:

```bash
python3 --version
```

---

## 7. Clone the project from GitHub

Open a WSL terminal and run:

```bash
git clone <github-repo-url>
```

Example:

```bash
git clone https://github.com/your-user/childcare-chatbot.git
```

Then move into the project folder:

```bash
cd childcare-chatbot
```

If you do not have VS Code, you can still do this entirely in the terminal. The repository can be edited later with any text editor you prefer, such as Notepad++, VS Code later, or a plain text editor.

---

## 8. Add the required OpenAI key

Before starting the app, create a file named `.env` in the project root.

Add the following line:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

This key is required because the app uses OpenAI models for document classification and answer generation.

If the `.env` file is missing or the key is invalid, the app will not run properly.

You do not need VS Code to create this file. You can create it in any terminal editor, or with a basic text editor on Windows.

---

## 9. Check required files

Before running the project, confirm you have the app structure in the cloned repository, including:

You do not need VS Code to verify these files. You can use:

```bash
ls
```

or:

```bash
dir
```

in the terminal to see the project folders.

- docker-compose.yml
- frontend/
- backend/
- chat-feature/
- config.json
- Data/
- .env (if required by the project)

If the repository includes environment variables, create the `.env` file from a sample if one exists.

---

## 10. Start the app with Docker Compose

From the project root:

```bash
docker compose up --build -d
```

This will:

- build Docker images
- start all services
- create the required containers for the frontend, backend, database, and vector database

### If you want a fresh reset

Run:

```bash
docker compose down -v --remove-orphans
```

Then start again:

```bash
docker compose up --build -d
```

---

## 11. Check whether the app is running

After startup, run:

```bash
docker compose ps
```

This shows all running containers.

You can also check logs:

```bash
docker compose logs -f
```

---

## 12. Access the app in a browser

No VS Code is required. Once the containers are running, open your browser on Windows and go to the app URLs below.

Depending on the project setup, common ports are:

- Frontend app: http://localhost:8501
- Chat frontend: http://localhost:8601
- Backend API: http://localhost:9000
- DB API: http://localhost:9200
- PostgreSQL: localhost:5433
- ChromaDB: http://localhost:8000

Open the relevant URL in your browser.

---

## 12. Common setup problems

### Problem: `docker: command not found`

Cause: Docker Desktop is not connected to WSL.

Fix:
- Open Docker Desktop
- Go to Settings → Resources → WSL Integration
- Enable your distro
- Restart Docker Desktop

Then run:

```bash
docker --version
```

### Problem: Docker is running but containers fail to start

Check logs:

```bash
docker compose logs -f
```

Look for:
- missing environment variables
- missing `.env` file
- database connection errors
- port conflicts

### Problem: app cannot connect to Postgres or Chroma

Make sure all containers are running:

```bash
docker compose ps
```

If needed, restart:

```bash
docker compose restart
```

---

## 13. Minimum software checklist

The setup is not complete until the OpenAI API key is added in `.env`.

This should be checked before launch:

- `.env` exists in the project root
- `OPENAI_API_KEY` is present
- key is valid and active


A new computer should have:

- Windows 11 or 10
- WSL 2 installed
- Ubuntu distro installed in WSL
- Git installed
- Docker Desktop installed and running
- WSL integration enabled
- Python installed
- Internet access
- GitHub repo access
- No VS Code required

This means the person can begin with only Windows, internet access, and administrator rights.

---

## 14. One-line summary

To run this app on a different computer, the user needs:

1. WSL 2
2. Git
3. Docker Desktop with WSL integration enabled
4. Python installed
5. The repository cloned from GitHub
6. Docker Compose used to build and start the project

---

## 15. Recommended first command after cloning

```bash
cd childcare-chatbot
docker compose up --build -d
```

This is the main startup command for the project.

---

## Final note

If this project expects environment secrets or a `.env` file, that file must be copied into the project root before the containers are started. Without it, some services may fail at startup.

If the repository includes a README, read it before running Docker, because some apps require extra setup steps or generated files.
