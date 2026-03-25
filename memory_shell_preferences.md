---
name: shell preferences
description: User approves shell commands without prompts for read/start/stop/kill operations
type: feedback
---

User approves running shell commands (Bash tool) without asking for confirmation for:
- Reading files or logs (cat, tail, grep, etc.)
- Starting/stopping services (uvicorn, etc.)
- Killing processes (pkill, kill, etc.)
- curl to check health/endpoints

Do NOT ask "Are you sure?" for these operations. Just execute them.
