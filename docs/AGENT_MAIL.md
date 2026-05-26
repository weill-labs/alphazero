# Agent Mail MCP diagnostics

This note records the 2026-05-26 investigation into NTM workers failing to
start the `mcp-agent-mail` MCP server. The visible symptom was an HTTP
handshake failure against `127.0.0.1`.

## Findings

- Agent Mail is installed at `/home/cweill/mcp_agent_mail` and reports version
  `0.3.6` for both `am` and `mcp-agent-mail`.
- `mcp-agent-mail config` shows the intended stable HTTP endpoint as
  `127.0.0.1:8765/mcp/`, backed by
  `/home/cweill/.mcp_agent_mail_git_mailbox_repo/storage.sqlite3`.
- Nothing was listening on `127.0.0.1:8765` during the failure. A direct probe
  returned `curl: (7) Failed to connect to 127.0.0.1 port 8765`.
- A different Agent Mail HTTP runtime was listening on `127.0.0.1:38191`:
  `am serve-http --host 127.0.0.1 --no-tui --no-auth --port 38191`. Its parent
  process was `ntm internal-monitor alphago-cleanup`.
- The user systemd service `agent-mail.service` is enabled, but it is stuck in
  an auto-restart loop. `am service status` showed
  `ExecStartPre=/home/cweill/mcp_agent_mail/am migrate` failing because the
  mailbox activity lock is already held by another Agent Mail runtime.
- The mailbox data itself appears healthy. `am robot health --json` reported
  green health for database connectivity, schema, archive parity, search index,
  circuit breakers, and disk. `am doctor triage` returned zero findings.
- The project-local MCP files `cursor.mcp.json` and `gemini.mcp.json` pointed
  at `127.0.0.1:8765/mcp/`. At the time of the failure, global agent configs
  pointed at the live dynamic endpoint `127.0.0.1:38191/mcp/`.
- `.ntm/logs/am-alphago-infra.log` shows NTM's Agent Mail sidecar restarting
  repeatedly. It briefly served different ports, including `8765`, then exited
  after its supervisor hit the max restart count.
- Service logs also showed later startup failures with
  `cannot open DB ... Query error: disk I/O error` while the filesystem was
  under disk pressure. After disk pressure cleared, the service started and its
  readiness probe passed.

## Current status

A later recheck on 2026-05-26 showed the stable user service recovered:

- `agent-mail.service` was `active (running)`.
- `ss` showed `am` listening on `127.0.0.1:8765`.
- `curl http://127.0.0.1:8765/mcp/` reached the server and returned
  `405 Method Not Allowed`, which is enough to prove the HTTP listener is up
  for this simple probe.
- Service logs showed the startup readiness self-probe passed on
  `127.0.0.1:8765`.

No risky system remediation was run as part of this investigation.

## Root cause

This is not a mailbox corruption problem. It is a listener availability problem
on the configured MCP endpoint.

During the incident, the stable service that should serve
`127.0.0.1:8765/mcp/` was not listening, so workers configured for `8765` failed
their HTTP handshake with connection refused.

Two conditions contributed:

1. Only one Agent Mail runtime can hold the mailbox activity lock. NTM
   per-session sidecars and the user systemd service were competing for the same
   mailbox. The runtime that won the lock was an NTM-owned, no-auth sidecar on a
   dynamic port (`38191`).
2. The host was also under disk pressure, and service logs showed SQLite disk
   I/O errors before the service recovered.

## Recommended fix

Keep enough disk headroom for SQLite, and pick one owner for the Agent Mail HTTP
endpoint. The least surprising setup for multi-agent work is a single user
service on a stable port, with all agents configured to that endpoint.

User-required commands, not run during this investigation:

```bash
/home/cweill/mcp_agent_mail/am service status
systemctl --user restart agent-mail.service
/home/cweill/mcp_agent_mail/am service status
```

If the restart still reports the mailbox activity lock as busy, first coordinate
with active NTM session owners and stop the NTM session or Agent Mail sidecar
that is currently serving the dynamic port. Then retry the service restart.

After the stable service is listening on `127.0.0.1:8765`, re-run setup for the
agents that NTM will spawn. Start with a dry run:

```bash
/home/cweill/mcp_agent_mail/am setup run --dry-run --agent codex --host 127.0.0.1 --port 8765 --path /mcp/ --project-dir /data/projects/alphago-infra
/home/cweill/mcp_agent_mail/am setup run --yes --agent codex --host 127.0.0.1 --port 8765 --path /mcp/ --project-dir /data/projects/alphago-infra
```

Repeat the setup command for `claude`, `cursor`, or `gemini` if those agents are
spawned for the same project.

## Workaround

If a stable service is not desired, configure spawned workers to use the
currently live NTM sidecar endpoint instead of `8765`. This is brittle because
the sidecar port can change after NTM restarts. It should be treated as a
temporary unblocker, not the long-term setup.

## NTM-side follow-up

NTM should avoid launching multiple Agent Mail owners for the same mailbox. It
should either use the stable user service, or publish the actual sidecar port to
spawned agents only after the sidecar has survived startup. Writing worker MCP
configs that point at `8765` while a different runtime owns the mailbox on a
dynamic port creates the connection-refused failure observed here.
