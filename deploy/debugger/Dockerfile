# Debugger-Claw: Wächter (root) und Helfer `claude` (Nutzer claw, uid 1000) in einem Container.
# Build-Kontext ist die Wurzel des la_fungus-Checkouts (siehe compose.yml und README.md). Was in den
# Kontext kommt, regelt `Dockerfile.dockerignore` neben dieser Datei (BuildKit), nur für diesen Build.
FROM python:3.11-slim-bookworm

ARG GH_VERSION=2.83.*
ARG CLAUDE_CODE_VERSION=2.1.268

# git, setpriv (util-linux), pkill (procps), Node für Claude Code; curl nur für die gh-Quelle.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git util-linux procps ca-certificates curl nodejs npm \
 && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
 && chmod 0644 /usr/share/keyrings/githubcli-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends "gh=${GH_VERSION}" \
 && apt-get purge -y curl \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# bookworm bringt Node 18 (18.19). --engine-strict: verlangt claude-code eine neuere Node-Version,
# bricht der Build hier ab statt mit Warnung weiterzulaufen (dann NodeSource, siehe README).
RUN node --version \
 && npm install -g --engine-strict "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
 && npm cache clean --force

# Helfer-Nutzer claw. HOME gehört root (die .gitconfig darin schreibt der Einstieg);
# beschreibbar für claw ist nur das Volume /home/claw/.claude.
RUN groupadd -g 1000 claw \
 && useradd -u 1000 -g 1000 -d /home/claw -M -s /usr/sbin/nologin claw \
 && mkdir -p /home/claw/.claude /data/klon /data/stand/wt /data/spiegel /data/konfig \
 && chown root:root /home/claw && chmod 0755 /home/claw \
 && chown claw:claw /home/claw/.claude /data/klon /data/stand/wt

# la_fungus (Paket selbstbild) aus dem Build-Kontext.
COPY pyproject.toml README.md /opt/la_fungus/
COPY src /opt/la_fungus/src
RUN pip install --no-cache-dir "/opt/la_fungus[selbstbild]"

COPY deploy/debugger/askpass.sh /usr/local/bin/debugger-askpass
COPY deploy/debugger/einstieg.sh /usr/local/bin/debugger-einstieg
RUN chmod 0755 /usr/local/bin/debugger-askpass /usr/local/bin/debugger-einstieg

ENV DEBUGGER_ASKPASS=/usr/local/bin/debugger-askpass \
    PYTHONUNBUFFERED=1

# Der Wächter läuft als root (rootless Docker: root im Container ist kein root auf dem Host).
ENTRYPOINT ["/usr/local/bin/debugger-einstieg"]
CMD ["python", "-m", "selbstbild", "debugger", "schleife", "/data/konfig", "--klon", "/data/klon", "--ordner", "/data/stand", "--spiegel", "/data/spiegel"]
