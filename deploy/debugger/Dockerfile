# Debugger-Claw: Wächter (root) und Helfer `claude` (Nutzer claw, uid 1000) in einem Container.
# Build-Kontext ist die Wurzel des la_fungus-Checkouts (siehe compose.yml und README.md). Was in den
# Kontext kommt, regelt `Dockerfile.dockerignore` neben dieser Datei (BuildKit), nur für diesen Build.
FROM python:3.11-slim-bookworm

ARG CLAUDE_CODE_VERSION=2.1.268
ARG NODE_MAJOR=22
# gh als Release-.deb mit fester Prüfsumme: Das apt-Repo von cli.github.com führt nur die neueste
# Version, ein Versions-Pin dort bricht also, sobald eine neue erscheint. Nur amd64 (VM .65).
ARG GH_VERSION=2.103.3
ARG GH_SHA256=e3bafef96328d1a67de3a968bdfe1cc2ffd58cf6aba8b5e201356897ed06bb3d

# git, setpriv (util-linux), pkill (procps), gh (der Wächter ruft `gh api` mit GH_TOKEN je Aufruf,
# funde/github.py; der Helfer darf gh nicht nutzen, nicht in ERLAUBT), Node für Claude Code.
# Node aus NodeSource (Hauptversion gepinnt): bookworms Node 18 reicht claude-code 2.1.268 nicht.
# curl und gnupg nur für die Quellen, danach wieder entfernt.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git util-linux procps ca-certificates curl gnupg \
 && test "$(dpkg --print-architecture)" = amd64 \
 && curl -fsSL -o /tmp/gh.deb "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_amd64.deb" \
 && echo "${GH_SHA256}  /tmp/gh.deb" | sha256sum -c - \
 && apt-get install -y --no-install-recommends /tmp/gh.deb \
 && rm /tmp/gh.deb \
 && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
      | gpg --dearmor -o /usr/share/keyrings/nodesource.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" \
      > /etc/apt/sources.list.d/nodesource.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends nodejs \
 && apt-get purge -y curl gnupg \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# --engine-strict: verlangt claude-code eine neuere Node-Version, bricht der Build ab.
RUN gh --version \
 && node --version \
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

# Der Wächter läuft als root. Unter rootless Docker ist das kein root auf dem Host; unter rootful
# Docker (VM .65) muss die Token-Datei auf dem Host root:root 0400 sein, sonst bricht der Einstieg ab.
ENTRYPOINT ["/usr/local/bin/debugger-einstieg"]
CMD ["python", "-m", "selbstbild", "debugger", "schleife", "/data/konfig", "--klon", "/data/klon", "--ordner", "/data/stand", "--spiegel", "/data/spiegel"]
