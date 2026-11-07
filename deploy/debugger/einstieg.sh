#!/bin/sh
# Einstieg des Debugger-Containers. Läuft als root, setzt Rechte und die Git-Konfiguration des
# Helfer-Nutzers claw und führt dann den Befehl aus (Wächter oder ein einmaliger Einrichtungsbefehl).
set -eu
umask 022

# R35: Die Token-Datei muss da sein, und claw darf sie nicht lesen können (fail-closed). Unter
# rootful Docker behält ein Secret aus `secrets: file:` den Host-Besitzer, oft uid 1000 = claw.
TOKEN_DATEI="${DEBUGGER_TOKEN_DATEI:-/run/secrets/lab_token}"
if [ ! -e "$TOKEN_DATEI" ]; then
  echo "debugger-einstieg: Token-Datei fehlt ($TOKEN_DATEI). Abbruch." >&2
  exit 1
fi
if setpriv --reuid=1000 --regid=1000 --init-groups -- test -r "$TOKEN_DATEI"; then
  echo "debugger-einstieg: claw (uid 1000) kann die Token-Datei lesen. Abbruch." \
       "Nötig: rootless Docker oder eine root-eigene Datei mit 0400 (siehe README)." >&2
  exit 1
fi
besitzer="$(stat -c '%u' "$TOKEN_DATEI")"
modus="$(stat -c '%a' "$TOKEN_DATEI")"
case "$modus" in
  *00) gruppe_andere=nein ;;
  *) gruppe_andere=ja ;;
esac
if [ "$besitzer" != "0" ] || [ "$gruppe_andere" = "ja" ]; then
  echo "debugger-einstieg: Warnung: Token-Datei hat Besitzer $besitzer und Rechte $modus" \
       "(erwartet 0 und keine Rechte für group/other)." >&2
fi

# Stand und Spiegel gehören root. claw darf den Spiegel nur lesen (fetch aus dem Klon) und im
# Stand-Ordner nur wt/ beschreiben.
mkdir -p /data/stand/wt
chown root:root /data/stand /data/spiegel
chmod 0755 /data/stand /data/spiegel
chmod -R a+rX /data/spiegel
chown claw:claw /data/klon /data/stand/wt /home/claw/.claude

# Git-Konfiguration für claw: root-eigen, claw kann sie nicht ändern (HOME gehört root).
# safe.directory: der Spiegel gehört root. protocol.file: Submodule holen aus dem Spiegelpfad.
cat > /home/claw/.gitconfig <<'EOF'
[safe]
	directory = *
[user]
	name = Vibemind-LAB
[protocol "file"]
	allow = always
EOF
chown root:root /home/claw/.gitconfig
chmod 0644 /home/claw/.gitconfig

exec "$@"
