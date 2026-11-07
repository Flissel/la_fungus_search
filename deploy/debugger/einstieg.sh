#!/bin/sh
# Einstieg des Debugger-Containers. Läuft als root, setzt Rechte und die Git-Konfiguration des
# Helfer-Nutzers claw und führt dann den Befehl aus (Wächter oder ein einmaliger Einrichtungsbefehl).
set -eu
umask 022

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
