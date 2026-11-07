#!/bin/sh
# Askpass des Debugger-Wächters (R7). Git ruft es nur in den Push- und Fetch-Prozessen des Spiegels auf;
# nur dort steht DEBUGGER_GIT_TOKEN in der Umgebung. Der Token steht nie in argv oder einer URL.
case "${1:-}" in
  Username*) printf '%s\n' "${DEBUGGER_GIT_USER:-x-access-token}" ;;
  *) printf '%s\n' "${DEBUGGER_GIT_TOKEN:-}" ;;
esac
