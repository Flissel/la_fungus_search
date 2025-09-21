#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

# Re-export the original CLI by importing from its old location to preserve behavior
from ..agent_fungus_rag import main  # type: ignore

if __name__ == "__main__":
    main()


