# Bangladesh Multi-Tool AI Agent

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KalXktn2_DuI_l25qmHL59BCJYWnxZZ7?usp=sharing)

A small LangChain agent that answers questions about Bangladesh by picking
the right tool for the job: three local SQLite databases for structured
facts (institutions, hospitals, restaurants), and a web search tool for
everything else.

## Overview

Ask it "how many hospitals are in Dhaka?" and it hits a SQLite database.
Ask it "what is the healthcare policy of Bangladesh?" and it goes to the
web instead. The agent decides which tool to use per question — you don't
have to tell it.
