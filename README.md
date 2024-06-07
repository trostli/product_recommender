---
title: "Product Recommender"
emoji: "🚀"
sdk: "docker"
app_file: app.py
pinned: true
---

# Prerequisites

- Python 3.11 or higher
- Pip 3.11 or higer
- Chainlit 1.1.101

# Setup

To run:

1. Create a `.env` file in the root directory with the following format:
```
OPENAI_API_KEY=<your_api_key>
```
2. `pip install chainlit==1.1.101`
3. `chainlit run app.py -w`

# Deployment

Push to the huggingface repository, to auto-deploy.