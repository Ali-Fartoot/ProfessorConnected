#!/bin/bash
curl -X 'POST' \
  'http://127.0.0.1:5333/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "max_tokens": 200,
  "messages": [
    {
      "content": "You are a helpful assistant.",
      "role": "system"
    },
    {
      "content": "Write a poem for France?",
      "role": "user"
    }
  ]
}'
