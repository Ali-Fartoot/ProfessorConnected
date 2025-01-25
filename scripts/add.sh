#!/bin/bash
# Add professor
curl -X POST "http://localhost:8000/add_professor" \
     -H "Content-Type: application/json" \
     -d '{"professor_name": "A. Barresi", "number_of_articles": 3}'


