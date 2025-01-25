#!/bin/bash
# Search without visualization
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"professor_name": "Sergey Levine", "min_similarity": 0.1, "limit": 5}'
