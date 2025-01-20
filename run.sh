#!/bin/bash

# Add professor
# curl -X POST "http://localhost:8000/add_professor" \
#      -H "Content-Type: application/json" \
#      -d '{"name": "Andrew Ng", "number_of_articles": 3}'


# Search without visualization
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"professor_name": "Sergey Levine", "min_similarity": 0.1, "limit": 5}'

# Cleanup database
# curl -X DELETE "http://localhost:8000/cleanup_database" 