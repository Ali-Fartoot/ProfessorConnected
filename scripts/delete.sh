#!/bin/bash

# Cleanup database
curl -X DELETE "http://localhost:8000/cleanup_database" 