#!/bin/bash
gunicorn api:app -b 0.0.0.0:5000
