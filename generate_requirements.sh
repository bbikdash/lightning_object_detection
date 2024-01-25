#!/bin/bash
pipreqs --use-local --ignore backup,assets,logs,debug_logs --savepath ./mock_requirements.txt --force --mode gt .

