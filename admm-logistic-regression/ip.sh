#!/bin/bash
watch "cat logs/$(ls -1t logs | grep 'async-master' | head -1) | grep '172.24'"
