#!/bin/bash
kill $(ps aux | grep "$1" | grep -v grep | awk '{print $2}')
