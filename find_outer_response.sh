#!/bin/bash
FILE="/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/cpp_vb_scf_optimizer.cpp"
echo "Searching for use_outer_response modifications:"
grep -n "use_outer_response" "$FILE" | head -20
echo ""
echo "Looking for assignment to use_outer_response:"
grep -n -B3 -A3 "use_outer_response.*=" "$FILE" | head -20
