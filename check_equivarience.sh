  ./.venv/bin/python -m deepVBH.check_equivariance \
    --data-root deepvbscf_data_f2_scan \
    --checkpoint /home/xiatao/vb_project/xmvb-cpp/artifacts/deepVBH_baseline.pt \
    --num-trials 4 \
    --hidden-dim 128 \
    --pair-hidden-dim 128 \
    --lmax 2 \
    --num-radial 8