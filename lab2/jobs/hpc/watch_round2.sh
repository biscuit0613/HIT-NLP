#!/bin/bash
# 等待 round2 任务（combo/base_s23/relpos_s23/beam10）结束后汇总 BLEU 结果。
while squeue -j 177611,177612,177613,177614 -h 2>/dev/null | /usr/bin/grep -q .; do
  sleep 60
done
OUT=/home/jnguan/lab2/round2_results.txt
{
  echo "=== ROUND2 RESULTS $(date) ==="
  for t in combo base_s23 relpos_s23 beam10; do
    echo "[$t]"
    for y in 2017 2018; do
      if [ "$t" = "beam10" ]; then
        L=$HOME/lab2/runs/$t/run_logs/bleu_${y}.log
      else
        L=$HOME/lab2/runs/$t/run_logs/bleu_${t}_${y}.log
      fi
      echo "  $y: $(/usr/bin/grep 'BLEU =' $L 2>/dev/null | cut -c1-100)"
    done
  done
  echo "--- exit states ---"
  sacct -j 177611,177612,177613,177614 --format=JobID,JobName,State,Elapsed 2>/dev/null | /usr/bin/grep -v "batch\|extern"
  echo DONE
} > $OUT 2>&1
