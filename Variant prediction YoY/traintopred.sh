#!/bin/bash

# Set current timepoint manually at the top
current_timepoint="2025_Mar"

# Extract year and month abbreviation
year=$(echo "$current_timepoint" | cut -d'_' -f1)
month=$(echo "$current_timepoint" | cut -d'_' -f2)

# Define previous quarter logic
case "$month" in
  "Mar")
    prev_year=$((year - 1))
    prev_month="Dec"
    ;;
  "Jun")
    prev_year=$year
    prev_month="Mar"
    ;;
  "Sep")
    prev_year=$year
    prev_month="Jun"
    ;;
  "Dec")
    prev_year=$year
    prev_month="Sep"
    ;;
  *)
    echo "Invalid month: $month"
    exit 1
    ;;
esac

previous_timepoint="${prev_year}_${prev_month}"

echo "Running pipeline for current: $current_timepoint, previous: $previous_timepoint"

# First script
python3 ./pretrain.py \
  -n "$current_timepoint" \
  -p "./${previous_timepoint}/${previous_timepoint}_traindat.csv" \
  -d "./${current_timepoint}/attr.roll.${current_timepoint}.csv" \
  -t "./${current_timepoint}/${current_timepoint}_m36dedupali.contree" \
  -f "./${current_timepoint}/${current_timepoint}_m36dedupali.fasta" \
  -o "./${current_timepoint}"

# Check success
if [ $? -ne 0 ]; then
  echo "pretrain.py failed. Skipping yoypred.py."
  exit 1
fi

# Second script
python3 ./yoypred.py \
  -n "$current_timepoint" \
  -t "./${current_timepoint}/${current_timepoint}_traindat.csv" \
  -v "./${current_timepoint}/variant.summary.active.csv" \
  -pp "./${previous_timepoint}/${previous_timepoint}_allpred.csv" \
  -pf "./${previous_timepoint}/${previous_timepoint}_predperf.csv" \
  -o "./${current_timepoint}"
