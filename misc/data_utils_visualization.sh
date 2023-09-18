#!/bin/bash



# Specify the directory
directory="data_dnb_plots"

# # Loop through each entry in the directory
# for entry in "$directory"/*; do
#     # Check if it's a directory
#     if [ -d "$entry" ]; then
#         # Print the full path
#         echo "Start $entry"
#         ./venv/bin/python scripts/dataset_gen.py --src_dir=$entry --dst_dir="data_dnb_plots_target/$entry" --split=1_1_1 --is_visualize
#     fi
# done

./venv/bin/python scripts/dataset_gen.py --src_dir=data_dnb_plots/3d_rpf --dst_dir="data_dnb_plots_target/$entry" --split=1_1_1 --is_visualize
