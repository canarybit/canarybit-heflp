#!/bin/bash

# Array containing the list of files and folders to copy
# Modify this if necessary, using Regular Expression
files_to_copy=(
    "./heflp/heflp/secureproto/.+py"  # all .py files in current folder and all subfolders
    "./heflp/heflp/strategy/.+py"
    "./heflp/heflp/utils/.+py"
    "./heflp/heflp/[^/]+.py"  # all .py files in current folder
    "./heflp/setup.py"
    "./src/fl-server.py"
)

verbose=0

# Default destination folder is the current folder
destination_folder=$(pwd)
current_dir=$(pwd)
heflp_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Function to display script usage
usage() {
    echo "Usage: $0 [-o <destination_folder>] Optional[-v verbose]"
    exit 1
}

# Parse command-line arguments
while getopts ":o:v:" opt; do
    case $opt in
        o)
            destination_folder="$OPTARG"
            ;;
        v)
            verbose=1
            ;;
        \?)
            echo "Invalid option: -$OPTARG"
            usage
            ;;
        :)
            echo "Option -$OPTARG requires an argument."
            usage
            ;;
    esac
done

# Temporary folder to store the files before packing
tmp_folder=$(mktemp -d)
heflp_server_folder="$tmp_folder/heflp-server"

cd $heflp_dir || exit 1
if [[ $verbose -eq 1 ]]; then
    echo "cd to $heflp_dir"
    echo "Start Copying..."
fi
# Copy each file and folder to the temporary folder
for file_or_folder in "${files_to_copy[@]}"; do
    if [[ $verbose -eq 1 ]]; then
        echo "$(find ./ -type f -regex $file_or_folder)"
    fi
    find ./ -type f -regex $file_or_folder | cpio -pd --quiet $heflp_server_folder
done

if [[ $verbose -eq 1 ]];then
    echo "Copying server_requirements.txt"
fi
cp scripts/server_requirements.txt $heflp_server_folder/requirements.txt

if [[ $verbose -eq 1 ]];then
    echo "Packing..."
fi
# Create a .tgz file
cd "$tmp_folder" || exit 1
tar -czf "$destination_folder/heflp-server.tgz" .
cd "$current_dir" || exit 1

# Clean up temporary folder
rm -rf "$tmp_folder"

echo "Files have been copied and packed into heflp-server.tgz in $destination_folder."
