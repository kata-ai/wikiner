
target=$1

find . -mindepth 1 -maxdepth 1 -mtime -40 | while read f; do cp "${f%.ann}.txt" $target/. ; done
