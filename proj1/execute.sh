#!/usr/bin/env bash
help_msg='Usage: execute.sh [-r] -i query-file -o ranked-list -m model-dir -d NTCIR-dir

OPTIONS:
  -r                  Enable the relevance feedback.
  -i QUERY-FILE       The input query file.
  -o RANKED-LIST      The output ranked list file.
  -m MODEL-DIR        The input model directory, which includes MODEL-DIR/vocab.all, MODEL-DIR/file-list, MODEL-DIR/inverted-index
  -d NTCIR-DIR        The path of the NTCIR documents.
  -h, --help          Show this message and exit.'
TEMP=$(getopt -o ri:o:m:d:h --long ,,,,,help -n "$help_msg" -- "$@")
if [ $? != 0 ] ; then exit 1 ; fi

eval set -- "$TEMP"

relevance=0
query_file=''
ranked_list=''
model_dir=''
ntcir_dir=''
while true; do
    case "$1" in
        -r) relevance=1; shift ;;
        -i) query_file="$2"; shift 2 ;;
        -o) ranked_list="$2"; shift 2 ;;
        -m) model_dir="$2"; shift 2 ;;
        -d) ntcir_dir="$2"; shift 2 ;;
		-h|--help) echo "$help_msg"; exit;;
        --) shift ; break ;;
        *) echo "Internal error!" ; exit 1 ;;
    esac
done

if [ $relevance == 1 ]; then
	python3 main.py -r -q "$query_file" -o "$ranked_list" -m "$model_dir" -d "$ntcir_dir"
else
	python3 main.py -q "$query_file" -o "$ranked_list" -m "$model_dir" -d "$ntcir_dir"
fi

