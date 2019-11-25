#!/usr/bin/awk -f

# This script expects the data is in CoNLL format with 2 columns. 
# First column is the words, second column is the ner tags.

BEGIN {
    FS="\t"
    OFS="\t"
    IGNORECASE=1
}
{   
    # retag B-Person
    # retag I-Person
    # retag B-Organisation
    # retag I-Organisation
    # retag B-Place
    # retag I-Place
    if (index($2, "B-PER")){
        sub("B-PER", "B-Person", $2)
        print $1, $2
    } else if (index($2, "I-PER")){
        sub("I-PER", "I-Person", $2)
        print $1, $2
    } else if (index($2, "B-ORG")){
        sub("B-ORG", "B-Organisation", $2)
        print $1, $2
    } else if (index($2, "I-ORG")){
        sub("I-ORG", "I-Organisation", $2)
        print $1, $2
    } else if (index($2, "B-LOC")){
        sub("B-LOC", "B-Place", $2)
        print $1, $2
    } else if (index($2, "I-LOC")){
        sub("I-LOC", "I-Place", $2)
        print $1, $2
    } else {
        print
    }
}