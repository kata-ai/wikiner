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
    if (index($2, "B-OTH")){
        sub("B-OTH", "B-MISC", $2)
        print $1, $2
    } else if (index($2, "I-OTH")){
        sub("I-OTH", "I-MISC", $2)
        print $1, $2
    }
    # else if (index($2, "B-Organisation")){
    #     sub("B-Organisation", "B-ORG", $2)
    #     print $1, $2
    # } else if (index($2, "I-Organisation")){
    #     sub("I-Organisation", "I-ORG", $2)
    #     print $1, $2
    # } else if (index($2, "B-Place")){
    #     sub("B-Place", "B-LOC", $2)
    #     print $1, $2
    # } else if (index($2, "I-Place")){
    #     sub("I-Place", "I-LOC", $2)
    #     print $1, $2
    # } 
    else {
        print
    }
}