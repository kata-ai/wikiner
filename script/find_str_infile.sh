#!/bin/bash

entity=$1
folder=$2

egrep -lir "*$entity*" $2
