set -x

NAME=COEN_145_GROUP_PRJ

rm work.zip
zip work.zip *.py *.h *.c Makefile *.cpp *.cl
scp work.zip coen-linux:work.zip
ssh coen-linux "scp work.zip wave1:$NAME.zip && ssh wave1 \"unzip -o $NAME.zip -d $NAME\""