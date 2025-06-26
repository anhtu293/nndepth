for ENV in $(ls)
do
    echo $ENV
    for FILE in $(ls $ENV/Easy)
    do
         unzip -n $ENV/Easy/$FILE
    done
done
