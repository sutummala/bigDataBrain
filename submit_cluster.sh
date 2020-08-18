DATA="/usr/users/tummala/testBigData"
for file in $DATA/*; do
   echo $DATA $(basename $file)
   sbatch grid.sh $DATA $(basename $file)
done

