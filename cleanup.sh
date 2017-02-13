#rm -rf graphs/*
#if false
#then
rm -f *pyc

for f in *_*dat; do
  #echo "$f"

  run=$(echo $f | cut -d "." -f 2)
  runnum=$(echo $f | cut -d "." -f 3)
  echo "$run.""$runnum"
  mkdir -p graphs/"$run"/"$runnum"
  mkdir -p backup_data/"$run"
  cp $f backup_data/"$run"
  cp $f graphs/"$run"/"$runnum"
  #echo "$run"
  #break
  #echo "$run"
done



#if false
#then
#mv "netspeed.dat" "netspeed.$run.dat"
#fi
#mkdir -p graphs/"$run"/1
#mkdir -p graphs/"$run"/2
#mkdir -p graphs/"$run"/3

#for f in 

#mkdir -p graphs/orig/"$run"
#for f in *.dat; do
#  firstpart=$(echo $f | cut -d "." -f 1)
#  mv $f "$firstpart.dat"  
#done
#mv *dat graphs/"$run"
#fi
#if false
#then

#for f in graphs/"$run"/*.dat; do
#  cp $f "$f.orig"
#done

#netspeed ok
#for f in "graphs/"$run"/*.dat"; do

#find graphs/"$run" -type f -print0 | while read -d $'\0' file; do
#  f=$file
#  cp -n $f $f.orig
#done

find graphs/"$run" -type f -name "*.dat" -print0 | while read -d $'\0' file; do
#for f in $files; do
  f=$file
  echo $f
  if [[ $f == *"vmstat_"* ]]
  then
    echo $f
    sed -i 's/procs ---------------memory-------------- ---swap-- -----io---- -system-- ------cpu-----//g' $f
    sed -i 's/ r  b     swpd     free    inact   active   si   so    bi    bo   in   cs us sy id wa st//g' $f
    sed -i '/^\s*$/d' $f
    sed -i 's/ /,/g' $f
    sed -i 's/,,/,/g' $f
    sed -i 's/,,,/,/g' $f
    sed -i 's/,,,,/,/g' $f
    sed -i 's/,,,,/,/g' $f
    sed -i 's/,,,,,/,/g' $f
    sed -i 's/,,,,,,/,/g' $f
    sed -i 's/,,/,/g' $f
    sed -i 's/^.//' $f
    sed -i 's/z/0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0/' $f
  fi
  if [[ $f == *"time_ryu"* ]]
  then
    echo $f
    sed -i 's/b_//g' $f
    awk -F, '{a[$3]+=$4;}END{for(i in a)print i", "a[i];}' $f >> $f".inter"
    mv $f".inter" $f
    sed -i 's/f//g' $f
  fi
  if [[ $f == *"time_sync"* ]]
  then
    echo $f
    sed -i '/all/d' $f
    sed -i '/Sync/d' $f
  fi
done
#fi
#>graphs/"$run"/times.dat
#awk -F ',' '{sum += $4} END {print sum,sum/60}' graphs/"$run"/time_sync_cv.dat >> graphs/"$run"/times.dat
#awk -F ',' '{sum += $4} END {print sum,sum/60}' graphs/"$run"/time_sync_pi.dat >> graphs/"$run"/times.dat
#awk -F ',' '{sum += $4} END {print sum,sum/60}' graphs/"$run"/time_sync_pii.dat >> graphs/"$run"/times.dat

echo "done."
