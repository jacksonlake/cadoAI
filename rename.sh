a=0
for i in *.jpg; do
  new=$(printf "cats.%04d.jpg" "$a") #Pad with zeroes
  mv -i -- "$i" "$new"
  let a=a+1
done
